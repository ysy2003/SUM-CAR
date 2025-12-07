"""
Sparse Finetuner
For task-specific memory slot finetuning and patch generation
Supports GPU/CUDA training
"""
import math
import os
from typing import Dict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm

from ..models.base_model import MemoryAugmentedCausalLM
from ..models.lora_memory_model import LoRAMemoryAugmentedCausalLM
from .kv_memory import KVMemoryLayer
from .metrics import choose_top_t_from_counts, SpecificityTracker, mask_kv_grads
from ..utils.io import ensure_dir, dump_json
from ..utils.logger import Logger
from ..utils.checkpoint import CheckpointManager, TrainingState


class SparseFinetuner:
    """
    Sparse Finetuner

    Features:
    1. Probe memory slot usage on task data
    2. Select top-t slots for finetuning
    3. Export task-specific patches
    4. Support GPU/CUDA training
    """

    def __init__(self, base_model: str, mem_cfg: Dict, train_cfg: Dict, tokenizer=None, logger=None):
        """
        Args:
            base_model: Base model name or path
            mem_cfg: Memory config dict {'num_slots', 'k_top', 'alpha', 'tau'}
            train_cfg: Training config dict {'lr', 'use_lora', 'lora_config', ...}
            tokenizer: Tokenizer (optional)
            logger: Logger (optional)
        """
        self.logger = logger or Logger('[train]')
        self.tok = tokenizer or AutoTokenizer.from_pretrained(base_model)

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        # Infer model dimension
        d_model = self._infer_d_model(base_model)

        # Create memory layer (enhanced version with gating)
        self.mem = KVMemoryLayer(
            d_model=d_model,
            num_slots=mem_cfg['num_slots'],
            k_top=mem_cfg['k_top'],
            alpha=mem_cfg.get('alpha', 1.0),
            tau=mem_cfg.get('tau', 10.0),
            use_gate=mem_cfg.get('use_gate', True),
            normalize_retrieval=mem_cfg.get('normalize_retrieval', True),
            track_hits=mem_cfg.get('track_hits', None),
            hits_source=mem_cfg.get('hits_source', 'topk'),
            track_interval=mem_cfg.get('track_interval', 200)
        )

        # Create augmented model (with LoRA support and FP16)
        use_lora = train_cfg.get('use_lora', False)
        use_fp16 = train_cfg.get('use_fp16', False)

        if use_fp16:
            self.logger.log('Using FP16 precision')
            # Convert memory layer to FP16 to match model dtype
            self.mem = self.mem.half()

        # Memory position: 'embedding' or 'middle'
        memory_position = mem_cfg.get('memory_position', 'embedding')
        self.logger.log(f'Memory position: {memory_position}')

        if use_lora:
            self.logger.log('Using LoRA + Memory augmented model')
            lora_config = train_cfg.get('lora_config', {})
            self.model = LoRAMemoryAugmentedCausalLM(base_model, self.mem, lora_config, use_fp16=use_fp16)
        else:
            self.logger.log('Using standard Memory augmented model')
            self.model = MemoryAugmentedCausalLM(base_model, self.mem, use_fp16=use_fp16, memory_position=memory_position)

        self.cfg = train_cfg

        # Specificity tracker (with task name)
        task_name = train_cfg.get('dataset', 'unknown')
        self.spec_tracker = SpecificityTracker(
            M=mem_cfg['num_slots'],
            task_name=task_name
        )
    
    def _infer_d_model(self, base_model: str) -> int:
        """
        Infer the model's hidden dimension

        Args:
            base_model: Model name or path

        Returns:
            Hidden dimension size
        """
        tmp = AutoModelForCausalLM.from_pretrained(base_model)
        return tmp.get_input_embeddings().weight.shape[1]
    
    def probe(self, dl: DataLoader, steps: int = 1000, device: str = None):
        """
        Probe phase: Collect slot access statistics

        Args:
            dl: DataLoader
            steps: Maximum probe steps
            device: Device (auto-select by default)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        self.model.eval()
        self.mem.enable_access_logging()

        cnt = 0
        with torch.no_grad():
            probe_pbar = tqdm(total=steps, desc="Probing memory slots", unit="batch")
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.model(**batch)

                # Update specificity tracking
                if cnt % 50 == 0:
                    # Trigger lightweight statistics (based on k_top only, CPU side)
                    self.mem.maybe_collect_hits_light()
                    hits = self.mem.pop_last_hits()
                    if hits is not None:
                        self.spec_tracker.update_from_hits(hits)

                cnt += 1
                probe_pbar.update(1)
                if cnt >= steps:
                    break
            probe_pbar.close()
        
        self.mem.disable_access_logging()
        total_accesses = int(self.mem.acc_counts.sum().item())
        self.logger.log(f'Probe done; counted {total_accesses} accesses')

        # Sync mem.acc_counts to spec_tracker (in case track_hits is disabled)
        if total_accesses > 0:
            # Find all accessed slots
            accessed_slots = (self.mem.acc_counts > 0).nonzero(as_tuple=True)[0]
            if len(accessed_slots) > 0:
                # Simulate hits format: repeat each slot ID access_count times
                # But for efficiency, we directly update tf_counts
                for slot_id in accessed_slots:
                    count = int(self.mem.acc_counts[slot_id].item())
                    self.spec_tracker.tf_counts[slot_id] = count
                self.logger.log(f'Synced {len(accessed_slots)} accessed slots to spec_tracker')
    
    def train(self, dl: DataLoader, epochs: int = 1, device: str = None, refresh_every: int = 200):
        """
        Training phase: Finetune selected slots

        Args:
            dl: DataLoader
            epochs: Number of training epochs
            device: Device (auto-select by default)
            refresh_every: How often to refresh top-t slots
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        self.model.train()

        # Gradient accumulation
        grad_accum_steps = self.cfg.get('gradient_accumulation_steps', 1)

        # Optimizer: Separate KV parameters and gate parameters
        kv_params = [self.mem.keys, self.mem.vals]
        gate_params = list(self.mem.W_q.parameters())
        if self.mem.use_gate:
            gate_params.extend(list(self.mem.gate.parameters()))

        opt = torch.optim.AdamW([
            {'params': kv_params, 'lr': self.cfg.get('lr_kv', self.cfg['lr'])},
            {'params': gate_params, 'lr': self.cfg.get('lr_gate', self.cfg['lr'] * 0.1)},
        ])

        steps_per_epoch = len(dl)
        total_steps = (steps_per_epoch * epochs) // grad_accum_steps
        sch = get_linear_schedule_with_warmup(opt, 0, total_steps)

        # Loss history tracking
        loss_history = []

        # Best checkpoint tracking
        best_loss = float('inf')
        best_mem_state = None

        step = 0
        accum_loss = 0.0
        for ep in range(epochs):
            epoch_pbar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}", unit="batch")
            for batch_idx, batch in enumerate(epoch_pbar):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss / grad_accum_steps  # Scale loss for accumulation
                accum_loss += loss.item()

                loss.backward()

                # Sparse gradient masking
                mask_kv_grads(self.mem, device)

                # Optimizer step only after accumulation
                if (batch_idx + 1) % grad_accum_steps == 0:
                    step += 1

                    # Compute norm of activated memory slots
                    trainable_slots = self.mem.trainable_slots if hasattr(self.mem, 'trainable_slots') and self.mem.trainable_slots else []
                    if trainable_slots:
                        slot_ids = list(trainable_slots)
                        keys_norm = self.mem.keys[slot_ids].norm().item()
                        vals_norm = self.mem.vals[slot_ids].norm().item()
                    else:
                        keys_norm = self.mem.keys.norm().item()
                        vals_norm = self.mem.vals.norm().item()

                    # Record loss (accumulated)
                    loss_history.append({
                        'step': step,
                        'epoch': ep + 1,
                        'loss': accum_loss,
                        'keys_norm': keys_norm,
                        'vals_norm': vals_norm
                    })

                    # Track best checkpoint (save memory state)
                    if accum_loss < best_loss:
                        best_loss = accum_loss
                        # Deep copy memory state (keys and values)
                        best_mem_state = {
                            'keys': self.mem.keys.detach().clone().cpu(),
                            'values': self.mem.vals.detach().clone().cpu(),
                            'step': step,
                            'epoch': ep + 1,
                            'loss': accum_loss
                        }

                    # Optimizer step
                    opt.step()
                    sch.step()
                    opt.zero_grad(set_to_none=True)

                    # Update progress bar with loss and memory norms
                    epoch_pbar.set_postfix({
                        "loss": f"{accum_loss:.4f}",
                        "K_norm": f"{keys_norm:.2f}",
                        "V_norm": f"{vals_norm:.2f}",
                        "step": step
                    })
                    accum_loss = 0.0

                # Update specificity (collect statistics every refresh_every steps)
                if step % refresh_every == 0:
                    # Trigger lightweight statistics (based on k_top only, CPU side)
                    self.mem.maybe_collect_hits_light()
                    hits = self.mem.pop_last_hits()
                    if hits is not None:
                        self.spec_tracker.update_from_hits(hits)
                    else:
                        # If hits not available (track_hits=false), sync from acc_counts
                        accessed_slots = (self.mem.acc_counts > 0).nonzero(as_tuple=True)[0]
                        for slot_id in accessed_slots:
                            count = int(self.mem.acc_counts[slot_id].item())
                            self.spec_tracker.tf_counts[slot_id] = count

                    # Refresh top-t trainable slots
                    top_t = self.cfg.get('top_t', 2048)
                    top_slots = self.spec_tracker.top_t(top_t).to(device)
                    self.mem.set_trainable_slots(top_slots.tolist())
                    self.logger.log(f'[step {step}] refresh Top-t={len(top_slots)}; loss={loss.item():.4f}')

            self.logger.log(f'Epoch {ep + 1}/{epochs} done')

        # Restore best checkpoint
        if best_mem_state is not None:
            self.logger.log(f'Restoring best checkpoint from step {best_mem_state["step"]} (epoch {best_mem_state["epoch"]}) with loss {best_mem_state["loss"]:.4f}')
            self.mem.keys.data = best_mem_state['keys'].to(device)
            self.mem.vals.data = best_mem_state['values'].to(device)
        else:
            self.logger.log('Warning: No best checkpoint found, using final state')

        return loss_history

    def build_patch(self, task: str, top_t: int, out_dir: str, train_stats: Dict = None, use_ckpt_manager: bool = True, loss_history: list = None) -> Dict:
        """
        Build and save task patch (using specificity scores)

        Args:
            task: Task name
            top_t: Number of slots to select
            out_dir: Output directory
            train_stats: Training statistics (optional)
            use_ckpt_manager: Whether to use new checkpoint manager

        Returns:
            Patch dictionary
        """
        # Use specificity tracker to select top-t slots
        slot_ids = self.spec_tracker.top_t(top_t, total_tasks=1).tolist()

        # Calculate specificity scores (TF-IDF, normalized to [0,1])
        specificity_all = self.spec_tracker.specificity(total_tasks=1, normalize=True)
        specificity = specificity_all[slot_ids].tolist()

        # Get statistics
        spec_stats = self.spec_tracker.get_stats()

        # Slot access statistics
        access_counts = self.spec_tracker.get_access_counts()
        access_counts_list = [int(access_counts[sid]) for sid in slot_ids]

        # Get K/V tensors
        keys = self.mem.keys[slot_ids]
        values = self.mem.vals[slot_ids]

        # If using new checkpoint manager
        if use_ckpt_manager and hasattr(self, 'ckpt_manager'):
            # Memory configuration
            memory_config = {
                'num_slots': self.mem.num_slots,
                'd_model': self.mem.d_model,
                'k_top': self.mem.k_top,
                'alpha': self.mem.alpha,
                'tau': self.mem.tau,
                'use_gate': self.mem.use_gate,
            }

            # Statistics
            stats = {
                'access_total': spec_stats['total_accesses'],
                'unique_slots_accessed': spec_stats['unique_slots_accessed'],
                'specificity_stats': {
                    'max': spec_stats['spec_max'],
                    'min': spec_stats['spec_min'],
                    'mean': spec_stats['spec_mean'],
                    'std': spec_stats['spec_std'],
                    'top_t_min': float(specificity_all[slot_ids].min().item()),
                },
                'access_stats': {
                    'total': spec_stats['total_accesses'],
                    'max': spec_stats['max_access'],
                    'unique_accessed': spec_stats['unique_slots_accessed'],
                    'top_t_total': sum(access_counts_list),
                },
                # IDF information (for reproducibility)
                'idf_df_counts': self.spec_tracker.idf_df_counts,
            }

            # Save using checkpoint manager
            self.ckpt_manager.save_patch(
                slot_ids=slot_ids,
                keys=keys,
                values=values,
                specificity=specificity,
                access_counts=access_counts_list,
                memory_config=memory_config,
                train_meta=train_stats or {},
                stats=stats
            )

        # Also save old format (for compatibility)
        patch = self.mem.get_patch(slot_ids)
        patch['task'] = task
        patch['specificity'] = specificity
        patch['access_counts'] = access_counts_list

        # Metadata
        meta = {
            'task': task,
            'top_t': top_t,
            'num_slots': self.mem.num_slots,

            # TF-IDF statistics
            'specificity_stats': {
                'max': spec_stats['spec_max'],
                'min': spec_stats['spec_min'],
                'mean': spec_stats['spec_mean'],
                'std': spec_stats['spec_std'],
                'top_t_min': float(specificity_all[slot_ids].min().item()),
            },

            # Access statistics
            'access_stats': {
                'total': spec_stats['total_accesses'],
                'max': spec_stats['max_access'],
                'unique_accessed': spec_stats['unique_slots_accessed'],
                'top_t_total': sum(access_counts_list),
            },

            # IDF information (for reproducibility)
            'idf_df_counts': self.spec_tracker.idf_df_counts,
        }

        # Add loss history
        if loss_history:
            meta['loss_history'] = loss_history
            # Calculate loss statistics
            losses = [x['loss'] for x in loss_history]
            meta['loss_stats'] = {
                'initial': losses[0] if losses else None,
                'final': losses[-1] if losses else None,
                'min': min(losses) if losses else None,
                'max': max(losses) if losses else None,
                'mean': sum(losses) / len(losses) if losses else None,
            }

        # Merge training statistics
        if train_stats:
            meta.update(train_stats)

        # Save old format
        ensure_dir(out_dir)
        dump_json(patch, os.path.join(out_dir, f'patch_{task}.json'))
        dump_json(meta, os.path.join(out_dir, f'patch_{task}_meta.json'))
        
        return patch
