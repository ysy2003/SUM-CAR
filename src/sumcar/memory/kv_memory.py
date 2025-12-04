"""
KV Memory Layer for SUM-CAR
Memory layer based on key-value retrieval, supporting top-k routing and sparse updates
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class KVMemoryLayer(nn.Module):
    """
    Key-Value Memory Layer

    Features:
    - Top-k retrieval mechanism
    - Access count statistics
    - Patch export/application
    - Dynamic slot expansion
    - Sparse training support
    """
    
    def __init__(
        self,
        d_model: int,
        num_slots: int = 200000,
        k_top: int = 32,
        alpha: float = 1.0,
        log_access: bool = True,
        tau: float = 10.0,
        use_gate: bool = True,
        normalize_retrieval: bool = True,
        track_hits: Optional[bool] = None,
        hits_source: str = "topk",
        track_interval: int = 200
    ):
        """
        Args:
            d_model: Model dimension
            num_slots: Number of memory slots
            k_top: Top-k retrieval count
            alpha: Output scaling factor
            log_access: Whether to log access statistics
            tau: Temperature parameter (for softmax)
            use_gate: Whether to use gating mechanism
            normalize_retrieval: Whether to normalize query and key
            track_hits: Whether to track hits (None=defaults to True)
            hits_source: Hit source "topk" or "none"
            track_interval: Statistics interval steps
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        self.k_top = k_top
        self.alpha = alpha
        self._log_access = log_access
        self.tau = tau
        self.normalize_retrieval = normalize_retrieval

        # Hit statistics control
        if track_hits is None:
            track_hits = True
        self.track_hits = track_hits
        self.hits_source = hits_source
        self.track_interval = track_interval
        self._step = 0
        self._last_hits_cpu = None  # Hit statistics stored on CPU

        # Keys and Values
        self.keys = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.vals = nn.Parameter(torch.zeros(num_slots, d_model))

        # Query projection (for better retrieval)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.W_q.weight)  # Initialize to identity matrix

        # Gating mechanism (scalar gating)
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Linear(d_model, 1, bias=True)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)  # Initial gate is very small

        # Access counts (not involved in gradient)
        self.register_buffer('acc_counts', torch.zeros(num_slots, dtype=torch.long))

        # Trainable mask (for sparse training)
        self.register_buffer('_trainable_mask', torch.zeros(num_slots, dtype=torch.bool))

        # Recently hit slots (for specificity tracking)
        self._last_hits = None
    
    def forward(self, x, record_hits: bool = True):
        """
        Forward pass

        Args:
            x: [B, L, D] input tensor
            record_hits: Whether to record hits (for specificity tracking)

        Returns:
            [B, L, D] output tensor
        """
        # x: [B, L, D]
        B, L, D = x.shape

        # Query projection
        q = self.W_q(x)  # [B, L, D]

        # Normalization (if enabled)
        if self.normalize_retrieval:
            q = F.normalize(q, dim=-1)
            keys = F.normalize(self.keys, dim=-1)
        else:
            keys = self.keys

        # Calculate similarity [B, L, M]
        scores = torch.einsum('bld,md->blm', q, keys) / (D ** 0.5)

        # Top-k retrieval (use int32 indices to reduce memory)
        topv, topi_i64 = torch.topk(scores, k=min(self.k_top, self.num_slots), dim=-1)  # [B, L, K]
        topi = topi_i64.to(torch.int32)  # Reduce memory usage

        # Attention weights (with temperature)
        attn = F.softmax(topv / self.tau, dim=-1)  # [B, L, K]

        # Get selected values
        sel_vals = self.vals[topi]  # [B, L, K, D]

        # Weighted sum
        out = torch.einsum('blk,blkd->bld', attn, sel_vals)  # [B, L, D]

        # Gating fusion
        if self.use_gate:
            g = torch.sigmoid(self.gate(x))  # [B, L, 1]
            out = g * out

        # Record access statistics (count only, no CPU copying)
        if self._log_access:
            with torch.no_grad():
                flat = topi.reshape(-1).to(torch.int64)  # bincount requires int64
                binc = torch.bincount(flat, minlength=self.num_slots)
                self.acc_counts += binc.to(self.acc_counts.device)

        # Save small k_top indices for external statistics (no CPU copying here)
        if self.training and record_hits and self.track_hits:
            # Only save reference, no CPU operations
            self.last_k_indices = topi.detach()  # [B, L, K], very small
        
        self._step += 1
        
        return self.alpha * out
    
    def get_patch(self, slot_ids: List[int]) -> Dict:
        """
        Export patch

        Args:
            slot_ids: List of slot IDs to export

        Returns:
            Dictionary containing slot_ids, keys, vals, access_counts
        """
        with torch.no_grad():
            return {
                'slot_ids': [int(i) for i in slot_ids],
                'keys': self.keys[slot_ids].cpu().tolist(),
                'vals': self.vals[slot_ids].cpu().tolist(),
                'access_counts': [int(self.acc_counts[i]) for i in slot_ids]
            }
    
    @torch.no_grad()
    def apply_patch(self, patch: Dict):
        """
        Apply patch

        Args:
            patch: Dictionary containing slot_ids, keys, vals
        """
        for sid, k, v in zip(patch['slot_ids'], patch['keys'], patch['vals']):
            self.keys[sid] = torch.tensor(k, device=self.keys.device, dtype=self.keys.dtype)
            self.vals[sid] = torch.tensor(v, device=self.vals.device, dtype=self.vals.dtype)
    
    @torch.no_grad()
    def expand_slots(self, add_n: int):
        """
        Dynamically expand slots

        Args:
            add_n: Number of slots to add
        """
        if add_n <= 0:
            return

        device, dtype = self.keys.device, self.keys.dtype

        # Create new keys and vals
        new_k = torch.randn(add_n, self.d_model, device=device, dtype=dtype) * 0.02
        new_v = torch.zeros(add_n, self.d_model, device=device, dtype=dtype)

        # Concatenate
        self.keys = nn.Parameter(torch.cat([self.keys, new_k], dim=0))
        self.vals = nn.Parameter(torch.cat([self.vals, new_v], dim=0))

        # Expand access counts and mask
        self.acc_counts = torch.cat([
            self.acc_counts,
            torch.zeros(add_n, dtype=torch.long, device=device)
        ], dim=0)
        
        self._trainable_mask = torch.cat([
            self._trainable_mask,
            torch.zeros(add_n, dtype=torch.bool, device=device)
        ])
        
        self.num_slots += add_n
    
    def set_trainable_slots(self, slot_ids: List[int]):
        """
        Set trainable slots (for sparse training)

        Args:
            slot_ids: List of trainable slot IDs
        """
        self._trainable_mask.zero_()
        self._trainable_mask[slot_ids] = True

    def get_trainable_slots(self) -> List[int]:
        """
        Get current trainable slot IDs

        Returns:
            List of slot IDs
        """
        return torch.where(self._trainable_mask)[0].tolist()

    def reset_access_counts(self):
        """Reset access counts"""
        self.acc_counts.zero_()

    def get_access_counts(self) -> torch.Tensor:
        """
        Get access counts

        Returns:
            [num_slots] access count tensor
        """
        return self.acc_counts.clone()

    def enable_access_logging(self):
        """Enable access statistics"""
        self._log_access = True

    def disable_access_logging(self):
        """Disable access statistics"""
        self._log_access = False
    
    @torch.no_grad()
    def maybe_collect_hits_light(self):
        """
        Low-frequency, small-volume statistics: based only on k_top indices (very small), and only on CPU.
        Avoid triggering large memory allocations during training.
        """
        if not self.track_hits or self.hits_source == "none":
            self._last_hits_cpu = None
            return

        # Only collect at specified step intervals
        if (self._step % self.track_interval) != 0:
            self._last_hits_cpu = None
            return

        # Check if k_indices are available
        if not hasattr(self, 'last_k_indices') or self.last_k_indices is None:
            self._last_hits_cpu = None
            return

        try:
            # CPU-side statistics throughout (small volume: k_top ≪ top_t), won't use much memory
            k_indices = self.last_k_indices
            hit_ids = k_indices.to('cpu').reshape(-1).to(torch.int64)
            counts = torch.bincount(hit_ids, minlength=self.num_slots)
            unique_ids = torch.nonzero(counts, as_tuple=False).reshape(-1).to(torch.int64)
            self._last_hits_cpu = unique_ids  # Keep on CPU
        except Exception as e:
            # If statistics fail, silently ignore (training continues)
            self._last_hits_cpu = None
        finally:
            # Clean up reference
            self.last_k_indices = None

    def pop_last_hits(self):
        """Get and clear recently hit slots (CPU version)"""
        hits = self._last_hits_cpu
        self._last_hits_cpu = None
        return hits

    def freeze_all(self):
        """Freeze all parameters"""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_slots(self, slot_ids: List[int]):
        """Unfreeze specified slots (for sparse training)"""
        self.set_trainable_slots(slot_ids)
        # Note: keys and vals are always trainable, controlled by gradient mask
        for p in self.parameters():
            p.requires_grad = True

    def top_slots(self, t: int) -> List[int]:
        """Get top t slots with highest access frequency"""
        return torch.topk(self.acc_counts, k=min(t, self.num_slots)).indices.tolist()

    def extra_repr(self) -> str:
        """Extra representation info"""
        return (f'd_model={self.d_model}, num_slots={self.num_slots}, '
                f'k_top={self.k_top}, alpha={self.alpha}, tau={self.tau}')
