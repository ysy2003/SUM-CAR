# Qwen3 Fine-tuning Setup Complete ✓

## What Was Done

Fixed the issue where command-line arguments (`--base_model`, `--out_dir`) were not supported by `train_task.py`.

### Solution

Created **Qwen3-specific configuration files** that specify the base model and output directories directly in YAML:

```
Qwen3finetune/
├── configs/
│   ├── train_math_cot.yaml      # base_model: Qwen/Qwen3-8B, save_dir: out_qwen3/math_cot
│   ├── train_code.yaml          # base_model: Qwen/Qwen3-8B, save_dir: out_qwen3/code
│   └── train_finqa_cot.yaml     # base_model: Qwen/Qwen3-8B, save_dir: out_qwen3/finqa_cot
├── run_finetune_all_tasks_qwen3.sh   # Updated to use Qwen3finetune/configs/
├── run_merge_qwen3.sh
└── README.md                     # Updated documentation
```

## Usage

### Train on Qwen3-8B:

```bash
# From test_tpu/ directory
bash Qwen3finetune/run_finetune_all_tasks_qwen3.sh
```

**What it does:**
- Task 1: Math (GSM8K) with CoT → `out_qwen3/math_cot/`
- Task 2: Code (CodeXGLUE) → `out_qwen3/code/`
- Task 3: FinQA with CoT → `out_qwen3/finqa_cot/`

### Merge Qwen3 patches:

```bash
bash Qwen3finetune/run_merge_qwen3.sh
```

**Outputs:** `out_qwen3/merged/`

## Configuration Details

Each Qwen3 config file specifies:

```yaml
base_model: Qwen/Qwen3-8B               # ← Qwen3 model
mem:
  num_slots: 4096
  k_top: 8
  alpha: 1.0
  # ... (same as Llama)
train:
  save_dir: out_qwen3/math_cot          # ← Training logs directory
  patch_output_dir: out_qwen3           # ← Patch files directory (NEW!)
  # ... (same settings as Llama for fair comparison)
```

### Important: Output Directory Parameters

To prevent overwriting Llama files, we added two parameters to all Qwen3 configs:

1. **`patch_output_dir: out_qwen3`** - For JSON patch files
   - **Llama patches** → `out/patch_gsm8k.json`, `out/patch_codexglue.json`, `out/patch_finqa.json`
   - **Qwen3 patches** → `out_qwen3/patch_gsm8k.json`, `out_qwen3/patch_codexglue.json`, `out_qwen3/patch_finqa.json`

2. **`checkpoint_base_dir: out_qwen3_ckpt`** - For checkpoint directories
   - **Llama checkpoints** → `patches/`, `merges/`, `runs/`
   - **Qwen3 checkpoints** → `out_qwen3_ckpt/patches/`, `out_qwen3_ckpt/merges/`, `out_qwen3_ckpt/runs/`

## Directory Structure After Training

```
test_tpu/
├── out/                           # Llama-3-8B outputs (unchanged)
│   ├── math_cot/
│   ├── code/
│   ├── finqa_cot/
│   ├── patch_gsm8k.json
│   ├── patch_codexglue.json
│   ├── patch_finqa.json
│   └── merged/
│
├── patches/                       # Llama-3-8B checkpoint patches (unchanged)
│   ├── patch_gsm8k_*.pt
│   ├── patch_codexglue_*.pt
│   └── patch_finqa_*.pt
│
├── out_qwen3/                     # Qwen3-8B patch JSONs (new)
│   ├── math_cot/
│   ├── code/
│   ├── finqa_cot/
│   ├── patch_gsm8k.json
│   ├── patch_codexglue.json
│   ├── patch_finqa.json
│   └── merged/
│
├── out_qwen3_ckpt/                # Qwen3-8B checkpoints (new)
│   ├── patches/                   # Tensor checkpoints
│   ├── merges/                    # Merge artifacts
│   └── runs/                      # Training checkpoints
│
└── Qwen3finetune/                 # Qwen3-specific scripts & configs
    ├── configs/
    ├── run_finetune_all_tasks_qwen3.sh
    ├── run_merge_qwen3.sh
    └── README.md
```

## What's Different from the Error

**Before (caused error):**
```bash
python -m sumcar.cli.train_task \
    --config configs/train_math_cot.yaml \
    --base_model Qwen/Qwen3-8B \        # ← Not supported!
    --out_dir out_qwen3/math_cot \      # ← Not supported!
    --use_xla False
```

**After (now works):**
```bash
python -m sumcar.cli.train_task \
    --config Qwen3finetune/configs/train_math_cot.yaml \  # ← Config has base_model & save_dir
    --use_xla False                                        # ← Only supported args
```

## Next Steps

1. **Train:** Run `bash Qwen3finetune/run_finetune_all_tasks_qwen3.sh`
2. **Merge:** Run `bash Qwen3finetune/run_merge_qwen3.sh`
3. **Evaluate:** Use `scripts/eval_merged.py` with `--base_model Qwen/Qwen3-8B --merged_dir out_qwen3/merged`

## Device Usage

✅ The model will automatically use **CUDA** if available (checked via `torch.cuda.is_available()`)
- Model is moved to device via: `self.model.to(device)`
- Batches are moved to device during training
- Falls back to CPU if no CUDA devices found
