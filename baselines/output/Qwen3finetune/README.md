# Qwen3-8B Fine-tuning Pipeline

This directory contains scripts for training and merging memory patches using **Qwen3-8B** as the base model, with thinking mode support.

## Directory Structure

```
test_tpu/
├── Qwen3finetune/                    # This directory
│   ├── run_finetune_all_tasks_qwen3.sh   # Training script
│   ├── run_merge_qwen3.sh                # Merging script
│   ├── README.md                          # This file
│   └── configs/                           # Qwen3-specific configs
│       ├── train_math_cot.yaml
│       ├── train_code.yaml
│       └── train_finqa_cot.yaml
│
├── out_qwen3/                        # Output directory (created after training)
│   ├── math_cot/
│   │   └── patch_gsm8k.json
│   ├── code/
│   │   └── patch_codexglue.json
│   ├── finqa_cot/
│   │   └── patch_finqa.json
│   └── merged/
│       ├── memory.pt
│       ├── remap.json
│       └── patch_meta.json
```

## Usage

### Step 1: Train on All Tasks

Train memory patches for Math, Code, and FinQA tasks on Qwen3-8B:

```bash
# From project root (test_tpu/)
bash Qwen3finetune/run_finetune_all_tasks_qwen3.sh
```

**What it does:**
- Trains on **Qwen/Qwen3-8B** base model
- Tasks:
  1. Math (GSM8K) with Chain-of-Thought
  2. Code (CodeXGLUE)
  3. FinQA with Chain-of-Thought
- Outputs to: `out_qwen3/` (separate from Llama outputs in `out/`)

### Step 2: Merge Patches

Merge the three task-specific patches into a unified memory:

```bash
# From project root (test_tpu/)
bash Qwen3finetune/run_merge_qwen3.sh
```

**What it does:**
- Merges patches using TF-IDF scoring and capacity budgeting
- Outputs to: `out_qwen3/merged/`
- Creates:
  - `memory.pt` - Unified memory module
  - `remap.json` - Task routing table
  - `patch_meta.json` - Merge statistics

### Step 3: Evaluate (Optional)

Evaluate the merged Qwen3 model:

```bash
# From project root (test_tpu/)
python scripts/eval_merged.py \
    --base_model Qwen/Qwen3-8B \
    --merged_dir out_qwen3/merged \
    --out scripts/qwen3_merged_results.json \
    --max_samples 100
```

## Key Differences from Llama Pipeline

| Feature | Llama-3-8B | Qwen3-8B |
|---------|------------|----------|
| Base Model | `meta-llama/Meta-Llama-3-8B-Instruct` | `Qwen/Qwen3-8B` |
| Thinking Mode | No | Yes (native support) |
| Output Directory | `out/` | `out_qwen3/` |
| Scripts Location | Root directory | `Qwen3finetune/` |

## Configuration

Both scripts use the same configurations as the Llama pipeline but override:
- `--base_model`: Changed to `Qwen/Qwen3-8B`
- `--out_dir`: Changed to `out_qwen3/`

## Configuration Files

The config files in `Qwen3finetune/configs/` specify:
- `base_model: Qwen/Qwen3-8B` - Uses Qwen3 instead of Llama
- `save_dir: out_qwen3/` - Outputs to separate directory
- All other settings match Llama configuration for fair comparison

### Key Differences from Llama Configs:
- **Base Model**: `Qwen/Qwen3-8B` vs `meta-llama/Meta-Llama-3-8B-Instruct`
- **Output Dir**: `out_qwen3/` vs `out/`
- **Thinking Mode**: Qwen3 has native support, enabled via chat templates

## Notes

- Qwen3-8B has built-in thinking mode support via chat templates with `enable_thinking=True`
- Memory parameters: `k_top=8`, `alpha=1.0`, `num_slots=4096` (matches Llama configuration)
- LoRA is enabled for both models to reduce memory usage during training
