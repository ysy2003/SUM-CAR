import os
import json
import fire
import torch
from transformers import AutoModelForCausalLM
from ..memory.kv_memory import KVMemoryLayer


def main(
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    patches: str = None,   # 'noLoRA/patch_finqa.json,noLoRA_math/patch_gsm8k_meta.json'
    out: str = "noLoRA/merged_concat",
    k_top: int = 8,
    alpha: float = 1.0,
    use_fp16: bool = False,
):
    # 1) analyze patches param
    if isinstance(patches, str):
        patch_list = [p.strip() for p in patches.split(",") if p.strip()]
    else:
        patch_list = patches or []
    assert patch_list, "Provide --patches=patch1.json,patch2.json"

    os.makedirs(out, exist_ok=True)

    # 2) 从 base model push d_model
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
    d_model = base.get_input_embeddings().weight.shape[1]

    # 3) read and preprocess all patches
    plist = [json.load(open(p, "r", encoding="utf-8")) for p in patch_list]

    for i, p in enumerate(plist):
        if "task" not in p:
            p["task"] = f"t{i}"

        if "slot_ids" not in p:
            assert "keys" in p, f"{patch_list[i]} has no 'keys'"
            p["slot_ids"] = list(range(len(p["keys"])))

        if "specificity" in p and isinstance(p["specificity"], list) and len(p["specificity"]) > 0:
            spec = torch.tensor(p["specificity"], dtype=torch.float32)
            s_min, s_max = spec.min(), spec.max()
            if (s_max - s_min) > 1e-6:
                spec = (spec - s_min) / (s_max - s_min)
            else:
                spec = torch.full_like(spec, 0.5)
            p["specificity"] = spec.tolist()

        if "usage" in p and isinstance(p["usage"], list) and len(p["usage"]) > 0:
            usage = torch.tensor(p["usage"], dtype=torch.float32)
            u_min, u_max = usage.min(), usage.max()
            if (u_max - u_min) > 1e-6:
                usage = (usage - u_min) / (u_max - u_min)
            else:
                usage = torch.full_like(usage, 0.5)
            p["usage"] = usage.tolist()

    total_slots = sum(len(p["slot_ids"]) for p in plist)

    mem = KVMemoryLayer(
        d_model=d_model,
        num_slots=total_slots,
        k_top=k_top,
        alpha=alpha,
    )
    if use_fp16:
        mem = mem.half()

    current = 0
    for p in plist:
        keys = torch.tensor(p["keys"], dtype=mem.keys.dtype)
        vals = torch.tensor(p["vals"], dtype=mem.vals.dtype)
        n = keys.shape[0]

        mem.keys.data[current:current + n] = keys
        mem.vals.data[current:current + n] = vals

        current += n

    with torch.no_grad():
        key_norms = mem.keys.norm(dim=1, keepdim=True)  # [num_slots, 1]
        key_norms.clamp_(min=1e-6)
        mem.keys.data /= key_norms

    torch.save(
        {
            "keys": mem.keys.detach().cpu(),
            "vals": mem.vals.detach().cpu(),
        },
        os.path.join(out, "memory.pt"),
    )

    patch_meta = {"total_slots": total_slots}
    for p in plist:
        patch_meta[p["task"]] = {
            "slot_ids": p["slot_ids"],
            "n_slots": len(p["slot_ids"]),
        }
    with open(os.path.join(out, "patch_meta.json"), "w") as f:
        json.dump(patch_meta, f, indent=2)

    print(f"[concat+norm] merged slots: {total_slots}")


if __name__ == "__main__":
    fire.Fire(main)
