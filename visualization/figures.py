# figures.py
# Usage:
#   python figures.py \
#       --hits_csv logs/memory_hits.csv \
#       --remap_csv logs/remap_events.csv \
#       --scores_csv logs/scores.csv \
#       --num_slots 4096 --slot_bins 256 --top_flows 30

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, path as mpath

# ------------------------------------------------------------
# Expected CSV schemas (minimal):
#
# 1) memory_hits.csv  (用于 图1：每任务的内存激活热力图)
#    columns: task,batch_id,slot_id,weight
#
# 2) remap_events.csv (用于 图2：冲突与重映射 Sankey)
#    columns: task_from,slot_old,task_to,slot_new,weight
#
# 3) scores.csv       (用于 图3：Retention–Forgetting 散点)
#    columns: model,task,score_single,score_merged
# ------------------------------------------------------------

# 任务顺序和名称，需要与你的 CSV 数据中的 'task' 列完全对应
TASK_ORDER = ["humaneval", "gsm8k", "finqa"]
# 用于图表标题的美化名称
TASK_DISPLAY_NAMES = {
    "humaneval": "code",
    "gsm8k": "math",
    "finqa": "finance"
}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_memory_heatmaps(hits_csv, out_file, num_slots=4096, slot_bins=256):
    df = pd.read_csv(hits_csv)
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(4 * len(TASK_ORDER), 4), sharey=True)
    if len(TASK_ORDER) == 1:
        axes = [axes]

    for ax, task in zip(axes, TASK_ORDER):
        sub = df[df["task"] == task].copy()
        if sub.empty:
            display_name = TASK_DISPLAY_NAMES.get(task, task.capitalize())
            ax.imshow(np.zeros((1, slot_bins)), aspect="auto", origin="lower", cmap='viridis')
            ax.set_title(f"{display_name} (no data)", fontsize=18, fontweight="bold")
            ax.set_xlabel("slot index (binned)", fontsize=15, fontweight="bold")
            if ax is axes[0]:
                ax.set_ylabel("batch id", fontsize=15, fontweight="bold")
            continue

        sub["slot_bin"] = pd.cut(sub["slot_id"], bins=slot_bins, labels=False, right=False)
        sub.loc[sub["slot_id"] == num_slots, "slot_bin"] = slot_bins - 1 # right edge
        
        mat = sub.groupby(["batch_id", "slot_bin"])["weight"].sum().unstack(fill_value=0)
        mat = mat.reindex(columns=range(slot_bins), fill_value=0)
        
        mm = mat.values.astype(float)
        if mm.max() > 0:
            mm = mm / mm.max()
        ax.imshow(mm, aspect="auto", origin="lower", cmap='viridis')
        display_name = TASK_DISPLAY_NAMES.get(task, task.capitalize())
        ax.set_title(f"{display_name}", fontsize=18, fontweight="bold")
        ax.set_xlabel("slot index (binned)", fontsize=15, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("batch id", fontsize=15, fontweight="bold")

    fig.suptitle("Memory Access Heatmaps per Task", fontsize=22, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_file)
    plt.close(fig)

def plot_remap_sankey(remap_csv, out_file, top_flows=50):
    """
    简洁实现，聚合后画 top flows
    """
    try:
        df = pd.read_csv(remap_csv)
        if df.empty or 'weight' not in df.columns or df['weight'].sum() == 0:
            raise FileNotFoundError # Treat as no data
    except (FileNotFoundError, pd.errors.EmptyDataError):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No remapping data available.", ha="center", va="center", fontsize=15)
        ax.axis('off')
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        return

    # 筛选出实际发生重映射的流
    df = df[df["slot_old"] != df["slot_new"]].copy()
    if df.empty:
        print("Warning: No actual remapping events found (slot_old != slot_new). Sankey plot will be empty.")
    # 聚合后画 top flows: 根据权重排序，只取最重要的 N 条流
    df = df.sort_values("weight", ascending=False).head(top_flows)

    fig, ax = plt.subplots(figsize=(10, 8))
    tasks = sorted(df["task_from"].unique())
    n_tasks = len(tasks)
    # 使用更柔和的Set2配色方案
    colors = plt.cm.get_cmap("Set2", n_tasks)
    task_colors = {task: colors(i) for i, task in enumerate(tasks)}

    # 准备节点
    # 计算每个任务slot_old和slot_new的最小值，实现相对编号
    slot_old_min = df.groupby('task_from')['slot_old'].transform('min')
    slot_new_min = df.groupby('task_to')['slot_new'].transform('min')
    df = df.copy()
    df['slot_old_rel'] = df['slot_old'] - slot_old_min
    df['slot_new_rel'] = df['slot_new'] - slot_new_min

    left_keys = sorted(df.apply(lambda r: (r['task_from'], r['slot_old_rel']), axis=1).unique())
    right_keys = sorted(df.apply(lambda r: (r['task_to'], r['slot_new_rel']), axis=1).unique())
    n_left, n_right = len(left_keys), len(right_keys)

    # 节点位置
    pos_left = {k: (0, (i + 0.5) / n_left) for i, k in enumerate(left_keys)}
    pos_right = {k: (1, (i + 0.5) / n_right) for i, k in enumerate(right_keys)}

    # --- 绘制连接线 (flows) ---
    # 根据权重计算线宽和透明度
    max_weight = df['weight'].max() if not df.empty else 1.0
    min_lw, max_lw = 1, 10 # 最小和最大线宽

    for _, r in df.iterrows():
        key_left = (r['task_from'], r['slot_old_rel'])
        key_right = (r['task_to'], r['slot_new_rel'])
        x1, y1 = pos_left[key_left]
        x2, y2 = pos_right[key_right]
        
        weight_ratio = r['weight'] / max_weight if max_weight > 0 else 0
        lw = min_lw + (max_lw - min_lw) * weight_ratio
        alpha = 0.5 + 0.4 * weight_ratio # 权重越大，越不透明

        # 使用Bézier曲线绘制平滑的流
        Path = mpath.Path
        path_data = [
            (Path.MOVETO, (x1, y1)),
            (Path.CURVE4, (x1 + 0.3, y1)), # 控制点1
            (Path.CURVE4, (x2 - 0.3, y2)), # 控制点2
            (Path.CURVE4, (x2, y2)),       # 结束点
        ]
        codes, verts = zip(*path_data)
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor=task_colors.get(r['task_from'], 'gray'), lw=lw, alpha=alpha)
        ax.add_patch(patch)

    # 绘制节点标签
    for k, (x, y) in pos_left.items():
        _, slot = k
        ax.text(x - 0.01, y, f"{slot}", ha="right", va="center", fontsize=8)

    for k, (x, y) in pos_right.items():
        _, slot = k
        ax.text(x + 0.01, y, f"{slot}", ha="left", va="center", fontsize=8)

    # 任务区块标注
    if n_left > 0: # Only draw if there are nodes
        task_regions = df.groupby('task_from').apply(lambda g: sorted([pos_left[(r['task_from'], r['slot_old_rel'])][1] for _, r in g.iterrows()]))
        for task, y_vals in task_regions.items():
            if not y_vals: continue
            min_y, max_y = min(y_vals), max(y_vals)
            height = (max_y - min_y) + (1/n_left)
            center_y = min_y + height/2 - (0.5/n_left)
            ax.add_patch(plt.Rectangle((-0.25, min_y - (0.5/n_left)), 0.2, height, facecolor=task_colors.get(task, 'gray'), alpha=0.6))
            ax.text(-0.125, center_y, TASK_DISPLAY_NAMES.get(task, task), ha="center", va="center", color="white", weight="bold", fontsize=12)

    # 增加左右列的标题
    ax.text(0, 1.02, "Source Slots (Task:Slot)", transform=ax.transAxes, ha="center", fontsize=13, weight="bold")
    ax.text(1, 1.02, "Target Slots", transform=ax.transAxes, ha="center", fontsize=13, weight="bold")

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title(f"Top {top_flows} Slot Remapping Flows by Weight", pad=20, fontsize=16, fontweight="bold")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

def plot_retention_scatter(scores_csv, out_file):
    """
    y=x 参考线
    """
    try:
        df = pd.read_csv(scores_csv)
        if df.empty:
            raise FileNotFoundError
    except (FileNotFoundError, pd.errors.EmptyDataError):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No score data available.", ha="center", va="center", fontsize=15)
        ax.axis('off')
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        return

    if "model" not in df.columns:
        df["model"] = "default"
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    models = df["model"].unique()
    markers = ['o', 's', '^', 'D', 'v', '*', 'p']
    all_scores = []

    for i, model_name in enumerate(models):
        model_df = df[df["model"] == model_name]
        x = model_df["score_single"].values
        y = model_df["score_merged"].values
        tasks = model_df["task"].tolist()

        if len(x) > 0:
            ax.scatter(x, y, s=60, marker=markers[i % len(markers)], label=model_name, zorder=10, alpha=0.8)
            all_scores.extend(x)
            all_scores.extend(y)

        for xi, yi, task_name in zip(x, y, tasks):
            lab = TASK_DISPLAY_NAMES.get(task_name, task_name.capitalize())
            ax.text(xi + 0.005, yi, f" {lab}", va="center", fontsize=9)
    
    if not all_scores:
        ax.text(0.5, 0.5, "Scores are all zero.", ha="center", va="center", fontsize=15)
        all_scores = [0, 1] # Default view

    lim_min = min(all_scores)
    lim_max = max(all_scores)
    pad = (lim_max - lim_min) * 0.1 if lim_max > lim_min else 0.1
    plot_min = lim_min - pad
    plot_max = lim_max + pad

    ax.plot([plot_min, plot_max], [plot_min, plot_max], ls="--", color="gray", lw=1.5, label="y=x (ideal retention)", zorder=5)

    ax.set_xlabel("Single-task Score ($S_{single}$)")
    ax.set_ylabel("Merged Score ($S_{merged}$)")
    ax.set_title("Model Retention vs. Forgetting")
    
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.legend(title="Model" if len(models) > 1 else "")
    
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits_csv", required=True)
    ap.add_argument("--remap_csv", required=True)
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--num_slots", type=int, default=4096)
    ap.add_argument("--slot_bins", type=int, default=256)
    ap.add_argument("--top_flows", type=int, default=50)
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    plot_memory_heatmaps(
        hits_csv=args.hits_csv,
        out_file=os.path.join(args.outdir, "memory_heatmap.pdf"),
        num_slots=args.num_slots,
        slot_bins=args.slot_bins,
    )
    print("Memory heatmap plot saved.")

    plot_remap_sankey(
        remap_csv=args.remap_csv,
        out_file=os.path.join(args.outdir, "remap_sankey.pdf"),
        top_flows=args.top_flows,
    )
    print("Remap sankey plot saved.")

    plot_retention_scatter(
        scores_csv=args.scores_csv,
        out_file=os.path.join(args.outdir, "retention_scatter.pdf"),
    )
    print("Retention scatter plot saved.")

    print("Done. Saved to:", args.outdir)
