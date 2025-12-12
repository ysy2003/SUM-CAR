import matplotlib.pyplot as plt
import numpy as np

raw_data = {
    'Llama-3-8B-Instruct': [65.24, 74.30, 23.10],
    'FT-Code':    [61.59, 73.31, 22.42],
    'FT-Math':    [64.02, 74.83, 24.24],
    'FT-FinQA':   [60.37, 73.69, 23.44],
    'Multi-task Joint':    [66.46, 74.53, 22.88],
    'SUM-CAR (Ours)':      [68.90, 75.28, 23.56]
}

model_names = list(raw_data.keys())
data_matrix = np.array(list(raw_data.values()))


min_vals = data_matrix.min(axis=0)
max_vals = data_matrix.max(axis=0)


def normalize(values, mins, maxs):
    return 0.2 + 0.8 * (values - mins) / (maxs - mins)

norm_data = {}
for key, val in raw_data.items():
    norm_data[key] = normalize(np.array(val), min_vals, max_vals)

labels = [
    f'Code', 
    f'Math', 
    f'Finance',
]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] 

styles = {
    'Llama-3-8B-Instruct': {'color': '#7f7f7f', 'ls': '--', 'lw': 2, 'marker': '.'},
    'FT-Code':    {'color': '#a6cee3', 'ls': '-', 'lw': 2, 'marker': 'o'},
    'FT-Math':    {'color': '#1f78b4', 'ls': '-', 'lw': 2, 'marker': 's'},
    'FT-FinQA':   {'color': '#b2df8a', 'ls': '-', 'lw': 2, 'marker': '^'},
    'Multi-task Joint': {'color': '#33a02c', 'ls': '-.', 'lw': 2, 'marker': 'D'},
    'SUM-CAR (Ours)':   {'color': '#e31a1c', 'ls': '-', 'lw': 2.5, 'marker': '*'}
}

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for model_name in model_names:
    values = norm_data[model_name].tolist()
    values += values[:1] 
    
    st = styles.get(model_name, {'color': 'black'})
    
    ax.plot(angles, values, label=model_name, 
            color=st['color'], linewidth=st['lw'], linestyle=st['ls'], 
            marker=st['marker'], markersize=6)
    
    if 'SUM-CAR' in model_name:
        ax.fill(angles, values, color=st['color'], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=16, weight='bold')


ax.set_yticklabels([])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, color='grey', linestyle=':', alpha=0.5)
ax.set_ylim(0, 1.05)

legend = plt.legend(loc='upper center', bbox_to_anchor=(0.92, 0.96), fontsize=12,
                    frameon=True, fancybox=True, framealpha=0.95)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.5)
for text in legend.get_texts():
    if 'SUM-CAR' in text.get_text():
        text.set_fontweight('bold')
    else:
        text.set_fontweight('normal')

plt.tight_layout()
plt.savefig('radar_normalized.png', dpi=300, bbox_inches='tight')
plt.show()