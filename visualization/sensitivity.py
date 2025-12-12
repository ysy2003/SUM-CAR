import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = {
    'Setting': ['K_top=8', 'K_top=32', 'K_top=64'],
    'Code (Pass@1)': [80, 70, 69],
    'Math (Accuracy)': [71, 67, 70],
    'Finance (Accuracy)': [26, 26, 28]
}

df = pd.DataFrame(data)
df.set_index('Setting', inplace=True)
metrics = df.columns 

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))


df.plot(kind='bar', ax=ax, width=0.8, rot=0, edgecolor='white', linewidth=2, alpha=0.6)


num_metrics = len(metrics)  
bar_containers = [container for container in ax.containers
                  if len(container) > 0 and isinstance(container[0], matplotlib.patches.Rectangle)]

metric_colors = [container[0].get_facecolor() for container in bar_containers[:num_metrics]]


for container, color in zip(bar_containers[:num_metrics], metric_colors):
    line_x_points = []
    line_y_points = []

    for bar in container:
        center_x = bar.get_x() + bar.get_width() / 2
        top_y = bar.get_height()
        line_x_points.append(center_x)
        line_y_points.append(top_y)

    ax.plot(line_x_points, line_y_points,
            color=color,             
            marker='o',             
            markersize=8,           
            markerfacecolor='white', 
            linewidth=2,             
            linestyle='-',           
            zorder=10)               


# ax.set_title('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Score (%)', fontsize=20, fontweight='bold')
ax.set_xlabel('')
ax.set_ylim(0, 105) 
ax.tick_params(axis='both', labelsize=20)
for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
    tick_label.set_fontweight('bold')

for container in bar_containers[:num_metrics]:
    ax.bar_label(container, fmt='%.1f', padding=3, fontsize=20,
                 color='black', fontweight='bold')


legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   borderaxespad=0., fontsize=20, title_fontsize=16,
                   prop={'weight': 'bold', 'size': 20}, ncol=3)
legend.get_title().set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.savefig('sensitivity_analysis.png', dpi=300)
plt.show()
