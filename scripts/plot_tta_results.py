import matplotlib.pyplot as plt
import os

# Create img directory if it doesn't exist
os.makedirs("docs/img", exist_ok=True)

views = [1, 2, 5, 10]
top1 = [71.19, 71.91, 72.70, 73.12]
top5 = [89.79, 90.25, 90.93, 91.15]

# Set a clean, modern aesthetic similar to Seaborn
plt.style.use('ggplot')

fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = '#1f77b4' # Plotly Blue
ax1.set_xlabel('Number of TTA Views', fontsize=12, fontweight='bold')
ax1.set_ylabel('Top-1 Accuracy (%)', color=color1, fontsize=12, fontweight='bold')
line1 = ax1.plot(views, top1, marker='o', markersize=8, color=color1, linewidth=2.5, label='Top-1 Accuracy')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax1.set_xticks(views)

# Add grid lines for the primary axis
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()
color2 = '#ff7f0e' # Plotly Orange
ax2.set_ylabel('Top-5 Accuracy (%)', color=color2, fontsize=12, fontweight='bold')
line2 = ax2.plot(views, top5, marker='s', markersize=8, color=color2, linewidth=2.5, label='Top-5 Accuracy')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=10)

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=11, framealpha=0.9)

plt.title('Effect of Test-Time Augmentation on ResNet18', fontsize=14, fontweight='bold', pad=15)

fig.tight_layout()

output_path = 'docs/img/tta_accuracy_plot.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved plot to {output_path}")
