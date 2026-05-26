import matplotlib.pyplot as plt
import numpy as np

# 1. Data from the ablation study
tau = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# throughput = np.array([49.6, 67.3, 79.3, 85.2, 86.3, 85.8, 84.7])
# acc_length = np.array([5.06, 4.83, 4.53, 4.20, 3.87, 3.54, 3.07])
# acc_rate = np.array([28.81, 46.16, 61.57, 74.29, 83.88, 89.20, 93.74])
throughput = np.array([63.9, 78.5, 87.8, 91.3, 91.8, 90.5, 88.6])
acc_length = np.array([5.69, 5.35, 4.92, 4.42, 3.97, 3.52, 3.02])
acc_rate = np.array([36.81, 53.24, 67.43, 80.26, 88.57, 93.44, 96.41])

# 2. General Figure Styling (Matches NeurIPS / standard ML paper formats)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

# Define colors (matching the Dual Decoding orange/red and a contrasting blue/teal)
color_tput = '#E26D5C'   # Coral / Orange-Red for Throughput
color_rate = '#3E7CB1'   # Steel Blue for Acceptance Rate
color_len = '#4A8F79'    # Muted Teal for Acceptance Length

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 4))

# =================================================================
# Subplot 1: Confidence Threshold vs. Throughput & Acceptance Rate
# =================================================================
ax1.set_xlabel(r'Confidence Threshold ($\tau$)')
ax1.set_ylabel('Throughput (tokens/s)', color=color_tput, fontweight='bold')
line1 = ax1.plot(tau, throughput, marker='o', color=color_tput, label='Throughput', linewidth=2, markersize=6)
ax1.tick_params(axis='y', labelcolor=color_tput)
ax1.grid(True, linestyle='--', alpha=0.5)

# Create a twin y-axis for Acceptance Rate
ax2 = ax1.twinx()
ax2.set_ylabel('Acceptance Rate (%)', color=color_rate, fontweight='bold')
line2 = ax2.plot(tau, acc_rate, marker='s', color=color_rate, label='Acceptance Rate', linewidth=2, markersize=6)
ax2.tick_params(axis='y', labelcolor=color_rate)

# Combine legends from both axes
lines_1 = line1 + line2
labels_1 = [l.get_label() for l in lines_1]
ax1.legend(lines_1, labels_1, loc='center right')
ax1.set_title('Throughput vs. Acceptance Rate')

# =================================================================
# Subplot 2: Confidence Threshold vs. Throughput & Acceptance Length
# =================================================================
ax3.set_xlabel(r'Confidence Threshold ($\tau$)')
ax3.set_ylabel('Throughput (tokens/s)', color=color_tput, fontweight='bold')
line3 = ax3.plot(tau, throughput, marker='o', color=color_tput, label='Throughput', linewidth=2, markersize=6)
ax3.tick_params(axis='y', labelcolor=color_tput)
ax3.grid(True, linestyle='--', alpha=0.5)

# Create a twin y-axis for Acceptance Length
ax4 = ax3.twinx()
ax4.set_ylabel('Acceptance Length', color=color_len, fontweight='bold')
line4 = ax4.plot(tau, acc_length, marker='^', color=color_len, label='Acceptance Length', linewidth=2, markersize=6)
ax4.tick_params(axis='y', labelcolor=color_len)

# Combine legends from both axes
lines_2 = line3 + line4
labels_2 = [l.get_label() for l in lines_2]
ax3.legend(lines_2, labels_2, loc='center right')
ax3.set_title('Throughput vs. Acceptance Length')

# =================================================================
# Finalizing and Saving
# =================================================================
fig.tight_layout()

# Save the figure as a high-resolution PDF/PNG ready for your LaTeX document
# plt.savefig('ablation_tau.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/hankun/tmp.pdf', dpi=300, bbox_inches='tight')

plt.show()