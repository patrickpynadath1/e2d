import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.dpi": 100,
    "savefig.dpi": 300,
})

# Data: Ablation on block size
block_sizes = [2, 4, 6, 8, 10, 12]
throughput = [65.8, 86.3, 87.6, 90.7, 91.8, 91.1]
acc_length = [2.78, 3.87, 4.03, 3.89, 3.97, 3.77]

fig, ax1 = plt.subplots(figsize=(9, 4.5))
fig.subplots_adjust(left=0.12, right=0.88, bottom=0.16, top=0.95)

bar_color = "#9ecae1"
line_color = "#d62728"

# ---------------------------------------------------------
# 1. Left Axis (ax1) - Throughput Line Plot
# ---------------------------------------------------------
ax1.plot(
    block_sizes,
    throughput,
    color=line_color,
    marker="o",
    linewidth=2.5,
    markersize=7,
    label="Throughput (tokens/s)"
)
ax1.set_xlabel("Training Block Size", fontsize=17)
ax1.set_ylabel("Throughput (tokens/s)", fontsize=17, color=line_color)
ax1.tick_params(axis="y", labelcolor=line_color)
ax1.set_xticks(block_sizes)
ax1.set_ylim(60, 95)

# ---------------------------------------------------------
# 2. Right Axis (ax2) - Acceptance Length Bar Plot
# ---------------------------------------------------------
ax2 = ax1.twinx()
ax2.bar(
    block_sizes,
    acc_length,
    width=1.2,
    color=bar_color,
    edgecolor="black",
    alpha=0.8,
    label="Acceptance Length"
)
ax2.set_ylabel("Acceptance Length", fontsize=17, color="black")
ax2.tick_params(axis="y", labelcolor="black")
ax2.set_ylim(2.5, 4.2)

# ---------------------------------------------------------
# 3. Z-Order Fix: Bring ax1 (Line) to the front
# ---------------------------------------------------------
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# ---------------------------------------------------------
# 4. Grid Settings
# ---------------------------------------------------------
ax1.grid(axis="y", alpha=0.6)
ax1.grid(axis="x", visible=False)
ax2.grid(False)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right", frameon=True)

# Annotate throughput points
for x, y in zip(block_sizes, throughput):
    ax1.text(x, y + 0.6, f"{y:.1f}", ha="center", va="bottom", fontsize=13, color=line_color)

plt.tight_layout()
plt.savefig("/home/hankun/tmp.pdf", dpi=300)
plt.show()