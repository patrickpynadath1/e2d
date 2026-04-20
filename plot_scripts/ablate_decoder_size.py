import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.dpi": 100,
    "savefig.dpi": 300,
})

# Data
decoder_sizes = [1, 2, 4, 6, 8, 10, 12, 14]
accuracy = [56.0, 57.5, 57.9, 59.4, 58.8, 57.4, 57.0, 55.6]
throughput = [75.7, 74.3, 66.1, 60.8, 56.3, 53.3, 48.8, 46.4]
acc_length = [2.60, 2.89, 3.06, 3.24, 3.31, 3.47, 3.56, 3.63]
acc_rate = [65.64, 73.00, 77.26, 81.85, 83.67, 87.80, 89.89, 91.99]

fig, ax1 = plt.subplots(figsize=(9, 4.5))
fig.subplots_adjust(left=0.12, right=0.88, bottom=0.16, top=0.95)

bar_color = "#9ecae1"
line_color = "#d62728"

# ---------------------------------------------------------
# 1. Left Axis (ax1) - Throughput Line Plot
# ---------------------------------------------------------
ax1.plot(decoder_sizes, throughput, color=line_color, marker="o", linewidth=2.5, markersize=7, label="Throughput (tokens/s)")
ax1.set_xlabel("Decoder Size (# Layers)", fontsize=17)
ax1.set_ylabel("Throughput (tokens/s)", fontsize=17, color=line_color)
ax1.tick_params(axis="y", labelcolor=line_color)
ax1.set_xticks(decoder_sizes)
ax1.set_ylim(40, 80)

# ---------------------------------------------------------
# 2. Right Axis (ax2) - Acceptance Rate Bar Plot
# ---------------------------------------------------------
ax2 = ax1.twinx()
ax2.bar(decoder_sizes, acc_rate, width=1.2, color=bar_color, edgecolor="black", alpha=0.8, label="Acceptance Rate (%)")
ax2.set_ylabel("Acceptance Rate (%)", fontsize=17, color="black")
ax2.tick_params(axis="y", labelcolor="black")
ax2.set_ylim(60, 95)

# ---------------------------------------------------------
# 3. Z-Order Fix: Bring ax1 (Line) to the front
# ---------------------------------------------------------
ax1.set_zorder(ax2.get_zorder() + 1) # Put line axis on top of bar axis
ax1.patch.set_visible(False)         # Make top axis transparent so bars underneath are visible

# ---------------------------------------------------------
# 4. Less Grids
# ---------------------------------------------------------
ax1.grid(axis='y', alpha=0.6)      # Keep ONLY horizontal grid lines from the Left axis
ax1.grid(axis='x', visible=False)  # Remove vertical grid lines
ax2.grid(False)                    # Turn off Right axis grids entirely to avoid messy double-lines

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right", frameon=True)

# Annotate throughput points (now mapped back to ax1)
for x, y in zip(decoder_sizes, throughput):
    ax1.text(x, y + 0.8, f"{y:.1f}", ha="center", va="bottom", fontsize=13, color=line_color)

plt.tight_layout()
plt.savefig('/home/hankun/tmp2.png')
plt.show()