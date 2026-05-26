import matplotlib.pyplot as plt
import numpy as np

# =========================
# Data
# =========================
t_steps = np.array([1, 2, 3, 4, 5, 6])

e2d_6 = np.array([84.28, 69.63, 58.54, 50.99, 44.80, 39.64])
e2d_adaptive_tree = np.array([96.2, 92.5, 90.2, 87.3, 85.7, 84.4])

mtp_8 = np.array([79.4, 67.5, 58.6, 50.9, 44.4, 38.8])
mtp_adaptive_tree = np.array([95.0, 91.8, 90.8, 89.9, 89.3, 89.1])

# =========================
# Plot Style
# =========================
plt.rcParams['font.family'] = 'serif'

# Colors
c_e2d_6 = '#DD795E'
c_e2d_adaptive_tree = '#7A7A7A'
c_mtp_8 = "#43A585"
c_mtp_adaptive_tree = '#5B9BD5'

# Figure
fig, ax = plt.subplots(figsize=(8, 5))

# =========================
# Plot Lines
# =========================
ax.plot(
    t_steps,
    e2d_6,
    marker='x',
    color=c_e2d_6,
    linestyle='--',
    linewidth=2,
    markersize=9,
    markeredgewidth=2.5,
    label='SEED (draft_len=6)'
)

ax.plot(
    t_steps,
    mtp_8,
    marker='x',
    color=c_mtp_8,
    linestyle=':',
    linewidth=2,
    markersize=9,
    markeredgewidth=2.5,
    label='Apple MTP (draft_len=8, same as Table 1)'
)

ax.plot(
    t_steps,
    e2d_adaptive_tree,
    marker='x',
    color=c_e2d_adaptive_tree,
    linestyle='-',
    linewidth=2,
    markersize=9,
    markeredgewidth=2.5,
    label='SEED (adaptive drafting + tree attn, same as Table 1)'
)

ax.plot(
    t_steps,
    mtp_adaptive_tree,
    marker='x',
    color=c_mtp_adaptive_tree,
    linestyle='-',
    linewidth=2,
    markersize=9,
    markeredgewidth=2.5,
    label='Apple MTP (adaptive drafting + tree attn)'
)

# =========================
# Axes Formatting
# =========================
ax.set_xlabel('Drafting Step', fontsize=13)
ax.set_ylabel('Draft Token Acceptance Rate (%)', fontsize=13)

ax.set_xticks(t_steps)

ax.tick_params(
    axis='both',
    which='major',
    labelsize=11
)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Thicken remaining spines
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)

# Optional: add subtle grid
ax.grid(
    axis='y',
    linestyle='--',
    alpha=0.25
)

# =========================
# Legend
# =========================
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    fontsize=10,
    frameon=False
)

# =========================
# Layout & Save
# =========================
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig(
    '/home/hankun/tmp.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()