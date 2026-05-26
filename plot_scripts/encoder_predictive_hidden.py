import matplotlib.pyplot as plt
import numpy as np

# 1. Parse and prepare the data
t_steps = np.array([1, 2, 3, 4, 5])

# Seed 1 Data
ar_s1 = np.array([87.40, 70.05, 55.93, 46.08, 38.66])
e2d_s1 = np.array([85.95, 73.09, 62.52, 54.08, 46.27])
# mtp_s1 = np.array([83.88, 73.61, 64.89, 57.62, 50.11])

# Seed 42 Data
ar_s42 = np.array([87.28, 68.30, 55.41, 44.19, 37.38])
e2d_s42 = np.array([86.04, 71.79, 62.16, 53.42, 44.90])
# mtp_s42 = np.array([83.93, 73.58, 65.75, 57.77, 50.32])

# Calculate means for the lines
ar_mean = (ar_s1 + ar_s42) / 2
e2d_mean = (e2d_s1 + e2d_s42) / 2
# mtp_mean = (mtp_s1 + mtp_s42) / 2

# 2. Set up the plot style (matching the image)
plt.rcParams['font.family'] = 'serif'  # Serif font used in your image

# Hex colors extracted to match your reference image
c_ours = '#DD795E'  # Salmon/Orange
c_baseline = '#3D3E53'      # Dark Slate Blue
# c_mtp = '#5B9BD5'           # Medium Blue (for MTP, if we had data)

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# 3. Plot individual data points as thick crosses
# We only add the 'label' argument to the first scatter call so it appears correctly in the legend
# ax.scatter(t_steps, mtp_s1, marker='x', color=c_mtp, s=80, linewidths=2.5, label='MTP')
# ax.scatter(t_steps, mtp_s42, marker='x', color=c_mtp, s=80, linewidths=2.5)

ax.scatter(t_steps, e2d_s1, marker='x', color=c_ours, s=80, linewidths=2.5, label='SEED')
ax.scatter(t_steps, e2d_s42, marker='x', color=c_ours, s=80, linewidths=2.5)

ax.scatter(t_steps, ar_s1, marker='x', color=c_baseline, s=80, linewidths=2.5, label='AR')
ax.scatter(t_steps, ar_s42, marker='x', color=c_baseline, s=80, linewidths=2.5)

# 4. Plot lines connecting the means
# Baseline: Solid line
ax.plot(t_steps, ar_mean, color=c_baseline, linestyle='-', linewidth=2)
# Ours: Dashed line
ax.plot(t_steps, e2d_mean, color=c_ours, linestyle='--', linewidth=2)
# MTP: Dotted line
# ax.plot(t_steps, mtp_mean, color=c_mtp, linestyle=':', linewidth=2)

# 5. Customize axes and remove unnecessary spines (box lines)
ax.set_xlabel('Lookahead Step', fontsize=12)
ax.set_ylabel('Prediction Accuracy (%)', fontsize=12)

# Force x-axis to only show the integer steps we have data for
ax.set_xticks(t_steps)
ax.tick_params(axis='both', which='major', labelsize=11)

# Despine: Remove top and right borders to match the target style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Make the remaining axes slightly thicker
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)

# 6. Format the legend
# Placed in the lower left as the lines curve downward
ax.legend(loc='lower left', fontsize=11, framealpha=1.0, edgecolor='#D3D3D3', borderpad=0.6)

plt.tight_layout()
plt.savefig('/home/hankun/tmp.pdf') #, dpi=300, bbox_inches='tight')
plt.show()