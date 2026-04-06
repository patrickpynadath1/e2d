import matplotlib.pyplot as plt
import numpy as np

# 1. Parse and prepare the single set of data
t_steps = np.array([1, 2, 3, 4, 5, 6])

layerskip = np.array([59.06, 42.75, 32.84, 26.45, 21.29, 18.04])
e2d = np.array([85.07, 73.87, 63.47, 54.65, 46.05, 38.31])
eagle3 = np.array([86.42, 64.30, 43.06, 28.37, 19.77, 14.43])

# 2. Set up the plot style
plt.rcParams['font.family'] = 'serif'

# Colors
c_ours = '#DD795E'      # Salmon/Orange
c_layerskip = '#7A7A7A'  # Neutral Medium-Dark Gray to complement the orange
c_eagle3 = "#43A585"

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# 3 & 4. Plot data points and connecting lines
# Using ax.plot with markers automatically connects the crosses and creates a perfect legend handle.
# markersize=9 and markeredgewidth=2.5 visually match your previous scatter(s=80, linewidths=2.5)

# Ours: Dashed line with crosses
ax.plot(t_steps, e2d, marker='x', color=c_ours, linestyle='--', linewidth=2, 
        markersize=9, markeredgewidth=2.5, label='Dual Decoding')

# Eagle3: Dotted line with crosses
ax.plot(t_steps, eagle3, marker='x', color=c_eagle3, linestyle='-', linewidth=2, 
        markersize=9, markeredgewidth=2.5, label='EAGLE-3')

# LayerSkip: Solid line with crosses
ax.plot(t_steps, layerskip, marker='x', color=c_layerskip, linestyle='-', linewidth=2, 
        markersize=9, markeredgewidth=2.5, label='LayerSkip')

# 5. Customize axes and remove unnecessary spines (box lines)
ax.set_xlabel('Drafting Step', fontsize=12)
ax.set_ylabel('Draft Token Acceptace Rate (%)', fontsize=12)

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
ax.legend(loc='lower left', fontsize=11, framealpha=1.0, edgecolor='#D3D3D3', borderpad=0.6)

plt.tight_layout()
plt.savefig('/home/hankun/tmp.png') #, dpi=300, bbox_inches='tight')
plt.show()