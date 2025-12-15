
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Data
ps_data = np.array([[5.0, 11.5], [6.0, 11.25], [7.0, 11.15], [8.0, 11.0], [9.5, 10.75],
    [10.25, 10.60], [11.25, 10.5], [11.75, 9.5], [13.0, 8.0], [13.75, 6.5], [14.5, 5.0]])
rps_data = np.array([[4.0, 9.25], [5.25, 9], [6.0, 8.75], [7.75, 8.25], [8.250, 8.0],
    [9.0, 7.5], [9.75, 7.0], [10.5, 6.5], [11.0, 5.25], [11.25, 4.75], [11.35, 4.25]])
nr_data = np.array([[3.75, 8.5], [5.0, 8.25], [5.75, 8.15], [6.75, 8.0], [7.60, 7.5],
    [8.5, 7.25], [9.0, 6.5], [9.5, 5.75], [10.25, 4.70], [10.70, 3.75], [10.80, 3.5]])
sf_data = np.array([[3.0, 7.30], [3.50, 7.25], [4.25, 7.0], [5.150, 6.60], [5.50, 6.5],
    [6.250, 6.25], [6.5, 5.75], [7.5, 4.75], [7.750, 4.0], [8.150, 3.75], [8.250, 3]])

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.grid(True, linestyle='--', alpha=0.3)

colors = {
    "ps": (0.4660, 0.6740, 0.1880),
    "rps": (1.0, 0.6471, 0.0),
    "nr": (0.9290, 0.6940, 0.1250),
    "sf": (0.0, 0.4470, 0.7410)
}

ax.plot(ps_data[:,0], ps_data[:,1], 'd-', color=colors['ps'], label='PS', linewidth=2)
ax.plot(rps_data[:,0], rps_data[:,1], '^-', color=colors['rps'], label='RPS', linewidth=2)
ax.plot(nr_data[:,0], nr_data[:,1], 'p-', color=colors['nr'], label='NR', linewidth=2)
ax.plot(sf_data[:,0], sf_data[:,1], 'o-', color=colors['sf'], label='SF', linewidth=2)

ax.set_xlim([2, 16])
ax.set_ylim([2, 13])
ax.set_xticks(np.arange(2, 17, 2))
ax.set_yticks(np.arange(2, 14, 1))
ax.set_xlabel('Average sensing SNR (dB)')
ax.set_ylabel("CU's average-sum-rate (bits/s/Hz)")
ax.set_title('Fig. 3: Rate-SNR Regions for PS and Baseline Schemes')

# Annotations
ax.text(ps_data[0,0] - 0.1, ps_data[0,1] + 0.45, 'β_C = 1', fontsize=10, ha='right')
ax.text(ps_data[-1,0] + 0.2, ps_data[-1,1] + 0.25, 'β_C = 0', fontsize=10, ha='left')

arc = FancyArrowPatch((12, 10), (9, 11), connectionstyle="arc3,rad=0.3",
                      arrowstyle='->', linewidth=1.2, color='black')
ax.add_patch(arc)
ax.text(10.5, 11.7, 'Increasing β_C', fontsize=10, ha='center')

ax.legend(loc='lower left', fontsize=10)
plt.tight_layout()
plt.show()
