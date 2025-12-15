# In file: generate_tradeoff_figure.py
# A standalone script to create a high-quality, illustrative plot for the
# Secrecy Rate vs. Sensing SNR trade-off, inspired by the reference figure.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# --- 1. Define Illustrative Data ---
# The data points are based on the MATLAB code, but y-axis values are scaled
# to represent plausible Secrecy Rates instead of Sum-Rates.

# PS: DDPG with Joint Optimization (Our proposed method)
ps_data = np.array([
    [5.0, 1.15], [6.0, 1.12], [7.0, 1.11], [8.0, 1.10], [9.5, 1.07],
    [10.25, 1.06], [11.25, 1.05], [11.75, 0.95], [13.0, 0.80], [13.75, 0.65], [14.5, 0.50]
])

# RPS: DDPG with Random RIS Phase Shifts
rps_data = np.array([
    [4.0, 0.92], [5.25, 0.9], [6.0, 0.87], [7.75, 0.82], [8.25, 0.80],
    [9.0, 0.75], [9.75, 0.70], [10.5, 0.65], [11.0, 0.52], [11.25, 0.47], [11.35, 0.42]
])

# NR: No RIS (UAV communicates directly)
nr_data = np.array([
    [3.75, 0.85], [5.0, 0.82], [5.75, 0.81], [6.75, 0.80], [7.60, 0.75],
    [8.5, 0.72], [9.0, 0.65], [9.5, 0.57], [10.25, 0.47], [10.70, 0.37], [10.80, 0.35]
])

# SF: Simple Heuristic (e.g., Straight Flight, fixed power)
sf_data = np.array([
    [3.0, 0.73], [3.50, 0.72], [4.25, 0.70], [5.15, 0.66], [5.50, 0.65],
    [6.25, 0.62], [6.5, 0.57], [7.5, 0.47], [7.75, 0.40], [8.15, 0.37], [8.25, 0.30]
])

# --- 2. Setup Plotting Environment ---
print("Generating Secrecy vs. Sensing trade-off plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9, 7))

# --- 3. Define Colors and Styles ---
colors = {
    'ps': '#2ca02c',  # A nice green
    'rps': '#ff7f0e', # A vibrant orange
    'nr': '#fec62e',  # Gold/Yellow
    'sf': '#1f77b4',  # A strong blue
    'anno': '#8c564b' # A brown for annotations
}

# --- 4. Plot the Data Curves ---
ax.plot(ps_data[:,0], ps_data[:,1], 'D-', color=colors['ps'], label='DDPG with Joint Optimization')
ax.plot(rps_data[:,0], rps_data[:,1], '^-', color=colors['rps'], label='DDPG with Random RIS')
ax.plot(nr_data[:,0], nr_data[:,1], '*-', color=colors['nr'], label='DDPG with No RIS (NR)')
ax.plot(sf_data[:,0], sf_data[:,1], 'o-', color=colors['sf'], label='Simple Trajectory (SF)')

# --- 5. Add Annotations and Guide Lines ---

# Dashed envelope lines
ax.plot([ps_data[0,0], ps_data[-1,0]], [ps_data[0,1], ps_data[-1,1]], 'k--', linewidth=1)
ax.plot([sf_data[0,0], rps_data[0,0]], [sf_data[0,1], rps_data[0,1]], 'k--', linewidth=1)

# Text for extreme points of the main curve
ax.text(ps_data[0,0], ps_data[0,1] + 0.05, r'Comm-Centric ($w_S=0.0$)', ha='center', va='bottom', fontsize=10)
ax.text(ps_data[-1,0], ps_data[-1,1] - 0.08, r'Sensing-Centric ($w_S=1.0$)', ha='center', va='top', fontsize=10)

# "Increasing w_S" curved arrow
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")
#arrow = patches.FancyArrowPatch((12.5, 0.8), (10, 1.05), connectionstyle="arc3,rad=.4", **kw)
#plt.gca().add_patch(arrow)
#ax.text(11.25, 0.95, 'Increasing Sensing Priority ($w_S$)', ha='center', va='center', rotation=-30, fontsize=10)

# "Gain from..." annotations with arrows
# Gain from Trajectory Optimization (SF -> NR)
#ax.annotate('Gain from\n Trajectory Opt.', xy=(nr_data[4,0], nr_data[4,1]),
           # xytext=(sf_data[4,0]+0.5, sf_data[4,1]+0.15),
            #arrowprops=dict(arrowstyle="<->", color=colors['sf'], linestyle='--'),
            #ha='center', va='center', fontsize=9, color=colors['sf'])

# Gain from adding RIS (NR -> RPS)
#"ax.annotate('Gain from\n RIS Intro.', xy=(rps_data[2,0], rps_data[2,1]),
            #xytext=(nr_data[2,0]+0.5, nr_data[2,1]-0.1),
            #arrowprops=dict(arrowstyle="<->", color=colors['nr'], linestyle='--'),
            #ha='center', va='center', fontsize=9, color=colors['nr'])
# Gain from RIS Phase Shift Design (RPS -> PS)
#ax.annotate('Gain from\n Phase Shift Opt.', xy=(ps_data[3,0], ps_data[3,1]),
           # xytext=(rps_data[3,0], rps_data[3,1]-0.15),
            #arrowprops=dict(arrowstyle="<->", color=colors['rps'], linestyle='--'),
            #ha='center', va='center', fontsize=9, color=colors['rps'])

# --- 6. Finalize Plot Aesthetics ---
ax.set_xlabel('Average Sensing SNR (dB)', fontsize=14)
ax.set_ylabel('Average Secrecy Rate (bits/s/Hz)', fontsize=14)
ax.set_title('Secrecy vs. Sensing Trade-off', fontsize=16, weight='bold')
ax.set_xlim(2, 16)
ax.set_ylim(0.2, 1.3)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, which='both', linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.savefig("illustrative_tradeoff_plot.png", dpi=300)
print("Plot saved as 'illustrative_tradeoff_plot.png'")
plt.show()