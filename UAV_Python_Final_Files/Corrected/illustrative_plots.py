# In file: generate_illustrative_plots.py
# A single script to generate all figures with hand-crafted, illustrative data.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# =========================================================================
# 1. SETUP THE SCENE AND STATIC LOCATIONS
# =========================================================================
print("Setting up the simulation scene...")

# Static locations from your environment
loc_cu1 = np.array([230, 250])
loc_cu2 = np.array([250, 230])
loc_target = np.array([60, 235])
loc_eve = np.array([200, 100])
loc_ris = np.array([150, 90])
start_pos = np.array([0, 150])
end_pos = np.array([300, 150]) # A logical end-point for the mission
num_steps = 50 # Number of points in the trajectory

output_dir = "illustrative_results"
os.makedirs(output_dir, exist_ok=True)

# =========================================================================
# 2. GENERATE ILLUSTRATIVE TRAJECTORIES
# We will create logical paths for different agent "priorities".
# =========================================================================

# Helper function to create a smooth trajectory towards a waypoint
def create_trajectory(start, waypoint, end, steps):
    # Create a path from start to waypoint, then from waypoint to end
    path1 = np.linspace(start, waypoint, int(steps * 0.7))
    path2 = np.linspace(waypoint, end, int(steps * 0.3))
    return np.vstack((path1, path2))

# Scenario 1: Communication-focused (w_S = 0.0)
# Goal: Get close to the CUs and the RIS
comm_waypoint = (loc_cu1 + loc_ris) / 2
traj_comm = create_trajectory(start_pos, comm_waypoint, end_pos, num_steps)

# Scenario 2: Sensing-focused (w_S = 1.0)
# Goal: Get close to the Sensing Target
sense_waypoint = loc_target + np.array([20, -20]) # Hover near the target
traj_sense = create_trajectory(start_pos, sense_waypoint, end_pos, num_steps)

# Scenario 3: Balanced (w_S = 0.5)
# Goal: A path that is a compromise between the two
balanced_waypoint = (comm_waypoint + sense_waypoint) / 2
traj_balanced = create_trajectory(start_pos, balanced_waypoint, end_pos, num_steps)

trajectories = {
    0.0: traj_comm,
    0.5: traj_balanced,
    1.0: traj_sense
}

# =========================================================================
# 3. PLOT FIGURE 2: TRAJECTORIES
# =========================================================================
print("Generating Figure 2: Trajectories...")
fig2, ax2 = plt.subplots(figsize=(9, 8))
plt.style.use('seaborn-v0_8-whitegrid')
font_config = {'family': 'serif', 'size': 12}
plt.rc('font', **font_config)

colors = {'comm': 'darkblue', 'balanced': 'darkviolet', 'sense': 'orange'}
labels = {0.0: 'Comm-Focused (w_S=0.0)', 0.5: 'Balanced (w_S=0.5)', 1.0: 'Sensing-Focused (w_S=1.0)'}

# Plot static locations
ax2.scatter(loc_cu1[0], loc_cu1[1], c='red', marker='s', s=120, label='CU 1', zorder=5, edgecolors='black')
ax2.scatter(loc_cu2[0], loc_cu2[1], c='red', marker='s', s=120, label='CU 2', zorder=5, edgecolors='black')
ax2.scatter(loc_target[0], loc_target[1], c='blue', marker='o', s=120, label='Sensing Target', zorder=5, edgecolors='black')
ax2.scatter(loc_eve[0], loc_eve[1], c='black', marker='X', s=150, label='Eavesdropper', zorder=5)
ax2.scatter(loc_ris[0], loc_ris[1], c='green', marker='P', s=200, label='RIS', zorder=5, edgecolors='black')

# Plot trajectories
for w_S, traj in trajectories.items():
    color_key = list(colors.keys())[list(trajectories.keys()).index(w_S)]
    ax2.plot(traj[:, 0], traj[:, 1], color=colors[color_key], linestyle='-', linewidth=2.5, label=labels[w_S], zorder=3)
    ax2.scatter(traj[0, 0], traj[0, 1], color=colors[color_key], marker='v', s=100, zorder=4, edgecolors='black')
    ax2.scatter(traj[-1, 0], traj[-1, 1], color=colors[color_key], marker='^', s=100, zorder=4, edgecolors='black')

ax2.set_xlabel('X-coordinate (m)')
ax2.set_ylabel('Y-coordinate (m)')
ax2.set_title('UAV Trajectories for Different Priorities', fontsize=16, weight='bold')
ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
ax2.set_xlim(0, 300)
ax2.set_ylim(0, 300)
ax2.set_aspect('equal', adjustable='box')
fig2.tight_layout(rect=[0, 0, 0.75, 1])
fig2.savefig(os.path.join(output_dir, 'figure2_illustrative_trajectories.png'), dpi=300)
plt.close(fig2)

# =========================================================================
# 4. GENERATE ILLUSTRATIVE DATA & PLOTS FOR OTHER FIGURES
# =========================================================================

# --- Figure 3: Rate-SNR Trade-off ---
print("Generating Figure 3: Rate-SNR Trade-off...")
# Create plausible data: As SNR increases, secrecy rate must decrease
snr_data = np.array([37, 40, 43])
rate_data = np.array([0.9, 0.55, 0.2])
labels_tradeoff = ['w_S=0.0', 'w_S=0.5', 'w_S=1.0']

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(snr_data, rate_data, marker='o', markersize=10, linestyle='--', color='dodgerblue')
for i, label in enumerate(labels_tradeoff):
    ax3.text(snr_data[i], rate_data[i] + 0.02, label, ha='center', fontsize=10)
ax3.set_xlabel('Average Sensing SNR (dB)')
ax3.set_ylabel('Average Secrecy Rate (bits/s/Hz)')
ax3.set_title('Secrecy vs. Sensing Trade-off', fontsize=16, weight='bold')
ax3.grid(True, which='both', linestyle='--')
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, 'figure3_illustrative_tradeoff.png'), dpi=300)
plt.close(fig3)

# --- Figure 4: Performance vs. M ---
print("Generating Figure 4: Performance vs. M...")
M_values = np.array([25, 50, 75, 100, 125, 150])
# Plausible data: Performance increases with M, but with diminishing returns
base_rate_M = 0.5 + 1.5 * (1 - np.exp(-M_values/50)) + np.random.randn(len(M_values))*0.05
base_snr_M = 37 + 7 * (1 - np.exp(-M_values/60)) + np.random.randn(len(M_values))*0.1

fig4, ax4_rate = plt.subplots(figsize=(8, 6))
ax4_snr = ax4_rate.twinx()
p1, = ax4_rate.plot(M_values, base_rate_M, marker='o', color='crimson', label='Avg. Secrecy Rate')
ax4_rate.set_xlabel('Number of RIS Elements (M)')
ax4_rate.set_ylabel('Average Secrecy Rate (bits/s/Hz)', color='crimson')
ax4_rate.tick_params(axis='y', labelcolor='crimson')
p2, = ax4_snr.plot(M_values, base_snr_M, marker='s', linestyle='--', color='darkviolet', label='Avg. Sensing SNR')
ax4_snr.set_ylabel('Average Sensing SNR (dB)', color='darkviolet')
ax4_snr.tick_params(axis='y', labelcolor='darkviolet')
ax4_rate.set_title('Performance vs. Number of RIS Elements', fontsize=16, weight='bold')
ax4_rate.legend(handles=[p1, p2], loc='best')
fig4.tight_layout()
fig4.savefig(os.path.join(output_dir, 'figure4_illustrative_vs_M.png'), dpi=300)
plt.close(fig4)

# --- New Figure: Secrecy Rate vs. Power ---
print("Generating New Figure: Secrecy Rate vs. Power...")
power_dbm = np.array([20, 24, 28, 32, 36])
# Plausible data: Rate increases with power
rate_vs_power = 0.1 + 0.5 * (power_dbm - 20) / 16 + np.random.randn(len(power_dbm))*0.03

fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.plot(power_dbm, rate_vs_power, marker='^', markersize=8, linestyle='-', color='forestgreen')
ax5.set_xlabel('ISAC UAV Max Transmit Power (dBm)')
ax5.set_ylabel('Average Secrecy Rate (bits/s/Hz)')
ax5.set_title('Secrecy Rate vs. UAV Power Budget', fontsize=16, weight='bold')
ax5.set_ylim(bottom=0)
fig5.tight_layout()
fig5.savefig(os.path.join(output_dir, 'figure5_illustrative_vs_power.png'), dpi=300)
plt.close(fig5)

print(f"\nAll illustrative plots have been generated and saved in the '{output_dir}' folder.")