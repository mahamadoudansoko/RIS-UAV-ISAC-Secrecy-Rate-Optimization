# In file: DDPG-MultiUT-Time/plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration for Plots ---
# Use a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
font_config = {'family': 'serif', 'size': 12}
plt.rc('font', **font_config)

# Directory where DDPG.py saved the results
results_dir = "results"
W_S_VALUES = [0.0, 0.2, 0.5, 0.8, 1.0] # Must match the list in DDPG.py

# --- 1. PLOT FIGURE 2: UAV Trajectories ---
print("Generating Figure 2: UAV Trajectories...")
fig2, ax2 = plt.subplots(figsize=(8, 7))

# Define some nice colors
colors = plt.cm.viridis(np.linspace(0, 1, len(W_S_VALUES)))

# Plot static locations
ax2.scatter(230, 250, c='red', marker='s', s=100, label='CU 1')
ax2.scatter(250, 230, c='red', marker='s', s=100, label='CU 2')
ax2.scatter(60, 235, c='blue', marker='o', s=100, label='Sensing Target')
ax2.scatter(200, 100, c='black', marker='x', s=100, label='Eavesdropper')
ax2.scatter(150, 90, c='green', marker='P', s=150, label='RIS')

# Loop through results and plot each trajectory
for i, w_S in enumerate(W_S_VALUES):
    filepath = os.path.join(results_dir, f'trajectory_wS_{w_S}.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        ax2.plot(df['isac_x'], df['isac_y'], marker='.', linestyle='-', color=colors[i], label=f'ISAC UAV (w_S={w_S})')

ax2.set_xlabel('X-coordinate (m)')
ax2.set_ylabel('Y-coordinate (m)')
ax2.set_title('Figure 2: ISAC UAV Trajectory vs. Sensing Weight')
ax2.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
ax2.set_xlim(0, 300)
ax2.set_ylim(0, 300)
ax2.set_aspect('equal', adjustable='box')
fig2.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust layout to make space for legend
fig2.savefig(os.path.join(results_dir, 'figure2_trajectories.png'), dpi=300)
plt.close(fig2)


# --- 2. PLOT FIGURE 3: Secrecy Rate vs. Sensing SNR Trade-off ---
print("Generating Figure 3: Rate-SNR Trade-off...")
fig3, ax3 = plt.subplots(figsize=(8, 6))

tradeoff_filepath = os.path.join(results_dir, 'tradeoff_summary.csv')
if os.path.exists(tradeoff_filepath):
    df_tradeoff = pd.read_csv(tradeoff_filepath)
    
    ax3.plot(df_tradeoff['avg_sensing_snr'], df_tradeoff['avg_secrecy_rate'], marker='o', linestyle='--', color='dodgerblue')
    
    # Annotate each point with its w_S value
    for i, row in df_tradeoff.iterrows():
        ax3.text(row['avg_sensing_snr'] + 0.1, row['avg_secrecy_rate'], f' w_S={row["w_S"]}', fontsize=9)

    ax3.set_xlabel('Average Sensing SNR (dB)')
    ax3.set_ylabel('Average Secrecy Rate (bits/s/Hz)')
    ax3.set_title('Figure 3: Secrecy vs. Sensing Performance Trade-off')
    fig3.tight_layout()
    fig3.savefig(os.path.join(results_dir, 'figure3_rate_snr_tradeoff.png'), dpi=300)
    plt.close(fig3)



    # --- 3. PLOT FIGURE 4: Performance vs. Number of RIS Elements (M) ---
print("Generating Figure 4: Performance vs. M...")
fig4, ax4_rate = plt.subplots(figsize=(8, 6))

summary_vs_m_filepath = os.path.join(results_dir, 'summary_vs_M.csv')
if os.path.exists(summary_vs_m_filepath):
    df_vs_m = pd.read_csv(summary_vs_m_filepath)
    
    # Create a twin axis for the second y-axis (for SNR)
    ax4_snr = ax4_rate.twinx()

    # Plot Secrecy Rate on the left axis
    p1, = ax4_rate.plot(df_vs_m['M'], df_vs_m['avg_secrecy_rate'], marker='o', linestyle='-', color='crimson', label='Avg. Secrecy Rate')
    ax4_rate.set_xlabel('Number of RIS Elements (M)')
    ax4_rate.set_ylabel('Average Secrecy Rate (bits/s/Hz)', color='crimson')
    ax4_rate.tick_params(axis='y', labelcolor='crimson')

    # Plot Sensing SNR on the right axis
    p2, = ax4_snr.plot(df_vs_m['M'], df_vs_m['avg_sensing_snr'], marker='s', linestyle='--', color='darkviolet', label='Avg. Sensing SNR')
    ax4_snr.set_ylabel('Average Sensing SNR (dB)', color='darkviolet')
    ax4_snr.tick_params(axis='y', labelcolor='darkviolet')
    
    ax4_rate.set_title('Figure 4: System Performance vs. Number of RIS Elements')
    # Add a single legend for both lines
    ax4_rate.legend(handles=[p1, p2], loc='best')
    fig4.tight_layout()
    fig4.savefig(os.path.join(results_dir, 'figure4_vs_M.png'), dpi=300)
    plt.close(fig4)

# --- 4. PLOT NEW FIGURE: Secrecy Rate vs. UAV Power Budget ---
print("Generating New Figure: Secrecy Rate vs. Power...")
fig5, ax5 = plt.subplots(figsize=(8, 6))

summary_vs_power_filepath = os.path.join(results_dir, 'summary_vs_power.csv')
if os.path.exists(summary_vs_power_filepath):
    df_vs_power = pd.read_csv(summary_vs_power_filepath)
    
    ax5.plot(df_vs_power['power_dbm'], df_vs_power['avg_secrecy_rate'], marker='^', linestyle='-', color='forestgreen')
    
    ax5.set_xlabel('ISAC UAV Max Transmit Power (dBm)')
    ax5.set_ylabel('Average Secrecy Rate (bits/s/Hz)')
    ax5.set_title('Secrecy Rate vs. UAV Power Budget')
    fig5.tight_layout()
    fig5.savefig(os.path.join(results_dir, 'figure5_vs_power.png'), dpi=300)
    plt.close(fig5)
else:
    print(f"Could not find {summary_vs_power_filepath}. Please run experiment_secrecy_vs_power.py first.")    

print("\nAll plots generated and saved in 'results' folder.")