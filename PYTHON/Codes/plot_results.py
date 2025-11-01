# In file: UAV-RIS_EnergyHarvesting/plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration for Plots ---
plt.style.use('seaborn-v0_8-whitegrid')
font_config = {'family': 'serif', 'size': 12, 'weight': 'normal'}
plt.rc('font', **font_config)
plt.rc('axes', titlesize=14, titleweight='bold', labelsize=12, labelweight='normal')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)

results_dir = "results"
W_S_VALUES = [0.0, 0.2, 0.5, 0.8, 1.0] 

if not os.path.exists(results_dir):
    print(f"Error: Results directory '{results_dir}' not found. Please run the experiment scripts first.")
    exit()

# --- 1. PLOT FIGURE 2: UAV Trajectories (IMPROVED) ---
def plot_figure2():
    print("Generating Figure 2: UAV Trajectories...")
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(W_S_VALUES)))

    # Plot static locations
    ax.scatter(230, 250, c='red', marker='s', s=100, label='CU 1', zorder=5)
    ax.scatter(250, 230, c='red', marker='s', s=100, label='CU 2', zorder=5)
    ax.scatter(60, 235, c='blue', marker='o', s=100, label='Sensing Target', zorder=5)
    ax.scatter(200, 100, c='black', marker='X', s=120, label='Eavesdropper', zorder=5)
    ax.scatter(150, 90, c='green', marker='P', s=150, label='RIS', zorder=5)

    for i, w_S in enumerate(W_S_VALUES):
        filepath = os.path.join(results_dir, f'trajectory_wS_{w_S}.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Plot ISAC UAV trajectory
            ax.plot(df['isac_x'], df['isac_y'], color=colors[i], linestyle='-', label=f'ISAC UAV (w_S={w_S})', zorder=3)
            # Mark start and end points
            ax.scatter(df['isac_x'].iloc[0], df['isac_y'].iloc[0], color=colors[i], marker='v', s=80, zorder=4, edgecolors='black')
            ax.scatter(df['isac_x'].iloc[-1], df['isac_y'].iloc[-1], color=colors[i], marker='^', s=80, zorder=4, edgecolors='black')
    
    ax.set_xlabel('X-coordinate (m)')
    ax.set_ylabel('Y-coordinate (m)')
    ax.set_title('ISAC UAV Trajectory vs. Sensing Weight')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout(rect=[0, 0, 0.75, 1]) # Adjust layout to make space for legend
    fig.savefig(os.path.join(results_dir, 'figure2_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# --- 2. PLOT FIGURE 3: Rate vs. SNR Trade-off ---
def plot_figure3():
    print("Generating Figure 3: Rate-SNR Trade-off...")
    fig, ax = plt.subplots(figsize=(8, 6))
    tradeoff_filepath = os.path.join(results_dir, 'tradeoff_summary.csv')
    if os.path.exists(tradeoff_filepath):
        df = pd.read_csv(tradeoff_filepath).sort_values(by='avg_sensing_snr') # Sort values for a clean plot
        ax.plot(df['avg_sensing_snr'], df['avg_secrecy_rate'], marker='o', markersize=8, linestyle='--', color='dodgerblue')
        for i, row in df.iterrows():
            ax.text(row['avg_sensing_snr'], row['avg_secrecy_rate'] + 0.01, f'w_S={row["w_S"]}', fontsize=9, ha='center')
        ax.set_xlabel('Average Sensing SNR (dB)')
        ax.set_ylabel('Average Secrecy Rate (bits/s/Hz)')
        ax.set_title('Secrecy vs. Sensing Performance Trade-off')
        ax.grid(True, which='both', linestyle='--')
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, 'figure3_rate_snr_tradeoff.png'), dpi=300)
    else:
        print("Could not find tradeoff_summary.csv. Run DDPG.py first.")
    plt.close(fig)

# --- 3. PLOT FIGURE 4: Performance vs. M ---
def plot_figure4():
    print("Generating Figure 4: Performance vs. M...")
    fig, ax_rate = plt.subplots(figsize=(8, 6))
    summary_filepath = os.path.join(results_dir, 'summary_vs_M.csv')
    if os.path.exists(summary_filepath):
        df = pd.read_csv(summary_filepath)
        ax_snr = ax_rate.twinx()
        p1, = ax_rate.plot(df['M'], df['avg_secrecy_rate'], marker='o', color='crimson', label='Avg. Secrecy Rate')
        ax_rate.set_xlabel('Number of RIS Elements (M)')
        ax_rate.set_ylabel('Average Secrecy Rate (bits/s/Hz)', color='crimson')
        ax_rate.tick_params(axis='y', labelcolor='crimson')
        p2, = ax_snr.plot(df['M'], df['avg_sensing_snr'], marker='s', linestyle='--', color='darkviolet', label='Avg. Sensing SNR')
        ax_snr.set_ylabel('Average Sensing SNR (dB)', color='darkviolet')
        ax_snr.tick_params(axis='y', labelcolor='darkviolet')
        ax_rate.set_title('System Performance vs. Number of RIS Elements')
        ax_rate.legend(handles=[p1, p2], loc='best')
        ax_rate.grid(True, which='both', linestyle='--')
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, 'figure4_vs_M.png'), dpi=300)
    else:
        print("Could not find summary_vs_M.csv. Run experiment_vs_M.py first.")
    plt.close(fig)

# --- 4. PLOT NEW FIGURE: Secrecy Rate vs. Power ---
def plot_figure_power():
    print("Generating New Figure: Secrecy Rate vs. Power...")
    fig, ax = plt.subplots(figsize=(8, 6))
    summary_filepath = os.path.join(results_dir, 'summary_secrecy_vs_power.csv')
    if os.path.exists(summary_filepath):
        df = pd.read_csv(summary_filepath)
        ax.plot(df['power_dbm'], df['avg_secrecy_rate'], marker='^', markersize=8, linestyle='-', color='forestgreen')
        ax.set_xlabel('ISAC UAV Max Transmit Power (dBm)')
        ax.set_ylabel('Average Secrecy Rate (bits/s/Hz)')
        ax.set_title('Secrecy Rate vs. UAV Power Budget')
        ax.grid(True, which='both', linestyle='--')
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, 'figure5_vs_power.png'), dpi=300)
    else:
        print("Could not find summary_secrecy_vs_power.csv. Run experiment_secrecy_vs_power.py first.")
    plt.close(fig)

# --- Main execution block ---
if __name__ == '__main__':
    plot_figure2()
    plot_figure3()
    plot_figure4()
    plot_figure_power()
    print("\nPlotting complete.")