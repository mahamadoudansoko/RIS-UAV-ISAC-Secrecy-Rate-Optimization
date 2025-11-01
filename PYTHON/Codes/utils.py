# utils.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from config import SYSTEM_PARAMS, PATHS

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state, dtype=torch.float32).numpy()

# --- Plotting Functions ---

def plot_training_rewards(rewards, filename="training_rewards.png"):
    if not os.path.exists(PATHS['PLOT_SAVE_DIR']):
        os.makedirs(PATHS['PLOT_SAVE_DIR'])
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Average Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.savefig(os.path.join(PATHS['PLOT_SAVE_DIR'], filename))
    plt.close()

def plot_multi_trajectory(results, filename="multi_trajectory_tradeoff.png"):
    sp = SYSTEM_PARAMS
    if not os.path.exists(PATHS['PLOT_SAVE_DIR']):
        os.makedirs(PATHS['PLOT_SAVE_DIR'])


    plt.figure(figsize=(12, 9))
    # Use a colormap to automatically assign different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Plot each trajectory with a different color
    for i, (beta_c, data) in enumerate(results.items()):
        trajectory_c = data['trajectory_c']
        plt.plot(trajectory_c[:, 0], trajectory_c[:, 1], color=colors[i], label=f'ISAC UAV ($\\beta_C = {beta_c}$)')

    # Plot static elements
    plt.scatter(sp['UAV_C_START'][0], sp['UAV_C_START'][1], c='black', marker='s', s=150, zorder=5, label='Start Point')
    plt.scatter(sp['UAV_C_END'][0], sp['UAV_C_END'][1], c='black', marker='X', s=150, zorder=5, label='End Point')
    
    plt.scatter(sp['USER_LOCATIONS'][:, 0], sp['USER_LOCATIONS'][:, 1], c='red', marker='o', s=150, zorder=5, label='CUs')
    plt.scatter(sp['TARGET_LOCATION'][0], sp['TARGET_LOCATION'][1], c='purple', marker='^', s=150, zorder=5, label='Target')
    plt.scatter(sp['RIS_LOCATION'][0], sp['RIS_LOCATION'][1], c='orange', marker='p', s=150, zorder=5, label='RIS')
    
    plt.title("UAV Trajectory Trade-off vs. Communication Weighting Factor ($\\beta_C$)")
    plt.xlabel("X-coordinate (m)")
    plt.ylabel("Y-coordinate (m)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(PATHS['PLOT_SAVE_DIR'], filename))
    plt.close()



def plot_rate_snr_tradeoff(results, filename="rate_snr_tradeoff.png"):
    if not os.path.exists(PATHS['PLOT_SAVE_DIR']):
        os.makedirs(PATHS['PLOT_SAVE_DIR'])
        
    betas = sorted(results.keys())
    rates = [results[b]['avg_comm_rate'] for b in betas]
    snrs = [10 * np.log10(results[b]['avg_sensing_snr']) for b in betas] # Convert to dB

    plt.figure(figsize=(10, 6))
    plt.plot(snrs, rates, '-o', label='DDPG Policy Trade-off Curve', markersize=8)
    plt.title("Communication Rate vs. Sensing SNR Trade-off")
    plt.xlabel("Average Sensing SNR (dB)")
    plt.ylabel("Average Sum Communication Rate (bits/s/Hz)")
    
    # Add clear annotations for each point
    for i, beta in enumerate(betas):
        plt.annotate(
            f'$\\beta_C={beta}$', 
            (snrs[i], rates[i]),
            textcoords="offset points",
            xytext=(0,10), # 10 points vertical offset
            ha='center',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
        )

    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PATHS['PLOT_SAVE_DIR'], filename))
    plt.close()


    
def plot_performance_vs_m(results, filename="performance_vs_m.png"):
    if not os.path.exists(PATHS['PLOT_SAVE_DIR']):
        os.makedirs(PATHS['PLOT_SAVE_DIR'])

    m_values = sorted(results.keys())
    rates = [results[m]['avg_comm_rate'] for m in m_values]
    snrs_db = [10 * np.log10(results[m]['avg_sensing_snr']) for m in m_values]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Number of RIS Elements (M)")
    ax1.set_ylabel("Average Sum Rate (bits/s/Hz)", color='tab:blue')
    ax1.plot(m_values, rates, 'o-', color='tab:blue', label='Communication Rate')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Sensing SNR (dB)", color='tab:red')
    ax2.plot(m_values, snrs_db, 'o-', color='tab:red', label='Sensing SNR')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title("C&S Performance vs. Number of RIS Elements ($\\beta_C=0.5$)")
    plt.grid(True)
    plt.savefig(os.path.join(PATHS['PLOT_SAVE_DIR'], filename))
    plt.close()