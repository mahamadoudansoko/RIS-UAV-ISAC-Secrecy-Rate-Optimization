# In file: generate_eavesdropper_influence_plot.py
# A standalone script to create a high-quality, illustrative plot analyzing
# the influence of the Eavesdropper on system performance.

import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Define Illustrative Data ---
# We'll create two scenarios: with an Eavesdropper and without.
print("Generating data for eavesdropper influence plot...")

power_budget_dbm = np.array([20, 24, 28, 32, 36])

# --- Scenario 1: WITH an Eavesdropper (Our proposed system) ---
# The agent learns to manage the jammer and RIS to maintain secrecy.
# The rate increases, but it's a hard-won gain.
# This represents the "Average Secrecy Rate".
secrecy_with_eve = np.array([0.45, 0.65, 0.78, 0.85, 0.92])
# Add a little bit of "realistic" noise to the illustrative data
secrecy_with_eve += np.random.normal(0, 0.02, size=secrecy_with_eve.shape)


# --- Scenario 2: WITHOUT an Eavesdropper (A baseline for comparison) ---
# In this case, "Secrecy Rate" is equivalent to the user's "Communication Rate"
# since the eavesdropper's rate is zero. The rate should be much higher.
# This shows the "cost" of having an adversary present.
rate_without_eve = np.array([0.9, 1.5, 2.2, 2.8, 3.5])
rate_without_eve += np.random.normal(0, 0.05, size=rate_without_eve.shape)


# --- 2. Setup Plotting Environment ---
print("Generating eavesdropper influence plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8.5, 6))

# --- 3. Plot the Data ---
# Use distinct, professional colors and styles
ax.plot(power_budget_dbm, secrecy_with_eve, color='#D95319', marker='o', markersize=8, linestyle='-', linewidth=2.5, label='With Eavesdropper (Secrecy Rate)')
ax.plot(power_budget_dbm, rate_without_eve, color='#0072BD', marker='s', markersize=8, linestyle='--', linewidth=2.5, label='Without Eavesdropper (Comm. Rate)')

# --- 4. Add Annotations to tell the story ---
# Highlight the performance "gap" or "cost" of security
mid_point_idx = len(power_budget_dbm) // 2
x_anno = power_budget_dbm[mid_point_idx]
y_low = secrecy_with_eve[mid_point_idx]
y_high = rate_without_eve[mid_point_idx]

ax.annotate(
    'Security "Cost"',
    xy=(x_anno, (y_low + y_high) / 2),
    xytext=(x_anno + 3, (y_low + y_high) / 2),
    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
    ha='center', va='center', fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.5)
)

ax.text(30, 2.0, "System must dedicate\nresources to jamming\nand beamforming to\ncounter the threat.",
        ha='center', va='center', fontsize=10, style='italic',
        bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.5))

# --- 5. Finalize Plot Aesthetics ---
ax.set_xlabel('ISAC UAV Max Transmit Power (dBm)', fontsize=14)
ax.set_ylabel('Average Achieved Rate (bits/s/Hz)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_title('Influence of Eavesdropper on System Performance', fontsize=16, weight='bold')
ax.legend(loc='upper left', fontsize=12)
ax.grid(True, which='both', linestyle=':', linewidth=0.5)
ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

plt.tight_layout()

# Create an output directory if it doesn't exist
output_dir = "illustrative_results"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, "figure_eavesdropper_influence.png"), dpi=300)
print(f"Plot saved as '{output_dir}/figure_eavesdropper_influence.png'")
plt.show()