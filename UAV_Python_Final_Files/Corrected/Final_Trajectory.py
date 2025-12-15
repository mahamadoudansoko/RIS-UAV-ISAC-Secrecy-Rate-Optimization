# In file: generate_paper_figure.py
# A standalone script to reproduce the trajectory plot from the paper
# using pre-defined, illustrative paths.

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Helper function to create smooth trajectories through waypoints ---
def generate_trajectory(waypoints, points_per_segment):
    """
    Generates a path by connecting waypoints with straight lines.
    'waypoints' is a list of [x, y] coordinates.
    'points_per_segment' is a list of how many points to generate for each line segment.
    """
    path = []
    for i in range(len(waypoints) - 1):
        start_pt = np.array(waypoints[i])
        end_pt = np.array(waypoints[i+1])
        num_points = points_per_segment[i]
        
        # Use np.linspace to generate evenly spaced points
        segment = np.linspace(start_pt, end_pt, num_points)
        
        # Avoid duplicating the connection point
        if i > 0:
            segment = segment[1:]
            
        path.extend(segment)
        
    return np.array(path)

# --- 1. Define Static Point Locations (from MATLAB code) ---
initial_point = [0, 150]
final_point = [300, 150]
target = [60, 240]
ris = [150, 90]
rl1 = [60, 150] # Adjusted slightly to match image better
rl2 = [240, 150]
cu1 = [220, 245]
cu2 = [240, 220]

# --- 2. Define Waypoints for Each Illustrative Trajectory ---

# Trajectory 1: NR (No RIS), beta_c = 0.5 (Yellow Stars)
# Path: Initial -> Target -> near CUs -> Final
waypoints_nr = [initial_point, target, [230, 235], final_point]
points_nr = [8, 8, 8]
traj_nr = generate_trajectory(waypoints_nr, points_nr)

# Trajectory 2: PS (Proposed Scheme), beta_c = 1 (Blue Triangles)
# Path: Initial -> hovers near CUs -> Final
waypoints_ps1 = [initial_point, [150, 210], cu1, [260, 200], final_point]
points_ps1 = [10, 8, 5, 8]
traj_ps1 = generate_trajectory(waypoints_ps1, points_ps1)

# Trajectory 3: PS, beta_c = 0 (Orange/Red Squares)
# Path: Initial -> hovers near Target -> veers toward RIS -> Final
waypoints_ps0 = [initial_point, [65, 225], [100, 190], [180, 170], final_point]
points_ps0 = [6, 6, 8, 10]
traj_ps0 = generate_trajectory(waypoints_ps0, points_ps0)

# Trajectory 4: PS, beta_c = 0.5 (Green Diamonds)
# Path: Initial -> near Target -> dips toward RIS -> near RL2 -> Final
waypoints_ps05 = [initial_point, [50, 220], [120, 130], [200, 140], final_point]
points_ps05 = [8, 8, 6, 10]
traj_ps05 = generate_trajectory(waypoints_ps05, points_ps05)

# --- 3. Plotting ---
print("Generating trajectory plot...")

# Use a professional style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8.5, 7))

# Define colors to match the paper's style
colors = {
    'nr': '#fec62e',      # Gold/Yellow
    'ps1': '#0072BD',     # A strong blue
    'ps0': '#D95319',     # A burnt orange/red
    'ps05': '#77AC30',    # A muted green
    'static': '#00205B'   # A very dark navy blue
}

# --- Plot Trajectories with markers on the line ---
# Note: 'markevery' places markers at specified intervals along the line
ax.plot(traj_nr[:, 0], traj_nr[:, 1], color=colors['nr'], linestyle='-', marker='*', markersize=8, markevery=1, label=r'NR $\beta_c=0.5$')
ax.plot(traj_ps1[:, 0], traj_ps1[:, 1], color=colors['ps1'], linestyle='-', marker='>', markersize=7, markevery=1, label=r'PS $\beta_c=1$')
ax.plot(traj_ps0[:, 0], traj_ps0[:, 1], color=colors['ps0'], linestyle='-', marker='s', markersize=6, markevery=1, label=r'PS $\beta_c=0$')
ax.plot(traj_ps05[:, 0], traj_ps05[:, 1], color=colors['ps05'], linestyle='-', marker='D', markersize=6, markevery=1, label=r'PS $\beta_c=0.5$')

# --- Plot Static Points ---
# Target (Square)
ax.scatter(target[0], target[1], s=200, c=colors['static'], marker='s', zorder=5)
ax.text(target[0], target[1] + 8, 'Target', ha='center', fontsize=12, weight='bold')

# CUs (Circles)
ax.scatter(cu1[0], cu1[1], s=250, c=colors['static'], marker='o', zorder=5)
ax.text(cu1[0], cu1[1] + 8, 'CU1', ha='center', fontsize=12, weight='bold')
ax.scatter(cu2[0], cu2[1], s=250, c=colors['static'], marker='o', zorder=5)
ax.text(cu2[0] + 5, cu2[1] + 5, 'CU2', ha='left', fontsize=12, weight='bold')

# RIS, RL1, RL2 (Stars)
ax.scatter(ris[0], ris[1], s=300, c=colors['static'], marker='*', zorder=5)
ax.text(ris[0], ris[1] - 12, 'RIS', ha='center', fontsize=12, weight='bold')
ax.scatter(rl1[0], rl1[1], s=300, c=colors['static'], marker='*', zorder=5)
ax.text(rl1[0] + 8, rl1[1], 'RL1', va='center', fontsize=12, weight='bold')
ax.scatter(rl2[0], rl2[1], s=300, c=colors['static'], marker='*', zorder=5)
ax.text(rl2[0] + 8, rl2[1], 'RL2', va='center', fontsize=12, weight='bold')

# Start and End Points with arrows
ax.annotate('Initial Point', xy=initial_point, xytext=(-20, 135),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=12, ha='center')
ax.annotate('Final Point', xy=final_point, xytext=(280, 135),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=12, ha='center')

# --- Axis and Legend Configuration ---
ax.set_xlabel('x(n) (m)', fontsize=14)
ax.set_ylabel('y(n) (m)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlim(0, 300)
ax.set_ylim(80, 300)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', framealpha=0.8)

plt.tight_layout()
plt.savefig("illustrative_trajectory_plot.png", dpi=300)
print("Plot saved as 'illustrative_trajectory_plot.png'")
plt.show()