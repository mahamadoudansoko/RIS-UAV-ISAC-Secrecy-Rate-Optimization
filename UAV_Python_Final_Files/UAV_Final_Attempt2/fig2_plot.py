import numpy as np
import matplotlib.pyplot as plt

# --- Helper function to generate trajectory points ---
def generate_trajectory_points(waypoints, N_points_per_segment_vec):
    q_traj = []
    num_segments = waypoints.shape[1] - 1
    if num_segments < 1:
        return waypoints
    if len(N_points_per_segment_vec) != num_segments:
        raise ValueError('N_points_per_segment_vec length mismatch')
    for i in range(num_segments):
        startPt = waypoints[:, i]
        endPt = waypoints[:, i + 1]
        num_pts_seg = N_points_per_segment_vec[i]
        if num_pts_seg < 2:
            num_pts_seg = 2
        x_seg = np.linspace(startPt[0], endPt[0], num_pts_seg)
        y_seg = np.linspace(startPt[1], endPt[1], num_pts_seg)
        if i == 0:
            q_traj = np.vstack([x_seg, y_seg])
        else:
            q_traj = np.hstack([q_traj, np.vstack([x_seg[1:], y_seg[1:]])])
    return q_traj

# --- Define Static Point Locations ---
q0 = np.array([0, 150])      # Initial Point
qF = np.array([300, 150])    # Final Point
us = np.array([60, 240])     # Target
uR = np.array([150, 90])     # RIS
uR_RL1 = np.array([45, 150]) # RL1
uR_RL2 = np.array([240, 150])# RL2
CU1 = np.array([220, 245])   # CU1
CU2 = np.array([240, 220])   # CU2

# --- Define Waypoints and Marker Counts for Trajectories ---
wp_NR = np.array([q0, us, [150, 240], [220,230], qF]).T
N_NR  = [7, 6, 5, 8]
q_NR = generate_trajectory_points(wp_NR, N_NR)

wp_PS1 = np.array([q0, [50,170],[90,190], [150,210], [190,220], CU1, [250, 180], qF]).T
N_PS1  = [3, 3, 3, 3, 3, 4, 5]
q_PS1 = generate_trajectory_points(wp_PS1, N_PS1)

wp_PS0 = np.array([q0, [60,230], [90,200], [150,180], [210,160],[250,155], qF]).T
N_PS0  = [4, 4, 4, 4, 3, 4]
q_PS0 = generate_trajectory_points(wp_PS0, N_PS0)

wp_PS05 = np.array([q0, [60,220], [80,170], [120,130], [160,150], [200,180], [240,160], qF]).T
N_PS05  = [5, 4, 4, 3, 3, 4, 5]
q_PS05 = generate_trajectory_points(wp_PS05, N_PS05)

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6.5))

# Colors and styles
color_NR = '#D95319' # Orange
color_PS1 = '#0072BD' # Blue
color_PS0 = '#A2142F' # Red
color_PS05 = '#77AC30' # Green
line_width_traj = 2.0
marker_size_traj = 7
marker_edge_color_traj = 'k'

# Plot Trajectories
ax.plot(q_PS05[0,:], q_PS05[1,:], '-d', color=color_PS05, markerfacecolor=color_PS05, markeredgecolor=marker_edge_color_traj, linewidth=line_width_traj, markersize=marker_size_traj, label=r'PS $\beta_C = 0.5$')
ax.plot(q_PS0[0,:], q_PS0[1,:], '-s', color=color_PS0, markerfacecolor=color_PS0, markeredgecolor=marker_edge_color_traj, linewidth=line_width_traj, markersize=marker_size_traj, label=r'PS $\beta_C = 0$')
ax.plot(q_PS1[0,:], q_PS1[1,:], '-^', color=color_PS1, markerfacecolor=color_PS1, markeredgecolor=marker_edge_color_traj, linewidth=line_width_traj, markersize=marker_size_traj, label=r'PS $\beta_C = 1$')
ax.plot(q_NR[0,:], q_NR[1,:], '-*', color=color_NR, markerfacecolor=color_NR, markeredgecolor=color_NR, linewidth=line_width_traj, markersize=marker_size_traj+2, label=r'NR $\beta_C = 0.5$')

# Plot Static Points
static_marker_color = '#000040'
text_offset_y, text_offset_x = 10, 8
ax.scatter(us[0], us[1], s=180, marker='s', c=static_marker_color, edgecolors='k', zorder=5)
ax.text(us[0], us[1] + text_offset_y, 'Target', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.scatter(uR[0], uR[1], s=220, marker='p', c=static_marker_color, edgecolors='k', zorder=5)
ax.text(uR[0], uR[1] - text_offset_y, 'RIS', ha='center', va='top', fontsize=10, fontweight='bold')
ax.scatter(CU1[0], CU1[1], s=180, marker='o', c=static_marker_color, edgecolors='k', zorder=5)
ax.text(CU1[0], CU1[1] + text_offset_y, 'CU1', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.scatter(CU2[0], CU2[1], s=180, marker='o', c=static_marker_color, edgecolors='k', zorder=5)
ax.text(CU2[0] + text_offset_x, CU2[1] - text_offset_y, 'CU2', ha='left', va='center', fontsize=10, fontweight='bold')
ax.plot(q0[0], q0[1], 'o', markersize=9, markerfacecolor='k', markeredgecolor='k', zorder=5)
ax.text(q0[0] - text_offset_x, q0[1], 'Initial Point', ha='right', va='center', fontsize=10)
ax.plot(qF[0], qF[1], 'o', markersize=9, markerfacecolor='k', markeredgecolor='k', zorder=5)
ax.text(qF[0] + text_offset_x, qF[1], 'Final Point', ha='left', va='center', fontsize=10)

# Axis and Legend
ax.set_xlabel('x(n) (m)', fontsize=12)
ax.set_ylabel('y(n) (m)', fontsize=12)
ax.set_xlim(-20, 320)
ax.set_ylim(60, 310)
ax.set_xticks(np.arange(0, 301, 30))
ax.set_yticks(np.arange(90, 301, 30))
ax.legend(loc='lower right', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()