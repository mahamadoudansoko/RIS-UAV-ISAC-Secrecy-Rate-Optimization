
import matplotlib.pyplot as plt
import numpy as np

def generate_trajectory_points(waypoints, N_points_per_segment_vec):
    q_traj = np.empty((2, 0))
    num_segments = waypoints.shape[1] - 1
    for i in range(num_segments):
        startPt = waypoints[:, i]
        endPt = waypoints[:, i + 1]
        num_pts_seg = N_points_per_segment_vec[i]
        x_seg = np.linspace(startPt[0], endPt[0], num_pts_seg)
        y_seg = np.linspace(startPt[1], endPt[1], num_pts_seg)
        if i == 0:
            q_traj = np.vstack((x_seg, y_seg))
        else:
            q_traj = np.hstack((q_traj, np.vstack((x_seg[1:], y_seg[1:]))))
    return q_traj

q0, qF = np.array([0, 150]), np.array([300, 150])
us, uR = np.array([60, 240]), np.array([150, 90])
uR_RL1, uR_RL2 = np.array([45, 150]), np.array([240, 150])
CU1, CU2 = np.array([220, 245]), np.array([240, 220])

wp_NR = np.array([[0, 60, 150, 220, 300], [150, 240, 240, 230, 150]])
N_NR = [7, 6, 5, 8]
q_NR = generate_trajectory_points(wp_NR, N_NR)

wp_PS1 = np.array([[0, 50, 90, 150, 190, 220, 250, 300], [150,170,190,210,220,245,180,150]])
N_PS1 = [3, 3, 3, 3, 3, 4, 5]
q_PS1 = generate_trajectory_points(wp_PS1, N_PS1)

wp_PS0 = np.array([[0, 60, 90, 150, 210, 250, 300], [150,230,200,180,160,155,150]])
N_PS0 = [4, 4, 4, 4, 3, 4]
q_PS0 = generate_trajectory_points(wp_PS0, N_PS0)

wp_PS05 = np.array([[0, 60, 80, 120, 160, 200, 240, 300], [150,220,170,130,150,180,160,150]])
N_PS05 = [5, 4, 4, 3, 3, 4, 5]
q_PS05 = generate_trajectory_points(wp_PS05, N_PS05)

fig, ax = plt.subplots(figsize=(10, 7))
ax.grid(True, linestyle='--', alpha=0.4)

colors = {
    'NR': (0.9290, 0.6940, 0.1250),
    'PS1': (0, 0.4470, 0.7410),
    'PS0': (0.8500, 0.3250, 0.0980),
    'PS05': (0.4660, 0.6740, 0.1880)
}

ax.plot(q_PS05[0], q_PS05[1], '-d', color=colors['PS05'], label='PS β_C = 0.5', linewidth=2)
ax.plot(q_PS0[0], q_PS0[1], '-s', color=colors['PS0'], label='PS β_C = 0', linewidth=2)
ax.plot(q_PS1[0], q_PS1[1], '-^', color=colors['PS1'], label='PS β_C = 1', linewidth=2)
ax.plot(q_NR[0], q_NR[1], '-*', color=colors['NR'], label='NR β_C = 0.5', linewidth=2)

for pt, label in zip([us, uR, uR_RL1, uR_RL2, CU1, CU2], ['Target', 'RIS', 'RL1', 'RL2', 'CU1', 'CU2']):
    ax.scatter(*pt, s=120, c='darkblue', edgecolors='k')
    ax.text(pt[0], pt[1]+8, label, ha='center', fontsize=9)

ax.plot(*q0, 'ko')
ax.text(*q0 - np.array([10, 10]), 'Initial Point', ha='right', fontsize=8)
ax.plot(*qF, 'ko')
ax.text(*qF + np.array([10, -10]), 'Final Point', ha='left', fontsize=8)

ax.set_xlim(-20, 320)
ax.set_ylim(60, 310)
ax.set_xticks(np.arange(0, 301, 30))
ax.set_yticks(np.arange(90, 301, 30))
ax.set_xlabel('x(n) (m)')
ax.set_ylabel('y(n) (m)')
ax.legend(loc='lower left')
ax.set_title('Fig. 2: UAV Trajectories under Different β_C')
plt.tight_layout()
plt.show()
