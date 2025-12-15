import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- Helper Function ---
def generate_trajectory(waypoints, points_per_segment, height):
    path = []
    for i in range(len(waypoints) - 1):
        start_pt = np.array([*waypoints[i], height])
        end_pt = np.array([*waypoints[i+1], height])
        num_points = points_per_segment[i]
        segment = np.linspace(start_pt, end_pt, num_points)
        if i > 0:
            segment = segment[1:]
        path.extend(segment)
    return np.array(path)

# --- Define Static Points ---
initial_point = [0, 150]
final_point = [300, 150]
target = [60, 240]
ris = [150, 90]
rl1 = [60, 150]
rl2 = [240, 150]
cu1 = [220, 245]
cu2 = [240, 220]
height = 100  # UAV altitude

# --- Generate Trajectories ---
traj_nr = generate_trajectory([initial_point, target, [230, 235], final_point], [8, 8, 8], height)
traj_ps1 = generate_trajectory([initial_point, [150, 210], cu1, [260, 200], final_point], [10, 8, 5, 8], height)
traj_ps0 = generate_trajectory([initial_point, [65, 225], [100, 190], [180, 170], final_point], [6, 6, 8, 10], height)
traj_ps05 = generate_trajectory([initial_point, [50, 220], [120, 130], [200, 140], final_point], [8, 8, 6, 10], height)

# --- Setup Plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 300)
ax.set_ylim(80, 300)
ax.set_zlim(0, 120)
ax.set_xlabel('x(n) (m)')
ax.set_ylabel('y(n) (m)')
ax.set_zlabel('Altitude (m)')
ax.set_title('3D UAV Trajectory Animation')

colors = {
    'NR βc=0.5': '#fec62e',
    'PS βc=1': '#0072BD',
    'PS βc=0': '#D95319',
    'PS βc=0.5': '#77AC30'
}

segments = {
    'NR βc=0.5': traj_nr,
    'PS βc=1': traj_ps1,
    'PS βc=0': traj_ps0,
    'PS βc=0.5': traj_ps05
}

lines = {}
uavs = {}

# Plot each line and the UAV dot
for label, traj in segments.items():
    lines[label], = ax.plot([], [], [], lw=2, color=colors[label], label=label)
    uavs[label], = ax.plot([], [], [], 'o', color=colors[label], markersize=8)

# Plot static 3D points
static_points = {
    "Target": target,
    "CU1": cu1,
    "CU2": cu2,
    "RIS": ris,
    "RL1": rl1,
    "RL2": rl2
}
for label, coord in static_points.items():
    ax.scatter(*coord, height, s=60, marker='^', label=label)

ax.legend(loc='upper left', fontsize=9)

# --- Animation Frame Ranges ---
frames_per_traj = 60
frame_ranges = {}
start = 0
for key in segments:
    end = start + frames_per_traj
    frame_ranges[key] = (start, end)
    start = end
total_frames = start

# --- Animation Function ---
def animate(frame):
    for label, (start, end) in frame_ranges.items():
        traj = segments[label]
        traj_len = len(traj)
        if frame < start:
            lines[label].set_data([], [])
            lines[label].set_3d_properties([])
            uavs[label].set_data([], [])
            uavs[label].set_3d_properties([])
        elif start <= frame < end:
            idx = min(frame - start + 1, traj_len)
            lines[label].set_data(traj[:idx, 0], traj[:idx, 1])
            lines[label].set_3d_properties(traj[:idx, 2])
            uavs[label].set_data([traj[idx-1, 0]], [traj[idx-1, 1]])
            uavs[label].set_3d_properties([traj[idx-1, 2]])
        else:
            lines[label].set_data(traj[:, 0], traj[:, 1])
            lines[label].set_3d_properties(traj[:, 2])
            uavs[label].set_data([traj[-1, 0]], [traj[-1, 1]])
            uavs[label].set_3d_properties([traj[-1, 2]])
    return list(lines.values()) + list(uavs.values())

# --- Run Animation ---
ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=100, blit=True)

# --- Save as MP4 ---
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=10)
ani.save("uav_3d_animation.mp4", writer=writer)
print("✅ Saved as 'uav_3d_animation.mp4'")

# Optional: show the animation
plt.show()
