import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# --- Helper Function ---
def generate_trajectory(waypoints, points_per_segment):
    path = []
    for i in range(len(waypoints) - 1):
        start_pt = np.array(waypoints[i])
        end_pt = np.array(waypoints[i+1])
        num_points = points_per_segment[i]
        segment = np.linspace(start_pt, end_pt, num_points)
        if i > 0:
            segment = segment[1:]
        path.extend(segment)
    return np.array(path)

# --- Waypoints and Trajectories ---
initial_point = [0, 150]
final_point = [300, 150]
target = [60, 240]
ris = [150, 90]
rl1 = [60, 150]
rl2 = [240, 150]
cu1 = [220, 245]
cu2 = [240, 220]

traj_nr = generate_trajectory([initial_point, target, [230, 235], final_point], [8, 8, 8])
traj_ps1 = generate_trajectory([initial_point, [150, 210], cu1, [260, 200], final_point], [10, 8, 5, 8])
traj_ps0 = generate_trajectory([initial_point, [65, 225], [100, 190], [180, 170], final_point], [6, 6, 8, 10])
traj_ps05 = generate_trajectory([initial_point, [50, 220], [120, 130], [200, 140], final_point], [8, 8, 6, 10])

# --- Setup Figure ---
fig, ax = plt.subplots(figsize=(9, 7))
colors = {'nr': '#fec62e', 'ps1': '#0072BD', 'ps0': '#D95319', 'ps05': '#77AC30'}
segments = {
    'NR βc=0.5': traj_nr,
    'PS βc=1': traj_ps1,
    'PS βc=0': traj_ps0,
    'PS βc=0.5': traj_ps05
}

# Initialize plot lines
lines = {}
for label in segments:
    key = label.split()[0].lower()  # 'nr' or 'ps'
    if key == 'nr':
        color = colors['nr']
    elif '1' in label:
        color = colors['ps1']
    elif '0' in label and '0.5' not in label:
        color = colors['ps0']
    elif '0.5' in label:
        color = colors['ps05']
    else:
        color = 'black'
    lines[label], = ax.plot([], [], lw=2, color=color, label=label)

# Static points
points = {
    "Target": target,
    "CU1": cu1,
    "CU2": cu2,
    "RIS": ris,
    "RL1": rl1,
    "RL2": rl2
}
for name, coord in points.items():
    ax.scatter(*coord, s=120, marker='*' if 'RL' in name or name == 'RIS' else 'o', label=name)

ax.annotate('Initial Point', xy=initial_point, xytext=(-20, 135),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=10, ha='center')
ax.annotate('Final Point', xy=final_point, xytext=(280, 135),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=10, ha='center')

# --- Plot Settings ---
ax.set_xlim(0, 300)
ax.set_ylim(80, 300)
ax.set_xlabel('x(n) (m)')
ax.set_ylabel('y(n) (m)')
ax.set_title('Sequential UAV Trajectory Animation')
ax.legend(loc='lower right', fontsize=10)

# --- Animation Control ---
frames_per_traj = 60
frame_ranges = {}
start = 0
for key in segments:
    end = start + frames_per_traj
    frame_ranges[key] = (start, end)
    start = end
total_frames = start

def animate(frame):
    for label, (start, end) in frame_ranges.items():
        traj = segments[label]
        if frame < start:
            lines[label].set_data([], [])
        elif start <= frame < end:
            idx = frame - start + 1
            lines[label].set_data(traj[:idx, 0], traj[:idx, 1])
        else:
            lines[label].set_data(traj[:, 0], traj[:, 1])
    return list(lines.values())

ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=100, blit=True)

plt.tight_layout()

writer = FFMpegWriter(fps=10)
ani.save("uav_animation_trajector.mp4", writer=writer)
print("✅ Saved as uav_animation_trajector.mp4")

plt.show()





