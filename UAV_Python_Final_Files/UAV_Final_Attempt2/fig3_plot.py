import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --- Data Definitions (from your MATLAB code) ---
ps_data = np.array([[5.0, 11.5], [6.0, 11.25], [7.0, 11.15], [8.0, 11.0], [9.5, 10.75], [10.25, 10.60], [11.25, 10.5], [11.75, 9.5], [13.0, 8.0], [13.75, 6.5], [14.5, 5.0]])
rps_data = np.array([[4.0, 9.25], [5.25, 9], [6.0, 8.75], [7.75, 8.25], [8.250, 8.0], [9.0, 7.5], [9.75, 7.0], [10.5, 6.5], [11.0, 5.25], [11.25, 4.75], [11.35, 4.25]])
nr_data = np.array([[3.75, 8.5], [5.0, 8.25], [5.75, 8.15], [6.75, 8.0], [7.60, 7.5], [8.5, 7.25], [9.0, 6.5], [9.5, 5.75], [10.25, 4.70], [10.70, 3.75], [10.80, 3.5]])
sf_data = np.array([[3.0, 7.30], [3.50, 7.250], [4.25, 7.0], [5.150, 6.60], [5.50, 6.5], [6.250, 6.25], [6.5, 5.75], [7.5, 4.75], [7.750, 4.0], [8.150, 3.75], [8.250, 3]])

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6.5))

# Colors and styles
ps_color, rps_color, nr_color, sf_color = '#77AC30', '#EDB120', '#D95319', '#0072BD'
line_width_curves, marker_size_curves = 2.0, 7

# Plot Data Curves
ax.plot(ps_data[:,0], ps_data[:,1], 'd-', color=ps_color, markerfacecolor=ps_color, markeredgecolor='k', lw=line_width_curves, ms=marker_size_curves, label='PS')
ax.plot(rps_data[:,0], rps_data[:,1], '^-', color=rps_color, markerfacecolor=rps_color, markeredgecolor='k', lw=line_width_curves, ms=marker_size_curves, label='RPS')
ax.plot(nr_data[:,0], nr_data[:,1], 'p-', color=nr_color, markerfacecolor=nr_color, markeredgecolor='k', lw=line_width_curves, ms=marker_size_curves+1, label='NR')
ax.plot(sf_data[:,0], sf_data[:,1], 'o-', color=sf_color, markerfacecolor=sf_color, markeredgecolor='k', lw=line_width_curves, ms=marker_size_curves, label='SF')

# Annotations
ax.text(ps_data[0,0] - 0.2, ps_data[0,2] + 0.3, r'$\beta_C=1$', fontsize=11, ha='right')
ax.text(ps_data[-1,0] + 0.2, ps_data[-1,2] - 0.2, r'$\beta_C=0$', fontsize=11, ha='left')

# Curved arrow for "Increasing beta_C"
x_arc, y_arc = [12.0, 10.0, 8.5], [9.8, 11.0, 10.8]
spl = make_interp_spline(x_arc, y_arc)
x_smooth = np.linspace(min(x_arc), max(x_arc), 100)
y_smooth = spl(x_smooth)
ax.plot(x_smooth, y_smooth, 'k-', lw=1.2)
ax.arrow(x_smooth[0], y_smooth[0], x_smooth[1]-x_smooth[0], y_smooth[1]-y_smooth[0], head_width=0.2, head_length=0.3, fc='k', ec='k')
ax.text(10.5, 11.5, r'Increasing $\beta_C$', fontsize=11, ha='center')

# Gain Annotations with arrows
ax.annotate('Gain from the\nintroduction of RIS', xy=(6.5, 9.5), xytext=(3, 10.5),
            arrowprops=dict(arrowstyle='<->', color=nr_color, lw=1.5, linestyle='--'),
            fontsize=10, color=nr_color, ha='center')
ax.annotate('Gain from the\nphase shift design', xy=(9, 9), xytext=(11, 7.5),
            arrowprops=dict(arrowstyle='<->', color=rps_color, lw=1.5, linestyle='--'),
            fontsize=10, color=rps_color, ha='center')
ax.annotate('Gain from the\ntrajectory design', xy=(6, 8.5), xytext=(8, 6),
            arrowprops=dict(arrowstyle='<->', color=sf_color, lw=1.5, linestyle='--'),
            fontsize=10, color=sf_color, ha='center')

# Axis, Legend, and Layout
ax.set_xlabel('Average Sensing SNR (dB)', fontsize=12)
ax.set_ylabel("CU's Average Sum-Rate (bits/s/Hz)", fontsize=12)
ax.set_xlim(2, 17)
ax.set_ylim(2, 13)
ax.legend(loc='lower left', fontsize=11)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.show()