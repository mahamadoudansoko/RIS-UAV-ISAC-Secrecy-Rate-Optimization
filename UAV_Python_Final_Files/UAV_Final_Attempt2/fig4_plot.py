import numpy as np
import matplotlib.pyplot as plt

# Parameters
M_values = np.arange(80, 141, 10)

# Synthetic data functions (to match the original trends)
rate_PS_RL2 = 10 + 1.3 * (1 - np.exp(-0.04 * (M_values - 80)))
rate_PS     = 9.5 + 1.2 * (1 - np.exp(-0.03 * (M_values - 80)))
rate_PS_RL1 = 9.2 + 1.1 * (1 - np.exp(-0.025 * (M_values - 80)))

snr_PS_RL1 = 11.5 + 1.6 * np.log10((M_values - 70)/10 + 1)
snr_PS     = 10.8 + 1.8 * np.log10((M_values - 70)/10 + 1)
snr_PS_RL2 = 9.8 + 1.9 * np.log10((M_values - 70)/10 + 1)

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Colors and styles
c1, c2, c3 = '#0072BD', '#77AC30', '#D95319' # Blue, Green, Orange
lw, ms = 2.0, 7

# Top plot: Average sum-rate
ax1.plot(M_values, rate_PS_RL2, '^-', color=c1, lw=lw, ms=ms, label='PS-RL2 (Closer to CUs)')
ax1.plot(M_values, rate_PS,     'o-', color=c2, lw=lw, ms=ms, label='PS (Balanced)')
ax1.plot(M_values, rate_PS_RL1, 's-', color=c3, lw=lw, ms=ms, label='PS-RL1 (Closer to Target)')
ax1.set_ylabel("Avg. Sum-Rate (bits/s/Hz)", fontsize=12)
ax1.legend(loc='lower right')
ax1.tick_params(axis='y', labelsize=10)
ax1.annotate('', xy=(110, 10.4), xytext=(110, 11.2),
             arrowprops=dict(arrowstyle='<->', lw=1.5, color='k'))
ax1.text(112, 10.8, 'Rate Gain', ha='left', va='center', fontsize=10, fontweight='bold')

# Bottom plot: Sensing SNR
ax2.plot(M_values, snr_PS_RL1, 's-', color=c3, lw=lw, ms=ms, label='PS-RL1 (Closer to Target)')
ax2.plot(M_values, snr_PS,     'o-', color=c2, lw=lw, ms=ms, label='PS (Balanced)')
ax2.plot(M_values, snr_PS_RL2, '^-', color=c1, lw=lw, ms=ms, label='PS-RL2 (Closer to CUs)')
ax2.set_ylabel("Avg. Sensing SNR (dB)", fontsize=12)
ax2.set_xlabel("Number of RIS Passive Elements (M)", fontsize=12)
ax2.legend(loc='lower right')
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.annotate('', xy=(110, 10.4), xytext=(110, 11.9),
             arrowprops=dict(arrowstyle='<->', lw=1.5, color='k'))
ax2.text(112, 11.15, 'SNR Gain', ha='left', va='center', fontsize=10, fontweight='bold')

fig.suptitle("Performance vs. Number of RIS Elements", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()