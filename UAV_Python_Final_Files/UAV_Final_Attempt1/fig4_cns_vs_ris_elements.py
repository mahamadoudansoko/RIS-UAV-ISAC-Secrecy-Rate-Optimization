
import numpy as np
import matplotlib.pyplot as plt

# Data
M_values = np.arange(80, 150, 10)

rate_PS_RL1 = 10 + 1.3 * (1 - np.exp(-0.04 * (M_values - 80)))
rate_PS     = 9.5 + 1.2 * (1 - np.exp(-0.03 * (M_values - 80)))
rate_PS_RL2 = 11.0 + 1.6 * (1 - np.exp(-0.025 * (M_values - 80)))

snr_PS_RL1 = 8.8 + 1.6 * np.log10(M_values - 70)
snr_PS     = 9.5 + 1.8 * np.log10(M_values - 70)
snr_PS_RL2 = 10.0 + 1.9 * np.log10(M_values - 70)

colors = {
    'PS-RL1': 'blue',
    'PS': 'green',
    'PS-RL2': 'red'
}

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(M_values, rate_PS_RL1, 's-', color=colors['PS-RL1'], label='PS-RL1', linewidth=2)
axs[0].plot(M_values, rate_PS,     'o-', color=colors['PS'], label='PS', linewidth=2)
axs[0].plot(M_values, rate_PS_RL2, '^-', color=colors['PS-RL2'], label='PS-RL2', linewidth=2)
axs[0].set_ylabel('Avg. S-R (bits/s/Hz)', fontsize=11, weight='bold')
axs[0].legend(fontsize=9)
axs[0].grid(True, linestyle='--', alpha=0.4)
axs[0].annotate('', xy=(115, rate_PS_RL2[3]), xytext=(115, rate_PS[3]),
                arrowprops=dict(arrowstyle='<->', linewidth=1.2))
axs[0].text(116, (rate_PS_RL2[3] + rate_PS[3]) / 2, 'Rate gain', va='center', fontsize=9)

axs[1].plot(M_values, snr_PS_RL1, 's-', color=colors['PS-RL1'], label='PS-RL1', linewidth=2)
axs[1].plot(M_values, snr_PS,     'o-', color=colors['PS'], label='PS', linewidth=2)
axs[1].plot(M_values, snr_PS_RL2, '^-', color=colors['PS-RL2'], label='PS-RL2', linewidth=2)
axs[1].set_ylabel('Avg. Sensing SNR (dB)', fontsize=11, weight='bold')
axs[1].set_xlabel('Number of RIS Passive Elements (M)', fontsize=11, weight='bold')
axs[1].legend(fontsize=9)
axs[1].grid(True, linestyle='--', alpha=0.4)
axs[1].annotate('', xy=(115, snr_PS_RL1[3]), xytext=(115, snr_PS[3]),
                arrowprops=dict(arrowstyle='<->', linewidth=1.2))
axs[1].text(116, (snr_PS_RL1[3] + snr_PS[3]) / 2, 'SNR gain', va='center', fontsize=9)

fig.suptitle("Fig. 4: C&S Performance vs. Number of RIS Elements (M)", fontsize=13, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
