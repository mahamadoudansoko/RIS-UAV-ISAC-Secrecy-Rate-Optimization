
import numpy as np
import matplotlib.pyplot as plt

# Synthetic Data
uav_power_dBm = np.arange(20, 41, 2)
uav_power_watt = 10 ** ((uav_power_dBm - 30) / 10)

# Secrecy Rate Definitions
secrecy_rate_no_jammer = 2 + 1.2 * np.log10(uav_power_watt + 1)
secrecy_rate_with_jammer = secrecy_rate_no_jammer + 0.5 - 0.1 * np.exp(-uav_power_watt * 2)

# Plotting
plt.figure(figsize=(9, 6))
plt.plot(uav_power_dBm, secrecy_rate_no_jammer, 'o-', label='Without Jamming UAV', color='tab:red', linewidth=2)
plt.plot(uav_power_dBm, secrecy_rate_with_jammer, 's-', label='With Jamming UAV', color='tab:blue', linewidth=2)

plt.xlabel('UAV Transmit Power (dBm)', fontsize=11)
plt.ylabel('Average Secrecy Rate (bits/s/Hz)', fontsize=11)
plt.title('Fig. 5: Secrecy Rate vs UAV Transmit Power', fontsize=13, weight='bold')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
