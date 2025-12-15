import numpy as np
import matplotlib.pyplot as plt

def calculate_secrecy_rate(P_isac_dbm, P_jammer_dbm, G_isac_cu_db, G_isac_eve_db, G_jammer_cu_db, G_jammer_eve_db, N0_dbm):
    # Convert from dB/dBm to linear scale
    P_isac = 10**((P_isac_dbm - 30) / 10)
    P_jammer = 10**((P_jammer_dbm - 30) / 10) if P_jammer_dbm is not None else 0
    N0 = 10**((N0_dbm - 30) / 10)
    
    G_isac_cu = 10**(G_isac_cu_db / 10)
    G_isac_eve = 10**(G_isac_eve_db / 10)
    G_jammer_cu = 10**(G_jammer_cu_db / 10)
    G_jammer_eve = 10**(G_jammer_eve_db / 10)

    # Calculate SINR for CU and Eve
    sinr_cu = (P_isac * G_isac_cu) / (P_jammer * G_jammer_cu + N0)
    sinr_eve = (P_isac * G_isac_eve) / (P_jammer * G_jammer_eve + N0)

    # Calculate rates
    rate_cu = np.log2(1 + sinr_cu)
    rate_eve = np.log2(1 + sinr_eve)

    # Secrecy Rate (non-negative)
    secrecy_rate = np.maximum(0, rate_cu - rate_eve)
    return secrecy_rate

# --- Simulation Parameters ---
P_isac_range_dbm = np.linspace(20, 40, 50) # ISAC-UAV power from 20 to 40 dBm
P_jammer_fixed_dbm = 30 # Jammer power in dBm
N0_dbm = -114 # Noise power in dBm

# Channel Gains (plausible assumptions)
# ISAC-UAV is closer/has better beamforming to CU than Eve
G_isac_cu_db = -80
G_isac_eve_db = -90
# Jammer is closer to Eve and farther from CU
G_jammer_eve_db = -85
G_jammer_cu_db = -100

# --- Run Simulation ---
secrecy_rate_with_jammer = calculate_secrecy_rate(P_isac_range_dbm, P_jammer_fixed_dbm, G_isac_cu_db, G_isac_eve_db, G_jammer_cu_db, G_jammer_eve_db, N0_dbm)
secrecy_rate_without_jammer = calculate_secrecy_rate(P_isac_range_dbm, None, G_isac_cu_db, G_isac_eve_db, G_jammer_cu_db, G_jammer_eve_db, N0_dbm)

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(P_isac_range_dbm, secrecy_rate_with_jammer, 'o-', color='#0072BD', lw=2, ms=6, label='With Jamming UAV')
ax.plot(P_isac_range_dbm, secrecy_rate_without_jammer, 's--', color='#D95319', lw=2, ms=6, label='Without Jamming UAV')

ax.set_title('Average Secrecy Rate vs. ISAC-UAV Transmit Power', fontsize=14, fontweight='bold')
ax.set_xlabel('ISAC-UAV Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('Average Secrecy Rate (bits/s/Hz)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.show()