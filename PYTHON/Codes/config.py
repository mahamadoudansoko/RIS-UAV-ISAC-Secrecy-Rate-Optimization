# config.py

import numpy as np

# --- System and Environment Parameters ---
SYSTEM_PARAMS = {
    "V_ANTENNAS": 25,          # Number of ISAC-UAV Antennas
    "M_RIS_ELEMENTS": 100,     # Number of RIS Elements
    "K_USERS": 2,              # Number of CUs
    "H_UAV": 100.0,            # Fixed altitude for all UAVs (m)
    "H_RIS": 15.0,             # RIS Altitude (m)
    "V_MAX": 20.0,             # Max UAV Speed (m/s)
    "P_C_MAX_dBm": 36.0,       # Max ISAC-UAV Power (dBm)
    "P_J_MAX_dBm": 30.0,       # Max J-UAV Power (dBm)
    "P_C_MAX_W": 10**(36.0 / 10) / 1000, # Watts
    "P_J_MAX_W": 10**(30.0 / 10) / 1000, # Watts
    "USER_LOCATIONS": np.array([[230, 250], [250, 230]]),
    "TARGET_LOCATION": np.array([60, 235]),
    "RIS_LOCATION": np.array([150, 90]),
    "EVE_LOCATION_ESTIMATE": np.array([50, 50]), # Estimated for planning
    "UAV_C_START": np.array([0, 150]),
    "UAV_C_END": np.array([300, 150]),
    "UAV_J_START": np.array([0, 120]), # Start J-UAV slightly offset
    "UAV_J_END": np.array([300, 120]),
}

# --- Channel Model Parameters ---
CHANNEL_PARAMS = {
    "BETA_0_dB": -30,           # Path loss at 1m (dB)
    "BETA_0": 10**(-30 / 10),   # Linear scale
    "KAPPA_RICIAN_dB": 3,       # Rician Factor (dB)
    "NOISE_POWER_dBm": -114,
    "NOISE_POWER_W": 10**(-114 / 10) / 1000,
    "ALPHA_LOS": 2.2,           # Path Loss Exponent (LoS)
    "ALPHA_NLOS": 2.8,          # Path Loss Exponent (NLoS/Rician)
}

# --- QoS and Task Parameters ---
TASK_PARAMS = {
    "R_K_MIN": 2.5,             # Min CU Rate (bits/s/Hz)
    "GAMMA_S_MIN": 5.0,         # Min Sensing SNR (linear, not dB)
    "EPISODE_DURATION_S": 50.0, # Episode Duration (s)
    "DT_S": 0.5,                # Time Slot Duration (s)
    "N_STEPS": int(50.0 / 0.5),
}

# --- DRL Hyperparameters ---
DRL_PARAMS = {
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "GAMMA": 0.99,
    "TAU": 0.005,
    "BUFFER_SIZE": 1_000_000,
    "BATCH_SIZE": 128,
    "DEVICE": "cpu", # "cuda" or "cpu"
    "NUM_EPISODES": 500,
}

# --- Reward Weights ---
# These need careful tuning
REWARD_WEIGHTS = {
    "W_SEC": 1.0,               # Primary objective: secrecy rate
    "LAMBDA_S": 0.2,            # Penalty for low sensing SNR
    "LAMBDA_Q": 0.2,            # Penalty for low user QoS
    "LAMBDA_P": 0.01,           # Penalty for power consumption
    "LAMBDA_C": 50.0,           # Penalty for collision
    "D_MIN": 10.0,              # Min safety distance between UAVs
    # Beta_c for communication-sensing tradeoff (0 to 1)
    # This is not a penalty but a direct weight on the objective components
    # This will be varied during evaluation
    "BETA_C": 0.5
}

# --- File Paths ---
PATHS = {
    "MODEL_SAVE_DIR": "./models/",
    "PLOT_SAVE_DIR": "./plots/"
}