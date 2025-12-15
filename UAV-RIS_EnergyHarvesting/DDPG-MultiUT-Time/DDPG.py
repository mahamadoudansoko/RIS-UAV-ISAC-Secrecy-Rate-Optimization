# In file: DDPG-MultiUT-Time/DDPG.py

import os
import sys
import numpy as np
import pandas as pd # Import pandas for easy CSV saving

# --- Setup Paths and Imports ---
# This automatically finds the project root and adds it to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# =========================================================================
# EXPERIMENT CONFIGURATION
# =========================================================================
# List of sensing weights (w_S) we want to test. This is for Figure 3.
W_S_VALUES = [0.0, 0.2, 0.5, 0.8, 1.0] 

# Training timesteps. Increase for better results (e.g., 100000)
TOTAL_TIMESTEPS = 10000 # Using a smaller value for a quick test run

# Create a directory to save our results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# A list to store the final average metrics for the Rate-SNR plot (Fig 3)
rate_snr_tradeoff_data = []

# =========================================================================
# MAIN EXPERIMENT LOOP
# =========================================================================

for w_S in W_S_VALUES:
    print(f"\n\n{'='*50}")
    print(f"STARTING EXPERIMENT FOR w_S = {w_S}")
    print(f"{'='*50}\n")

    # --- 1. CREATE THE ENVIRONMENT WITH THE CURRENT w_S ---
    print("Creating the Secrecy ISAC Multi-User Environment...")
    env = gym.make(
        'foo-v0', 
        Train=True, 
        multiUT=True,
        w_S=w_S  # Pass the current sensing weight to the environment
    )
    env = DummyVecEnv([lambda: env])

    # --- 2. CONFIGURE AND TRAIN THE DDPG AGENT ---
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Model name is now specific to the w_S value
    MODEL_NAME = f"ddpg_secrecy_isac_wS_{w_S}"
    TENSORBOARD_LOG_PATH = "./DDPG_tensorboard_logs/"

    print(f"Configuring DDPG Agent: {MODEL_NAME}")
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=1e-4, # A slightly smaller learning rate is often more stable
        buffer_size=50000,
        batch_size=256,
        verbose=0, # Set to 0 to keep the output clean during the loop
        tensorboard_log=TENSORBOARD_LOG_PATH
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=MODEL_NAME)
    model.save(os.path.join(output_dir, MODEL_NAME)) # Save model in the results folder

    # --- 3. LOAD THE MODEL AND RUN A TEST EPISODE TO LOG DATA ---
    print("\n--- Running a Test Episode to Log Data ---")
    del model
    model = DDPG.load(os.path.join(output_dir, MODEL_NAME))

    test_env = gym.make('foo-v0', Train=False, multiUT=True, w_S=w_S)
    test_env = DummyVecEnv([lambda: test_env])

    # Create empty lists to store the data for this run
    isac_pos_log = []
    jammer_pos_log = []
    secrecy_rate_log = []
    sensing_snr_db_log = []

    obs = test_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        
        # Append data from the info dictionary to our lists
        info_dict = info[0]
        isac_pos_log.append(info_dict['isac_pos'])
        jammer_pos_log.append(info_dict['jammer_pos'])
        secrecy_rate_log.append(info_dict['secrecy_rate'])
        sensing_snr_db_log.append(info_dict['sensing_snr_db'])
        
        if dones[0]:
            break
    
    test_env.close()

    # --- 4. SAVE THE LOGGED DATA TO CSV FILES ---
    # Convert lists to numpy arrays for easier handling
    isac_pos_log = np.array(isac_pos_log)
    secrecy_rate_log = np.array(secrecy_rate_log)
    sensing_snr_db_log = np.array(sensing_snr_db_log)

    # Create a DataFrame and save trajectory data
    trajectory_df = pd.DataFrame({
        'isac_x': isac_pos_log[:, 0],
        'isac_y': isac_pos_log[:, 1]
    })
    trajectory_df.to_csv(os.path.join(output_dir, f'trajectory_wS_{w_S}.csv'), index=False)
    
    # Save performance metrics
    metrics_df = pd.DataFrame({
        'secrecy_rate': secrecy_rate_log,
        'sensing_snr_db': sensing_snr_db_log
    })
    metrics_df.to_csv(os.path.join(output_dir, f'metrics_wS_{w_S}.csv'), index=False)
    
    print(f"Data for w_S = {w_S} saved to '{output_dir}/'")

    # --- 5. STORE AVERAGE METRICS FOR THE TRADEOFF PLOT (FIG 3) ---
    avg_secrecy_rate = np.mean(secrecy_rate_log)
    avg_sensing_snr = np.mean(sensing_snr_db_log)
    rate_snr_tradeoff_data.append([w_S, avg_secrecy_rate, avg_sensing_snr])


# --- AFTER THE LOOP, SAVE THE TRADEOFF DATA ---
tradeoff_df = pd.DataFrame(rate_snr_tradeoff_data, columns=['w_S', 'avg_secrecy_rate', 'avg_sensing_snr'])
tradeoff_df.to_csv(os.path.join(output_dir, 'tradeoff_summary.csv'), index=False)

print("\n\nAll experiments finished. All data saved in 'results' folder.")