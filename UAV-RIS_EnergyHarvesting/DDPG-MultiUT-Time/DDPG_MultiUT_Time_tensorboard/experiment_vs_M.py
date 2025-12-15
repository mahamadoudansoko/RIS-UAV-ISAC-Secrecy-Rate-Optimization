# In file: DDPG-MultiUT-Time/experiment_vs_M.py

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import gym_foo

import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Configuration for this experiment ---
RIS_ELEMENT_VALUES = [25, 50, 75, 100, 125, 150]  # Values of M to test
FIXED_W_S = 0.5  # Fixed sensing weight for this experiment
TOTAL_TIMESTEPS = 10000  # Use a larger value for final runs

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# List to store summary data
performance_vs_m_data = []

for num_ris_elements in RIS_ELEMENT_VALUES:
    print(f"\n\n{'='*50}")
    print(f"STARTING EXPERIMENT FOR M = {num_ris_elements} RIS ELEMENTS")
    print(f"{'='*50}\n")

    # --- 1. Create Environment with specific number of RIS elements ---
    # We need to modify our foo_env.py to accept this argument
    # (We will do this in the next step)
    env = gym.make(
        'foo-v0', 
        multiUT=True,
        w_S=FIXED_W_S,
        NUM_RIS_ELEMENTS=num_ris_elements
    )
    env = DummyVecEnv([lambda: env])

    # --- 2. Train the Agent ---
    MODEL_NAME = f"ddpg_secrecy_isac_M_{num_ris_elements}"
    model = DDPG("MlpPolicy", env, action_noise=NormalActionNoise(np.zeros(env.action_space.shape[-1]), 0.1 * np.ones(env.action_space.shape[-1])), verbose=0)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # --- 3. Test and Log Data ---
    obs = env.reset()
    secrecy_rates, sensing_snrs = [], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, info = env.step(action)
        secrecy_rates.append(info[0]['secrecy_rate'])
        sensing_snrs.append(info[0]['sensing_snr_db'])
        if dones[0]:
            break
    
    # --- 4. Store Average Metrics ---
    avg_rate = np.mean(secrecy_rates)
    avg_snr = np.mean(sensing_snrs)
    performance_vs_m_data.append([num_ris_elements, avg_rate, avg_snr])
    print(f"Finished M={num_ris_elements}. Avg Rate: {avg_rate:.3f}, Avg SNR: {avg_snr:.2f} dB")

# --- 5. Save Summary Data ---
df = pd.DataFrame(performance_vs_m_data, columns=['M', 'avg_secrecy_rate', 'avg_sensing_snr'])
df.to_csv(os.path.join(output_dir, 'summary_vs_M.csv'), index=False)
print("\n'vs. M' experiment finished. Summary data saved.")