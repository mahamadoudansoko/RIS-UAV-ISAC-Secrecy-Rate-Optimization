# In file: DDPG-MultiUT-Time/experiment_secrecy_vs_power.py

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Configuration for the Secrecy Rate vs. Power experiment ---
POWER_BUDGET_DBM = [20, 24, 28, 32, 36] # Power in dBm
FIXED_W_S = 0.5
TOTAL_TIMESTEPS = 10000

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

performance_vs_power_data = []

for power_dbm in POWER_BUDGET_DBM:
    power_watts = 10**((power_dbm - 30) / 10)
    print(f"\n\n{'='*50}")
    print(f"STARTING EXPERIMENT FOR POWER = {power_dbm} dBm ({power_watts:.3f} W)")
    print(f"{'='*50}\n")
    
    env = gym.make(
        'foo-v0', 
        multiUT=True,
        w_S=FIXED_W_S,
        P_ISAC_MAX=power_watts
    )
    env = DummyVecEnv([lambda: env])

    # Train
    model = DDPG("MlpPolicy", env, action_noise=NormalActionNoise(np.zeros(env.action_space.shape[-1]), 0.1 * np.ones(env.action_space.shape[-1])), verbose=0)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Test
    obs = env.reset()
    secrecy_rates = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, info = env.step(action)
        secrecy_rates.append(info[0]['secrecy_rate'])
        if dones[0]:
            break
            
    avg_rate = np.mean(secrecy_rates)
    performance_vs_power_data.append([power_dbm, avg_rate])
    print(f"Finished Power={power_dbm} dBm. Avg Secrecy Rate: {avg_rate:.3f}")

# Save summary. The CSV filename clearly indicates what was tested.
df = pd.DataFrame(performance_vs_power_data, columns=['power_dbm', 'avg_secrecy_rate'])
df.to_csv(os.path.join(output_dir, 'summary_secrecy_vs_power.csv'), index=False)
print("\n'Secrecy vs. Power' experiment finished. Summary data saved.")