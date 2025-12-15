# In file: UAV_Env/experiment_secrecy_vs_power.py

import os
import numpy as np
import pandas as pd
import gymnasium as gym # <--- CORRECTED IMPORT
import gym_foo
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Configuration ---
POWER_BUDGET_DBM = [20, 24, 28, 32, 36]
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
    
    env = gym.make('foo-v0', w_S=FIXED_W_S, P_ISAC_MAX=power_watts)
    env = DummyVecEnv([lambda: env])

    MODEL_NAME = f"ddpg_secrecy_isac_P_{power_dbm}dBm"
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=0)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    obs, info = env.reset() # Gymnasium reset returns (obs, info)
    secrecy_rates = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated[0] or truncated[0]
        secrecy_rates.append(info[0]['secrecy_rate'])
            
    avg_rate = np.mean(secrecy_rates)
    performance_vs_power_data.append([power_dbm, avg_rate])
    print(f"Finished Power={power_dbm} dBm. Avg Secrecy Rate: {avg_rate:.3f}")
    env.close()

df = pd.DataFrame(performance_vs_power_data, columns=['power_dbm', 'avg_secrecy_rate'])
df.to_csv(os.path.join(output_dir, 'summary_secrecy_vs_power.csv'), index=False)
print("\n'Secrecy vs. Power' experiment finished. Summary data saved.")