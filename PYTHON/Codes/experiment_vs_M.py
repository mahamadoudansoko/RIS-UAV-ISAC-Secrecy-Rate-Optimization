# In file: UAV_Env/experiment_vs_M.py

import os
import numpy as np
import pandas as pd
import gymnasium as gym # <--- CORRECTED IMPORT
import gym_foo
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Configuration ---
RIS_ELEMENT_VALUES = [25, 50, 75, 100, 125, 150]
FIXED_W_S = 0.5
TOTAL_TIMESTEPS = 10000
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
performance_vs_m_data = []

for num_ris_elements in RIS_ELEMENT_VALUES:
    print(f"\n\n{'='*50}")
    print(f"STARTING EXPERIMENT FOR M = {num_ris_elements} RIS ELEMENTS")
    print(f"{'='*50}\n")

    env = gym.make('foo-v0', w_S=FIXED_W_S, NUM_RIS_ELEMENTS=num_ris_elements)
    env = DummyVecEnv([lambda: env])

    MODEL_NAME = f"ddpg_secrecy_isac_M_{num_ris_elements}"
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=0)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    obs, info = env.reset() # Gymnasium reset returns (obs, info)
    secrecy_rates, sensing_snrs = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated[0] or truncated[0]
        secrecy_rates.append(info[0]['secrecy_rate'])
        sensing_snrs.append(info[0]['sensing_snr_db'])

    avg_rate = np.mean(secrecy_rates)
    avg_snr = np.mean(sensing_snrs)
    performance_vs_m_data.append([num_ris_elements, avg_rate, avg_snr])
    print(f"Finished M={num_ris_elements}. Avg Rate: {avg_rate:.3f}, Avg SNR: {avg_snr:.2f} dB")
    env.close()

df = pd.DataFrame(performance_vs_m_data, columns=['M', 'avg_secrecy_rate', 'avg_sensing_snr'])
df.to_csv(os.path.join(output_dir, 'summary_vs_M.csv'), index=False)
print("\n'vs. M' experiment finished. Summary data saved.")