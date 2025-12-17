# evaluate.py

import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Make sure all imports are clean and consistent
from config import SYSTEM_PARAMS, TASK_PARAMS, DRL_PARAMS, PATHS
from environment import UAV_ISAC_Environment
from ddpg_agent import DDPGAgent
from utils import plot_multi_trajectory, plot_rate_snr_tradeoff, plot_performance_vs_m

def run_evaluation_episode(env, agent):
    """Runs one full episode with a trained agent and returns key metrics."""
    state = env.reset()
    done = False
    
    trajectory_c = []
    trajectory_j = []
    comm_rates = []
    sensing_snrs = []
    
    while not done:
        # Select action deterministically (no exploration noise)
        action = agent.select_action(state)
        state, _, done, info = env.step(action)
        
        trajectory_c.append(env.q_c.copy())
        trajectory_j.append(env.q_j.copy())
        # info['comm_rate'] is an array of rates for each user, so we sum them
        comm_rates.append(np.sum(info['comm_rate']))
        sensing_snrs.append(info['sensing_snr'])
        
    return {
        "trajectory_c": np.array(trajectory_c),
        "trajectory_j": np.array(trajectory_j),
        "avg_comm_rate": np.mean(comm_rates),
        "avg_sensing_snr": np.mean(sensing_snrs),
    }

def evaluate_model(model_name):
    """Loads a trained model and runs a full suite of evaluations."""
    
    model_path = os.path.join(PATHS['MODEL_SAVE_DIR'], model_name)
    
    # Initialize a base environment to get correct network dimensions
    base_env = UAV_ISAC_Environment()
    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]
    max_action = float(base_env.action_space.high[0])

    # Initialize agent and load its trained weights
    agent = DDPGAgent(state_dim, action_dim, max_action, DRL_PARAMS)
    try:
        agent.load(PATHS['MODEL_SAVE_DIR'], model_name)
        print(f"Successfully loaded model '{model_name}' for evaluation.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{model_path}'.")
        print("Please ensure you have trained a model by running main.py and that the model name is correct.")
        return

    # --- 1. Evaluate Rate-SNR Trade-off & Collect Trajectory Data ---
    print("\nEvaluating trade-off for different weighting factors (beta_c)...")
    tradeoff_results = {}
    beta_c_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for beta_c in tqdm(beta_c_values, desc="Evaluating Beta_C"):
        # Create a new environment for each specific beta_c value
        env = UAV_ISAC_Environment(config_override={'BETA_C': beta_c})
        tradeoff_results[beta_c] = run_evaluation_episode(env, agent)
        
    # --- 2. Generate and Save the Plots ---
    print("\nGenerating and saving plots...")

    # Plot the consolidated multi-trajectory graph
    plot_multi_trajectory(tradeoff_results, filename="multi_trajectory_tradeoff.png")
    print("-> Multi-trajectory plot saved.")

    # Plot the Rate vs. SNR trade-off curve
    plot_rate_snr_tradeoff(tradeoff_results, filename="rate_snr_tradeoff.png")
    print("-> Rate-SNR trade-off plot saved.")

    # --- 3. Evaluate Performance vs. Number of RIS Elements (M) ---
    # This section is commented out because it's fundamentally incorrect to use a single
    # trained model for environments with different state/action space dimensions.
    # The agent's neural network is built for a fixed M (e.g., 100 elements).
    # Changing M in the environment will lead to a tensor shape mismatch error.

    # print("\nEvaluating performance vs. M (NOTE: This is a conceptual test)...")
    # perf_vs_m_results = {}
    # m_values = [25, 50, 75, 100, 125, 150]
    # original_m = SYSTEM_PARAMS['M_RIS_ELEMENTS'] # Save original M
    
    # print("WARNING: The following evaluation assumes the model can generalize to different M values.")
    # print("This will likely fail if the state/action space changes. The correct approach is to train separate models.")
    
    # for m in tqdm(m_values, desc="Perf vs M"):
    #     current_params = SYSTEM_PARAMS.copy()
    #     current_params['M_RIS_ELEMENTS'] = m 
    #     try:
    #         # Re-initialize env with the new M. This will change action_dim.
    #         env = UAV_ISAC_Environment(config_override={'BETA_C': 0.5}, system_param_override=current_params)
    #         # This evaluation is only valid if the agent's architecture is independent of M,
    #         # which is not the case here. This call would likely fail.
    #         perf_vs_m_results[m] = run_evaluation_episode(env, agent)
    #     except Exception as e:
    #         print(f"\nCould not evaluate for M={m}. Error: {e}")
    #         print("This is expected, as the model architecture is fixed.")
    #         break

    # if perf_vs_m_results:
    #     plot_performance_vs_m(perf_vs_m_results, filename="performance_vs_m.png")
    #     print("-> Performance vs. M plot saved.")
        
    print("\nEvaluation complete. Plots are saved in the 'plots' directory.")


if __name__ == "__main__":
    # Specify the name of the model to load, matching the name used during saving in main.py
    model_to_evaluate = "ddpg_model_final"
    evaluate_model(model_to_evaluate)