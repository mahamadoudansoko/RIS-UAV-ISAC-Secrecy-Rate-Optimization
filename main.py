# main.py

import torch
from tqdm import tqdm
from config import DRL_PARAMS, TASK_PARAMS, PATHS
from environment import UAV_ISAC_Environment
from ddpg_agent import DDPGAgent, ReplayBuffer
from utils import OUNoise, plot_training_rewards

def main():
    # Initialize Environment
    env = UAV_ISAC_Environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize Agent, Replay Buffer, and Noise
    agent = DDPGAgent(state_dim, action_dim, max_action, DRL_PARAMS)
    replay_buffer = ReplayBuffer(DRL_PARAMS['BUFFER_SIZE'])
    noise = OUNoise(action_dim)

    print(f"Starting DDPG training on device: {DRL_PARAMS['DEVICE']}")
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")

    episode_rewards = []

    for episode in range(DRL_PARAMS['NUM_EPISODES']):
        state = env.reset()
        noise.reset()
        episode_reward = 0
        
        pbar = tqdm(range(TASK_PARAMS['N_STEPS']), desc=f"Episode {episode+1}/{DRL_PARAMS['NUM_EPISODES']}")
        for t in pbar:
            # Select action with exploration noise
            action = agent.select_action(state)
            action = (action + noise.sample()).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state
            episode_reward += reward

            # Train the agent
            agent.learn(replay_buffer, DRL_PARAMS['BATCH_SIZE'])
            
            pbar.set_postfix({"Avg Reward": f"{episode_reward/(t+1):.2f}"})

            if done:
                break
        
        avg_reward = episode_reward / TASK_PARAMS['N_STEPS']
        episode_rewards.append(avg_reward)
        print(f"Episode: {episode+1}, Avg Reward: {avg_reward:.2f}")

        # Save model periodically
        if (episode + 1) % 50 == 0:
            agent.save(PATHS['MODEL_SAVE_DIR'], f"ddpg_model_episode_{episode+1}")
            print(f"--- Model saved at episode {episode+1} ---")

    # Final model save
    agent.save(PATHS['MODEL_SAVE_DIR'], "ddpg_model_final")
    print("--- Final model saved ---")
    
    # Plot training rewards
    plot_training_rewards(episode_rewards)

if __name__ == "__main__":
    main()