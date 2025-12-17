# ddpg_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections
import random
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        return q1

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = collections.deque(maxlen=int(max_size))

    def add(self, state, action, reward, next_state, done):
        self.storage.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.storage)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, drl_params):
        self.device = torch.device(drl_params['DEVICE'])
        self.lr_actor = drl_params['LR_ACTOR']
        self.lr_critic = drl_params['LR_CRITIC']
        self.gamma = drl_params['GAMMA']
        self.tau = drl_params['TAU']

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def learn(self, replay_buffer, batch_size=100):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{filename}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{filename}_critic.pth"))

    def load(self, directory, filename):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic.pth")))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())