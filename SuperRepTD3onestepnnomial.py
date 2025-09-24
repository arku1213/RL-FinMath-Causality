import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# -------------------------
# Hedging Environment
# -------------------------
class HedgingEnv:
    def __init__(self, S0=100, K=100, possible_prices=None, probs=None):
        self.S0 = S0
        self.K = K
        self.possible_prices = possible_prices if possible_prices is not None else [80, 120]
        self.probabilities = probs if probs is not None else [0.5, 0.5]
        self.payoffs = [max(S - K, 0) for S in self.possible_prices]
        self.rng = random.Random(42)

    def reset(self):
        return np.array([self.S0, 1.0], dtype=np.float32)

    def step(self, action):
        delta, B = action
        # Sample one realized outcome
        S_T = self.rng.choices(self.possible_prices, weights=self.probabilities, k=1)[0]
        payoff = max(S_T - self.K, 0)
        portfolio = delta * S_T + B

        # Vectorized super-replication check
        shortfalls = np.maximum(0, np.array(self.payoffs) - (delta * np.array(self.possible_prices) + B))
        max_shortfall = np.max(shortfalls)
        cost = delta * self.S0 + B

        # Reward: penalize shortfall strongly but allow learning
        penalty_scale = 1e4
        if max_shortfall > 1e-8:
            reward = -cost - 1e3 * max_shortfall  # not 1e6
        else:
            reward = -cost  # valid hedge → minimize cost

        next_state = np.array([S_T, 0.0], dtype=np.float32)
        done = True
        info = {"shortfalls": shortfalls, "cost": cost, "max_shortfall": max_shortfall}
        return next_state, reward, done, info

# -------------------------
# Neural Networks
# -------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

# -------------------------
# TD3 Agent
# -------------------------
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4)

        self.max_action = max_action
        self.replay_buffer = deque(maxlen=1000000)
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.1
        self.noise_clip = 0.05
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        self.total_it += 1
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        # Target smoothing
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        target_Q1 = self.critic1_target(next_state, next_action)
        target_Q2 = self.critic2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

# -------------------------
# Training Loop
# -------------------------
if __name__ == "__main__":
    env = HedgingEnv(possible_prices=[80, 120], probs=[0.5, 0.5])  # Binomial
    state_dim = 2
    action_dim = 2
    max_action = 100.0
    agent = TD3(state_dim, action_dim, max_action)

    episodes = 20000
    rewards_history = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.add_transition(state, action, reward, next_state, done)
        agent.train(batch_size=64)

        rewards_history.append(reward)

        if ep % 1000 == 0:
            mean_r = np.mean(rewards_history[-1000:])
            print(f"Episode {ep}, mean reward (last 1000): {mean_r:.2f}, cost: {info['cost']:.2f}, max_shortfall: {info['max_shortfall']:.2f}")
