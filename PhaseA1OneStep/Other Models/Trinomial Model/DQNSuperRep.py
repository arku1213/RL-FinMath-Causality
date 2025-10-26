# SUPER REPLICATION WITH DQN
# One-step trinomial model
# Discrete action space: (Δ, B)
# Δ ∈ [-1, 1] with 21 steps, B ∈ [-50, 50] with 21 steps
# Total actions = 441

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# -------------------------
# Environment: 1-step trinomial
# -------------------------
class OneStepTrinomialEnv:
    def __init__(self,
                 S0=100.0,
                 K=110.0,
                 up_price=140.0,
                 mid_price=100.0,
                 down_price=80.0,
                 up_prob=0.33,
                 mid_prob=0.34,
                 down_prob=0.33,
                 seed=0):
        self.S0 = float(S0)
        self.K = float(K)
        self.up_price = float(up_price)
        self.mid_price = float(mid_price)
        self.down_price = float(down_price)
        self.up_prob = float(up_prob)
        self.mid_prob = float(mid_prob)
        self.down_prob = float(down_prob)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.state = np.array([self.S0, 1.0], dtype=np.float32)
        return self.state

    def step(self, delta, B):
        # Choose one of the three outcomes
        outcome = self.rng.choices(['up', 'mid', 'down'], weights=[self.up_prob, self.mid_prob, self.down_prob], k=1)[0]
        
        if outcome == 'up':
            S_T = self.up_price
        elif outcome == 'mid':
            S_T = self.mid_price
        else: # outcome == 'down'
            S_T = self.down_price

        # Option payoff
        payoff = max(S_T - self.K, 0.0)

        # Portfolio value at expiration
        portfolio = delta * S_T + B

        # Replication metrics
        err = portfolio - payoff
        shortfall = max(0.0, payoff - portfolio)
        cost = delta * self.S0 + B

        # Reward function: penalize shortfall most, then L2 error and cost
        w1 = 1000.0
        w2 = 1.0
        w3 = 1.0
        reward = -(w1 * shortfall + w2 * (err**2) + w3 * cost)

        # Next state
        next_state = np.array([S_T, 0.0], dtype=np.float32)
        done = True

        info = {
            "S_T": S_T,
            "payoff": payoff,
            "portfolio": portfolio,
            "err": err,
            "shortfall": shortfall,
            "cost": cost,
            "reward": reward
        }
        return next_state, reward, done, info

# -------------------------
# Discretization of action space
# -------------------------
deltas = np.linspace(-1.0, 1.0, 21)
Bs = np.linspace(-50.0, 50.0, 21)
action_space = [(d, b) for d in deltas for b in Bs]
NUM_ACTIONS = len(action_space)

# -------------------------
# Q-network
# -------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -------------------------
# DQN Training
# -------------------------
def train_dqn(
    episodes=200000,
    batch_size=64,
    gamma=1.0,
    lr=5e-4,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=20000,
    target_update=1000,
    buffer_capacity=100000,
    seed=0,
    verbose=True
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = OneStepTrinomialEnv(seed=seed)
    obs_dim = 2

    qnet = QNet(obs_dim)
    target_qnet = QNet(obs_dim)
    target_qnet.load_state_dict(qnet.state_dict())

    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay

    rewards_history = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if random.random() < epsilon:
            action_idx = random.randrange(NUM_ACTIONS)
        else:
            with torch.no_grad():
                q_values = qnet(state_tensor)
                action_idx = q_values.argmax(dim=1).item()

        delta, B = action_space[action_idx]

        next_state, reward, done, info = env.step(delta, B)
        replay.push(state, action_idx, reward, next_state, done)

        rewards_history.append(reward)

        if epsilon > epsilon_end:
            epsilon -= epsilon_decay_rate

        if len(replay) >= batch_size:
            states, actions, rewards, next_states, dones = replay.sample(batch_size)

            states_tensor = torch.tensor(states, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.int64)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)

            q_values = qnet(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = target_qnet(next_states_tensor).max(dim=1)[0]
                targets = rewards_tensor + gamma * (1 - dones_tensor) * next_q_values

            loss = F.mse_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % target_update == 0:
            target_qnet.load_state_dict(qnet.state_dict())

        if verbose and ep % 5000 == 0:
            mean_r = np.mean(rewards_history[-5000:])
            print(f"Episode {ep}, mean reward (last 5000): {mean_r:.4f}, epsilon={epsilon:.3f}")

    return qnet, env

# -------------------------
# Evaluation
# -------------------------
if __name__ == "__main__":
    qnet, env = train_dqn(seed=42)

    s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = qnet(s0)
        best_idx = q_values.argmax(dim=1).item()

    delta, B = action_space[best_idx]
    fair_price_est = delta * env.S0 + B

    print("\n--- Final policy at S0 ---")
    print(f"Δ = {delta:.4f}, B = {B:.4f}")
    print(f"Implied fair price X = Δ * S0 + B = {fair_price_est:.4f}")

    N = 1000
    errs, shortfalls, costs = [], [], []
    for _ in range(N):
        _, _, _, info = env.step(delta, B)
        errs.append(info["err"])
        shortfalls.append(info["shortfall"])
        costs.append(info["cost"])

    print(f"Mean abs replication error over {N} sims: {np.mean(np.abs(errs)):.6f}")
    print(f"Mean shortfall: {np.mean(shortfalls):.4f}")
    print(f"Mean cost: {np.mean(costs):.4f}")