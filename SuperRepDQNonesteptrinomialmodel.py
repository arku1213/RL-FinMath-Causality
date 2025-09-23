# PURE REPLICATION WITH DQN
# Discrete action space: (Δ, B)
# Δ ∈ [-1, 1] with 21 steps, B ∈ [-50, 50] with 21 steps
# Total actions = 441

import math
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
                 mid_price=110.0,   # new middle branch
                 down_price=80.0,
                 probs=(1/3, 1/3, 1/3),  # probabilities for (up, mid, down)
                 seed=0):
        self.S0 = float(S0)
        self.K = float(K)
        self.up_price = float(up_price)
        self.mid_price = float(mid_price)
        self.down_price = float(down_price)
        self.probs = probs
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.state = np.array([self.S0, 1.0], dtype=np.float32)
        return self.state

    def step(self, delta, B):
        # stochastic move up / mid / down
        r = self.rng.random()
        if r < self.probs[0]:
            S_T = self.up_price
            branch = "up"
        elif r < self.probs[0] + self.probs[1]:
            S_T = self.mid_price
            branch = "mid"
        else:
            S_T = self.down_price
            branch = "down"

        # call payoff
        payoff = max(S_T - self.K, 0.0)

        # portfolio
        portfolio = delta * S_T + B

        # error
        err = portfolio - payoff
        shortfall = max(0.0, payoff - portfolio)

        # --- reward choices ---
        #reward = -(shortfall**2)       # smooth L2 penalty
        reward = -100 if shortfall > 0 else 0   # hard penalty

        next_state = np.array([S_T, 0.0], dtype=np.float32)
        done = True
        info = {
            "branch": branch,
            "S_T": S_T,
            "payoff": payoff,
            "portfolio": portfolio,
            "err": err
        }
        return next_state, reward, done, info

# -------------------------
# Discretization of action space
# -------------------------
# Δ in [-1,1] with 21 steps, B in [-50,50] with 21 steps
deltas = np.linspace(-1.0, 1.0, 21)
Bs = np.linspace(-50.0, 50.0, 21)
action_space = [(d, b) for d in deltas for b in Bs]
NUM_ACTIONS = len(action_space)  # 441


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
            nn.Linear(hidden, NUM_ACTIONS)   # one Q-value per action
        )

    def forward(self, x):
        return self.net(x)  # shape: [batch, NUM_ACTIONS]


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
    episodes=50000,
    batch_size=64,
    gamma=1.0,          # no discounting since it's 1-step
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=5000,
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

    # epsilon-greedy exploration
    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay

    rewards_history = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randrange(NUM_ACTIONS)
        else:
            with torch.no_grad():
                q_values = qnet(state_tensor)
                action_idx = q_values.argmax(dim=1).item()

        delta, B = action_space[action_idx]

        # step environment
        next_state, reward, done, info = env.step(delta, B)

        # store transition
        replay.push(state, action_idx, reward, next_state, done)

        state = next_state
        rewards_history.append(reward)

        # decay epsilon
        if epsilon > epsilon_end:
            epsilon -= epsilon_decay_rate

        # learn if enough samples
        if len(replay) >= batch_size:
            states, actions, rewards, next_states, dones = replay.sample(batch_size)

            states_tensor = torch.tensor(states, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.int64)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)

            # current Q estimates
            q_values = qnet(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            # target Q
            with torch.no_grad():
                next_q_values = target_qnet(next_states_tensor).max(dim=1)[0]
                targets = rewards_tensor + gamma * (1 - dones_tensor) * next_q_values

            loss = F.mse_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update target network
        if ep % target_update == 0:
            target_qnet.load_state_dict(qnet.state_dict())

        # logging
        if verbose and ep % 5000 == 0:
            mean_r = np.mean(rewards_history[-5000:])
            print(f"Episode {ep}, mean reward (last 5000): {mean_r:.4f}, epsilon={epsilon:.3f}")

    return qnet, env


# -------------------------
# Evaluation
# -------------------------
if __name__ == "__main__":
    qnet, env = train_dqn(episodes=100000, seed=42, verbose=True)

    # Evaluate best action at S0
    s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = qnet(s0)
        best_idx = q_values.argmax(dim=1).item()

    delta, B = action_space[best_idx]
    fair_price_est = delta * env.S0 + B

    print("\n--- Final policy at S0 ---")
    print(f"Δ = {delta:.4f}, B = {B:.4f}")
    print(f"Implied fair price X = Δ * S0 + B = {fair_price_est:.4f}")

    # Replication error test
    N = 1000
    errs = []
    for _ in range(N):
        _, _, _, info = env.step(delta, B)
        errs.append(info["err"])
    print(f"Mean abs replication error over {N} sims: {np.mean(np.abs(errs)):.6f}")
