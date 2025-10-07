import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Binomial model parameters
S0 = 100
K = 100
r = 0.05
T = 1.0
u = 1.2
d = 0.8
dt = T / 2

# Risk-neutral probability
p = (np.exp(r * dt) - d) / (u - d)

# Stock prices and payoffs
prices = {
    'T2_UU': S0 * u * u,
    'T2_UD': S0 * u * d,
    'T2_DU': S0 * d * u,
    'T2_DD': S0 * d * d
}

payoffs = {
    'UU': max(prices['T2_UU'] - K, 0),
    'UD': max(prices['T2_UD'] - K, 0),
    'DU': max(prices['T2_DU'] - K, 0),
    'DD': max(prices['T2_DD'] - K, 0)
}

# Theoretical price
path_probs = {
    'UU': p * p,
    'UD': p * (1-p),
    'DU': (1-p) * p,
    'DD': (1-p) * (1-p)
}
theoretical_price = np.exp(-r * T) * sum(path_probs[path] * payoffs[path] for path in ['UU', 'UD', 'DU', 'DD'])

# Binary prices
binary_prices = {
    'b1': p * np.exp(-r * dt),
    'b2': (1-p) * np.exp(-r * dt),
    'b3': p * np.exp(-r * dt),
    'b4': (1-p) * np.exp(-r * dt),
    'b5': p * np.exp(-r * dt),
    'b6': (1-p) * np.exp(-r * dt),
}

print("="*60)
print("PURE DDPG FOR T=2 BINOMIAL DYNAMIC HEDGING")
print("="*60)
print(f"S0={S0}, K={K}, r={r}, T={T}, u={u}, d={d}")
print(f"Risk-neutral p={p:.4f}")
print(f"\nPayoffs: UU=${payoffs['UU']:.2f}, UD=${payoffs['UD']:.2f}, DU=${payoffs['DU']:.2f}, DD=${payoffs['DD']:.2f}")
print(f"Theoretical Price: ${theoretical_price:.2f}")
print("="*60 + "\n")

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Initialize with small weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        # Map tanh output [-1,1] to [0, 50]
        return self.net(state) * 25 + 25

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

# Environment
class BinomialEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.t = 0
        self.path = []
        self.binaries = {}
        self.state_price = S0
        return self._get_state()
    
    def _get_state(self):
        if self.t == 0:
            return np.array([0.0, S0/100, 0.0, 2.0], dtype=np.float32)
        elif self.t == 1:
            path_ind = 1.0 if self.path[-1] == 'U' else -1.0
            return np.array([1.0, self.state_price/100, path_ind, 1.0], dtype=np.float32)
        else:
            return np.array([2.0, self.state_price/100, 0.0, 0.0], dtype=np.float32)
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.array(action).flatten()
        
        if self.t == 0:
            self.binaries['b1'] = float(action[0])
            self.binaries['b2'] = float(action[1])
            
            next_move = 'U' if np.random.rand() < p else 'D'
            self.path.append(next_move)
            self.state_price = S0 * u if next_move == 'U' else S0 * d
            self.t = 1
            
            return self._get_state(), 0.0, False
        
        elif self.t == 1:
            if self.path[-1] == 'U':
                self.binaries['b3'] = float(action[0])
                self.binaries['b4'] = float(action[1])
            else:
                self.binaries['b5'] = float(action[0])
                self.binaries['b6'] = float(action[1])
            
            next_move = 'U' if np.random.rand() < p else 'D'
            self.path.append(next_move)
            self.t = 2
            
            reward = self._calculate_reward()
            return self._get_state(), reward, True
    
    def _calculate_reward(self):
        # Calculate cost
        cost = sum(self.binaries.get(f'b{i}', 0) * binary_prices[f'b{i}'] for i in range(1, 7))
        
        # Calculate violations
        violations = []
        max_violation = 0
        for scenario in ['UU', 'UD', 'DU', 'DD']:
            pv = self._get_portfolio_value(scenario)
            target = payoffs[scenario]
            if pv < target - 0.01:
                shortfall = target - pv
                violations.append(shortfall)
                max_violation = max(max_violation, shortfall)
        
        # Reward structure
        # 1. Minimize cost
        cost_penalty = abs(cost - theoretical_price)
        
        # 2. Heavy penalty for violations
        violation_penalty = sum(violations) * 1000 if violations else 0
        
        # 3. Bonus for success
        bonus = 0
        if not violations:
            if 0.95 * theoretical_price <= cost <= 1.05 * theoretical_price:
                bonus = 100
            else:
                bonus = 10  # Small bonus for feasibility even if costly
        
        reward = -cost_penalty - violation_penalty + bonus
        
        return reward
    
    def _get_portfolio_value(self, scenario):
        value = 0
        if scenario[0] == 'U':
            value += self.binaries.get('b1', 0)
        else:
            value += self.binaries.get('b2', 0)
        
        if scenario == 'UU':
            value += self.binaries.get('b3', 0)
        elif scenario == 'UD':
            value += self.binaries.get('b4', 0)
        elif scenario == 'DU':
            value += self.binaries.get('b5', 0)
        elif scenario == 'DD':
            value += self.binaries.get('b6', 0)
        
        return value
    
    def get_metrics(self):
        cost = sum(self.binaries.get(f'b{i}', 0) * binary_prices[f'b{i}'] for i in range(1, 7))
        violations = []
        for scenario in ['UU', 'UD', 'DU', 'DD']:
            pv = self._get_portfolio_value(scenario)
            target = payoffs[scenario]
            if pv < target - 0.01:
                violations.append(scenario)
        
        return {'cost': cost, 'violations': violations, 'binaries': dict(self.binaries)}

# DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer()
    
    def select_action(self, state, noise=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        
        if noise > 0:
            action += np.random.normal(0, noise * 25, size=action.shape)  # Noise scaled to action range
            action = np.clip(action, 0, 50)
        
        return action
    
    def update(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Update Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()

# Training
def train_ddpg(num_episodes=20000, batch_size=128):
    env = BinomialEnv()
    state_dim = 4
    action_dim = 2
    
    agent = DDPG(state_dim, action_dim)
    
    episode_rewards = []
    episode_costs = []
    episode_violations = []
    
    print("TRAINING PURE DDPG")
    print("="*60 + "\n")
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        noise = max(0.1, 1.0 - episode / 10000)  # Decay noise
        
        done = False
        trajectory = []
        
        while not done:
            action = agent.select_action(state, noise=noise)
            next_state, reward, done = env.step(action)
            
            trajectory.append((state, action, reward, next_state, done))
            
            episode_reward += reward
            state = next_state
        
        # Add to replay buffer
        for transition in trajectory:
            agent.replay_buffer.push(*transition)
        
        # Update networks
        if len(agent.replay_buffer) >= batch_size:
            for _ in range(2):  # Multiple updates per episode
                agent.update(batch_size)
        
        # Track metrics
        metrics = env.get_metrics()
        episode_rewards.append(episode_reward)
        episode_costs.append(metrics['cost'])
        episode_violations.append(1 if metrics['violations'] else 0)
        
        if episode % 1000 == 0:
            window = 100
            avg_reward = np.mean(episode_rewards[-window:])
            avg_cost = np.mean(episode_costs[-window:])
            viol_rate = np.mean(episode_violations[-window:]) * 100
            
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Cost: ${avg_cost:.2f} (Target: ${theoretical_price:.2f})")
            print(f"  Violation Rate: {viol_rate:.1f}%")
            print(f"  Noise: {noise:.3f}")
            print(f"  Sample: {metrics['binaries']}")
            print()
    
    return agent

# Train
agent = train_ddpg()

# Evaluate
print("="*60)
print("FINAL EVALUATION")
print("="*60 + "\n")

env = BinomialEnv()
eval_metrics = []

for _ in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, noise=0)
        state, reward, done = env.step(action)
    
    eval_metrics.append(env.get_metrics())

avg_cost = np.mean([m['cost'] for m in eval_metrics])
viol_rate = np.mean([1 if m['violations'] else 0 for m in eval_metrics]) * 100
costs = [m['cost'] for m in eval_metrics]

print(f"Results over 100 episodes:")
print(f"  Average Cost: ${avg_cost:.2f}")
print(f"  Theoretical: ${theoretical_price:.2f}")
print(f"  Cost Error: {abs(avg_cost - theoretical_price)/theoretical_price * 100:.2f}%")
print(f"  Violation Rate: {viol_rate:.1f}%")
print(f"  Within ±5%: {sum(1 for c in costs if 0.95*theoretical_price <= c <= 1.05*theoretical_price)}%")

print(f"\nSample solutions:")
for i in range(3):
    print(f"  {i+1}. {eval_metrics[i]['binaries']}")
    print(f"     Cost: ${eval_metrics[i]['cost']:.2f}, Violations: {eval_metrics[i]['violations']}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)