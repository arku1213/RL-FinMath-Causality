import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import math
import time

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------------
# Replay buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Actor & Critic
# -----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=4.0):
        super(Actor, self).__init__()
        self.max_action = float(max_action)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # weight init
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        nn.init.xavier_uniform_(self.log_std.weight, gain=0.01)
    
    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action_tanh = torch.tanh(x_t)  # in [-1,1]
        action_scaled = (action_tanh + 1.0) * (self.max_action / 2.0)  # [0, max_action]
        
        # log prob correction for tanh + affine scaling
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - action_tanh.pow(2)) * (self.max_action / 2.0) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action_scaled, log_prob
    
    def deterministic(self, state):
        mean, _ = self.forward(state)
        action_tanh = torch.tanh(mean)
        action_scaled = (action_tanh + 1.0) * (self.max_action / 2.0)
        return action_scaled

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

# -----------------------------
# SAC Agent
# -----------------------------
class SAC:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, tau=0.01, alpha=0.05, max_action=4.0):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.actor = Actor(state_dim, action_dim, max_action=max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate:
            action = self.actor.deterministic(state)
            return action.detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=128):
        if len(replay_buffer) < batch_size:
            return None, None, None  # losses not computed yet
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # Actor update
        new_action, log_prob = self.actor.sample(state)
        q1_pi = self.critic1(state, new_action)
        q2_pi = self.critic2(state, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_prob - q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update targets
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic1_loss.item(), critic2_loss.item(), actor_loss.item()

# -----------------------------
# Environment: normalized
# -----------------------------
class MultiStepBinomialEnvironment:
    """
    Normalized multi-step binomial with sequential binary trading.
    Prices normalized: S0=1.0, K=1.0 so payoffs are O(1).
    """
    def __init__(self, S0=1.0, K=1.0, r=0.05, T=1.0, u=1.2, d=0.8, n_steps=2, max_action=4.0,
                 penalty_coeff=10.0, penalty_cap=100.0):
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.T = T
        self.u = float(u)
        self.d = float(d)
        self.n_steps = int(n_steps)
        self.dt = T / n_steps
        self.p = (np.exp(r * self.dt) - d) / (u - d)
        self.discount = np.exp(-r * self.dt)
        self.price_up = self.discount * self.p
        self.price_down = self.discount * (1 - self.p)
        self.max_action = float(max_action)
        self.penalty_coeff = float(penalty_coeff)
        self.penalty_cap = float(penalty_cap)
        self.reset()
        
        print("="*60)
        print(f"{n_steps}-STEP BINOMIAL - SEQUENTIAL BINARY TRADING (normalized, max_action={self.max_action})")
        print("="*60)
        print(f"S0={self.S0}, K={self.K}, r={self.r}, T={self.T}, u={self.u}, d={self.d}")
        print(f"Risk-neutral p={self.p:.4f}")
        print(f"Binary prices: up={self.price_up:.4f}, down={self.price_down:.4f}")
        print("="*60)
    
    def reset(self):
        self.current_step = 0
        self.current_price = self.S0
        self.path = []
        self.binaries_held = {}
        self.total_cost = 0.0
        return self._get_state()
    
    def _get_state(self):
        path_indicator = 0
        if len(self.path) > 0:
            path_indicator = 1 if self.path[-1] == 'U' else -1
        return np.array([self.current_price / self.S0, self.current_step / self.n_steps, path_indicator],
                        dtype=np.float32)
    
    def step(self, action):
        # action: array-like (b_up, b_down), assumed scaled already in [0, max_action]
        b_up = float(np.clip(action[0], 0.0, self.max_action))
        b_down = float(np.clip(action[1], 0.0, self.max_action))
        
        cost = b_up * self.price_up + b_down * self.price_down
        self.total_cost += cost
        
        # store holdings by node
        if self.current_step == 0:
            self.binaries_held['b1'] = b_up
            self.binaries_held['b2'] = b_down
        elif len(self.path) == 1 and self.path[0] == 'U':
            self.binaries_held['b3'] = b_up
            self.binaries_held['b4'] = b_down
        elif len(self.path) == 1 and self.path[0] == 'D':
            self.binaries_held['b5'] = b_up
            self.binaries_held['b6'] = b_down
        
        self.current_step += 1
        
        # stochastic move according to risk-neutral p
        went_up = np.random.random() < self.p
        if went_up:
            self.current_price *= self.u
            self.path.append('U')
        else:
            self.current_price *= self.d
            self.path.append('D')
        
        done = (self.current_step >= self.n_steps)
        if done:
            reward = self._terminal_reward()
        else:
            # small intermediate penalty proportional to cost (keeps magnitude small)
            reward = -0.01 * cost
        
        next_state = self._get_state()
        info = {
            'step': self.current_step,
            'price': self.current_price,
            'path': ''.join(self.path),
            'binaries': self.binaries_held.copy(),
            'total_cost': self.total_cost
        }
        return next_state, reward, done, info
    
    def _terminal_reward(self):
        # Payoff in normalized units
        payoff_current = max(self.current_price - self.K, 0.0)
        path_str = ''.join(self.path)
        portfolio_value = 0.0
        if path_str == 'UU':
            portfolio_value = self.binaries_held.get('b3', 0.0)
        elif path_str == 'UD':
            portfolio_value = self.binaries_held.get('b4', 0.0)
        elif path_str == 'DU':
            portfolio_value = self.binaries_held.get('b5', 0.0)
        elif path_str == 'DD':
            portfolio_value = self.binaries_held.get('b6', 0.0)
        
        violation = max(0.0, payoff_current - portfolio_value)
        
        # intermediate t=1 check (normalized)
        intermediate_price = self.S0 * (self.u if self.path[0] == 'U' else self.d)
        intermediate_payoff = max(intermediate_price - self.K, 0.0)
        intermediate_portfolio = self.binaries_held.get('b1' if self.path[0] == 'U' else 'b2', 0.0)
        intermediate_violation = max(0.0, intermediate_payoff - intermediate_portfolio)
        
        total_violation = violation + intermediate_violation
        
        # Linear, capped penalty (more stable than squared explosion)
        penalty = self.penalty_coeff * total_violation
        penalty = min(penalty, self.penalty_cap)
        
        reward = - self.total_cost - penalty
        # return also some diagnostics via attributes if needed
        return reward

# -----------------------------
# Training & evaluation
# -----------------------------
def train_sac(env, sac, n_episodes=3000, batch_size=128, print_every=500, warmup_steps=2000):
    replay_buffer = ReplayBuffer()
    episode_rewards = []
    episode_violations = []
    avg_costs = []
    
    # Warm-up
    steps_collected = 0
    print(f"Prefilling replay buffer with {warmup_steps} random transitions...")
    while steps_collected < warmup_steps:
        state = env.reset()
        done = False
        while not done and steps_collected < warmup_steps:
            # uniform random action in [0, max_action]
            a = np.random.uniform(0.0, env.max_action, size=(2,))
            next_state, reward, done, info = env.step(a)
            replay_buffer.push(state, a, reward, next_state, done)
            state = next_state
            steps_collected += 1
    print("Warmup complete. Starting training...")
    
    # Running averages for logs
    running_start = time.time()
    critic1_losses = []
    critic2_losses = []
    actor_losses = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        episode_cost = 0.0
        episode_total_violation = 0.0
        steps = 0
        
        while not done:
            action = sac.select_action(state, evaluate=False)
            # exploration noise (small)
            action = np.clip(action + np.random.normal(scale=0.08, size=action.shape), 0.0, env.max_action)
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            losses = sac.update(replay_buffer, batch_size)
            if losses[0] is not None:
                critic1_losses.append(losses[0])
                critic2_losses.append(losses[1])
                actor_losses.append(losses[2])
            
            state = next_state
            episode_reward += reward
            steps += 1
            episode_cost = info['total_cost']
        
        episode_rewards.append(episode_reward)
        avg_costs.append(episode_cost)
        # violation if reward is very negative (heuristic)
        episode_violations.append(1 if episode_reward < -1.0 else 0)
        
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_violation = np.mean(episode_violations[-print_every:])
            avg_cost = np.mean(avg_costs[-print_every:])
            avg_c1 = np.mean(critic1_losses[-100:]) if len(critic1_losses) > 0 else float('nan')
            avg_c2 = np.mean(critic2_losses[-100:]) if len(critic2_losses) > 0 else float('nan')
            avg_a = np.mean(actor_losses[-100:]) if len(actor_losses) > 0 else float('nan')
            t_elapsed = time.time() - running_start
            print(f"Episode {episode+1}/{n_episodes}  |  AvgR (last {print_every}): {avg_reward:.3f}  |  Violation Rate: {avg_violation*100:.1f}%  |  AvgCost: {avg_cost:.3f}  |  Time: {t_elapsed:.1f}s")
            print(f"  Recent losses (c1,c2,actor): {avg_c1:.4f}, {avg_c2:.4f}, {avg_a:.4f}")
            # show sample last trajectory info (best-effort)
            try:
                print(f"  Sample last path: {info['path']}, binaries: {info['binaries']}, total_cost: {info['total_cost']:.3f}")
            except Exception:
                pass
    
    return episode_rewards

def evaluate_policy(env, sac, n_episodes=100):
    print("\n" + "="*60)
    print("EVALUATING LEARNED POLICY")
    print("="*60)
    total_rewards = []
    violations = []
    sample_episodes = []
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        actions_taken = []
        while not done:
            action = sac.select_action(state, evaluate=True)
            action = np.clip(action, 0.0, env.max_action)
            actions_taken.append(action)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
        total_rewards.append(ep_reward)
        violations.append(1 if ep_reward < -1.0 else 0)
        if ep < 5:
            sample_episodes.append({'actions': actions_taken, 'path': info['path'], 'binaries': info['binaries'], 'price': info['price'], 'reward': ep_reward})
    
    avg_reward = np.mean(total_rewards)
    violation_rate = np.mean(violations)
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"  Violation Rate: {violation_rate*100:.1f}%")
    print(f"  Success Rate: {(1-violation_rate)*100:.1f}%")
    print("\n  Sample episodes:")
    for i, ep in enumerate(sample_episodes):
        print(f"    Ep {i+1}: Path={ep['path']}, Price={ep['price']:.3f}, Reward={ep['reward']:.3f}")
        print(f"      Actions: {[[f'{a:.3f}' for a in act] for act in ep['actions']]}")
        print(f"      Binaries: {ep['binaries']}")
    return avg_reward, violation_rate

# -----------------------------
# main
# -----------------------------
def main():
    # You can increase n_steps to 3 or 5 to test scaling. Keep in mind training time grows.
    n_steps = 2  # change to 3 or 5 for experiments
    env = MultiStepBinomialEnvironment(n_steps=n_steps, S0=1.0, K=1.0, u=1.2, d=0.8,
                                      max_action=4.0, penalty_coeff=10.0, penalty_cap=100.0)
    state_dim = 3
    action_dim = 2
    sac = SAC(state_dim, action_dim, lr=1e-3, alpha=0.05, tau=0.01, max_action=env.max_action)
    
    print("\nTraining SAC...")
    train_sac(env, sac, n_episodes=2000, batch_size=128, print_every=500, warmup_steps=2000)
    
    evaluate_policy(env, sac, n_episodes=100)
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
