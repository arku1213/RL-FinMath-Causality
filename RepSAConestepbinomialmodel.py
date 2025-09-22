# -------------------------
# Fixed SAC for 1-step hedging, Δ in [0,1]
# -------------------------
import math, random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# environment (same as before)
class OneStepBinomialEnv:
    def __init__(self, S0=100., K=110., up_price=140., down_price=80., probability=0.5, seed=0):
        self.S0 = float(S0)
        self.K = float(K)
        self.up_price = float(up_price)
        self.down_price = float(down_price)
        self.probability = float(probability)
        self.rng = random.Random(seed)
        self.reset()
    def reset(self):
        self.state = np.array([self.S0, 1.0], dtype=np.float32)
        return self.state
    def step(self, action):
        delta, B = float(action[0]), float(action[1])
        S_T = self.up_price if self.rng.random() < self.probability else self.down_price
        payoff = max(S_T - self.K, 0.0)
        portfolio = delta * S_T + B
        err = portfolio - payoff
        reward = -(err**2)  # L2
        next_state = np.array([S_T, 0.0], dtype=np.float32)
        done = True
        info = {'err': err, 'portfolio': portfolio, 'payoff': payoff}
        return next_state, reward, done, info

# replay buffer
Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))
class ReplayBuffer:
    def __init__(self, capacity=50000): 
        self.buffer = deque(maxlen=capacity)
    def push(self, *args): 
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Convert to numpy arrays first, then to tensors (fixes the warning)
        states = np.array([b.state for b in batch], dtype=np.float32)
        actions = np.array([b.action for b in batch], dtype=np.float32)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.array([b.next_state for b in batch], dtype=np.float32)
        dones = np.array([b.done for b in batch], dtype=np.float32)
        
        s = torch.from_numpy(states)
        a = torch.from_numpy(actions)
        r = torch.from_numpy(rewards).unsqueeze(-1)
        ns = torch.from_numpy(next_states)
        d = torch.from_numpy(dones).unsqueeze(-1)
        return s,a,r,ns,d
    def __len__(self): 
        return len(self.buffer)

# networks
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.q = mlp([obs_dim+act_dim, hidden, hidden, 1])
    def forward(self, s, a): 
        return self.q(torch.cat([s,a], -1)).squeeze(-1)

LOG_STD_MIN, LOG_STD_MAX = -20, 2
EPS = 1e-6

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden], nn.ReLU, nn.ReLU)
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        
        # Initialize delta mean to be around 0.5 (middle of [0,1] range)
        with torch.no_grad():
            # Initialize mean for delta (first output) to 0 so sigmoid(0) = 0.5
            self.mean_head.bias[0] = 0.0
    
    def forward(self, s):
        h = self.net(s)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std, log_std
    
    def sample(self, s):
        mean, std, log_std = self.forward(s)
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()  # Use rsample for reparameterization trick
        
        # Apply transformations with proper Jacobian
        # Delta: constrain to [0,1] using tanh transformation (better than sigmoid)
        # tanh maps (-inf, inf) to (-1, 1), then scale/shift to (0, 1)
        delta_raw = raw_action[:, 0:1]
        delta = 0.5 * (torch.tanh(delta_raw) + 1)  # Maps to [0, 1]
        
        # B: keep unconstrained  
        B = raw_action[:, 1:2]
        
        action = torch.cat([delta, B], dim=-1)
        
        # Compute log probability
        log_prob = normal.log_prob(raw_action).sum(-1, keepdim=True)
        
        # Jacobian correction for tanh transformation on delta
        # d/dx [0.5 * (tanh(x) + 1)] = 0.5 * (1 - tanh²(x))
        tanh_delta = torch.tanh(delta_raw)
        jacobian_correction = torch.log(0.5 * (1 - tanh_delta.pow(2)) + EPS)
        log_prob = log_prob + jacobian_correction
        
        return action, log_prob, mean, log_std
    
    def deterministic(self, s):
        mean, _, _ = self.forward(s)
        delta = 0.5 * (torch.tanh(mean[:, 0:1]) + 1)
        B = mean[:, 1:2]
        return torch.cat([delta, B], -1)

# fast SAC training
def train_sac_fast(episodes=20000, batch_size=128, device='cpu'):
    env = OneStepBinomialEnv()
    obs_dim, act_dim = 2, 2
    
    # Initialize networks
    q1 = QNetwork(obs_dim, act_dim).to(device)
    q2 = QNetwork(obs_dim, act_dim).to(device)
    q1_target = QNetwork(obs_dim, act_dim).to(device)
    q2_target = QNetwork(obs_dim, act_dim).to(device)
    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    
    # Initialize targets
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    
    # Optimizers
    opt_q = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    opt_policy = optim.Adam(policy.parameters(), lr=3e-4)
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=3e-4)
    
    # SAC parameters
    target_entropy = -float(act_dim)
    replay = ReplayBuffer()
    tau = 5e-3
    gamma = 0.99

    def soft_update(source, target, tau): 
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    total_steps = 0
    updates = 0
    
    for ep in range(1, episodes + 1):
        s = env.reset()
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Sample action from policy
        with torch.no_grad(): 
            a, _, _, _ = policy.sample(s_tensor)
        a_np = a.squeeze(0).cpu().numpy()
        
        # Take step in environment
        next_s, r, done, info = env.step(a_np)
        replay.push(s, a_np, float(r), next_s, float(done))
        total_steps += 1

        # Training step
        if len(replay) >= batch_size:
            s_b, a_b, r_b, ns_b, d_b = replay.sample(batch_size)
            s_b = s_b.to(device)
            a_b = a_b.to(device) 
            r_b = r_b.to(device)
            ns_b = ns_b.to(device)
            d_b = d_b.to(device)
            
            with torch.no_grad():
                # Sample next actions
                next_a, next_logp, _, _ = policy.sample(ns_b)
                
                # Compute target Q values
                q1_next = q1_target(ns_b, next_a).unsqueeze(-1)
                q2_next = q2_target(ns_b, next_a).unsqueeze(-1)
                q_next_min = torch.min(q1_next, q2_next)
                
                alpha = torch.exp(log_alpha)
                q_target = r_b + (1 - d_b) * gamma * (q_next_min - alpha * next_logp)
            
            # Q-function update
            q1_pred = q1(s_b, a_b).unsqueeze(-1)
            q2_pred = q2(s_b, a_b).unsqueeze(-1)
            q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
            
            opt_q.zero_grad()
            q_loss.backward()
            opt_q.step()
            updates += 1

            # Policy update
            new_a, new_logp, _, _ = policy.sample(s_b)
            q1_new = q1(s_b, new_a).unsqueeze(-1)
            q2_new = q2(s_b, new_a).unsqueeze(-1)
            q_new_min = torch.min(q1_new, q2_new)
            
            policy_loss = (torch.exp(log_alpha) * new_logp - q_new_min).mean()
            
            opt_policy.zero_grad()
            policy_loss.backward()
            opt_policy.step()
            
            # Temperature (alpha) update
            alpha_loss = -(log_alpha * (new_logp + target_entropy).detach()).mean()
            
            opt_alpha.zero_grad()
            alpha_loss.backward()
            opt_alpha.step()
            
            # Soft update target networks
            soft_update(q1, q1_target, tau)
            soft_update(q2, q2_target, tau)

        # Periodic logging
        if ep % 2000 == 0:
            with torch.no_grad():
                s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32).unsqueeze(0).to(device)
                mean_action = policy.deterministic(s0).squeeze(0).cpu().numpy()
                delta_mean = mean_action[0]
                B_mean = mean_action[1]
                fair_price = delta_mean * env.S0 + B_mean
                alpha_val = torch.exp(log_alpha).item()
                print(f"ep {ep:6d} | Δ={delta_mean:.3f}, B={B_mean:.3f}, fair price={fair_price:.3f}, α={alpha_val:.3f}")

    return policy, q1, q2, env

# Test the training
if __name__ == "__main__":
    print("Training SAC for option hedging...")
    policy, q1, q2, env = train_sac_fast(episodes=10000)
    
    # Test final policy
    with torch.no_grad():
        s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32).unsqueeze(0)
        final_action = policy.deterministic(s0).squeeze(0).numpy()
        delta_final = final_action[0]
        B_final = final_action[1]
        fair_price_final = delta_final * env.S0 + B_final
        
        print(f"\nFinal Policy:")
        print(f"Delta: {delta_final:.4f}")
        print(f"B: {B_final:.4f}")
        print(f"Fair Price: {fair_price_final:.4f}")
        
        # Compare with Black-Scholes delta (theoretical)
        # For this simple binomial model, theoretical delta ≈ 0.5 for at-the-money options
        print(f"Delta is properly constrained in [0,1]: {0 <= delta_final <= 1}")