"""
Multi-Step SAC for Binomial Option Hedging with Binary Options
Risk-neutral probabilities, super-replication constraints
"""

import math
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# Multi-Step Binomial Environment with Binary Options
# -------------------------
class MultiStepBinomialEnv:
    """Multi-step binomial environment with binary options for dynamic hedging."""
    
    def __init__(self, 
                 S0: float,
                 K: float,
                 r: float,
                 T: int,  # Number of time steps
                 u: float,  # Up factor
                 d: float,  # Down factor
                 seed: int = 0):
        
        self.S0 = float(S0)
        self.K = float(K)
        self.r = r
        self.T = T
        self.dt = 1.0 / T
        
        self.u = u
        self.d = d
        
        # Calculate risk-neutral probability
        self.p = (np.exp(r * self.dt) - d) / (u - d)
        self.q = 1 - self.p
        
        print(f"Risk-neutral probabilities: p_up={self.p:.6f}, p_down={self.q:.6f}")
        
        # Binary option prices (equal to probabilities)
        self.binary_up_price = self.p
        self.binary_down_price = self.q
        
        self.np_rng = np.random.RandomState(seed)
        
        # State: [current_price/S0, time_remaining/T, moneyness]
        self.state_dim = 3
        
        # Action: [delta, B, b_up, b_down]
        self.action_dim = 4
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_time = 0
        self.current_price = self.S0
        
        state = np.array([
            self.current_price / self.S0,
            (self.T - self.current_time) / self.T,
            self.current_price / self.K
        ], dtype=np.float32)
        
        self.state = state
        return self.state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step with binary options."""
        delta = float(action[0])
        B = float(action[1])
        b_up = float(action[2])
        b_down = float(action[3])
        
        # Calculate initial hedge cost BEFORE advancing time (only at t=0)
        if self.current_time == 0:
            initial_hedge_cost = (delta * self.S0 + B + 
                                 b_up * self.binary_up_price + 
                                 b_down * self.binary_down_price)
            
            # Penalize if initial cost is negative or too high
            cost_penalty = 0.0
            if initial_hedge_cost <= 0:
                cost_penalty = -100.0 * abs(initial_hedge_cost)
            elif initial_hedge_cost > 50:
                cost_penalty = -10.0 * (initial_hedge_cost - 50)
        else:
            initial_hedge_cost = 0.0
            cost_penalty = 0.0
        
        # NOW advance time
        self.current_time += 1
        done = (self.current_time >= self.T)
        
        # Price evolution (binomial)
        if self.np_rng.rand() < self.p:
            # Up move
            new_price = self.current_price * self.u
            binary_payoff = b_up  # b_up pays 1, b_down pays 0
        else:
            # Down move
            new_price = self.current_price * self.d
            binary_payoff = b_down  # b_down pays 1, b_up pays 0
        
        self.current_price = new_price
        
        # Calculate portfolio value
        portfolio_value = delta * self.current_price + B * np.exp(self.r * self.dt) + binary_payoff
        
        if not done:
            # Intermediate step
            reward = cost_penalty
            
            next_state = np.array([
                self.current_price / self.S0,
                (self.T - self.current_time) / self.T,
                self.current_price / self.K
            ], dtype=np.float32)
            
            info = {
                'portfolio_value': portfolio_value,
                'current_price': self.current_price,
                'delta': delta,
                'B': B,
                'b_up': b_up,
                'b_down': b_down,
                'time_step': self.current_time,
                'initial_hedge_cost': initial_hedge_cost,
                'cost_penalty': cost_penalty
            }
            
        else:
            # Terminal step
            option_payoff = max(self.current_price - self.K, 0.0)
            
            # PnL = portfolio_value - option_payoff
            pnl = portfolio_value - option_payoff
            
            # Reward based on PnL (want to minimize negative PnL, i.e., hedge should cover option)
            # Negative PnL is bad (didn't cover the option)
            # Positive PnL is okay but excessive is wasteful
            if pnl < 0:
                # Heavily penalize not covering the option
                terminal_reward = -100.0 * (pnl ** 2)
            else:
                # Small penalty for excess coverage (want tight hedge)
                terminal_reward = -0.1 * (pnl ** 2)
            
            reward = terminal_reward + cost_penalty
            
            next_state = np.zeros(self.state_dim, dtype=np.float32)
            
            info = {
                'portfolio_value': portfolio_value,
                'option_payoff': option_payoff,
                'pnl': pnl,
                'current_price': self.current_price,
                'delta': delta,
                'B': B,
                'b_up': b_up,
                'b_down': b_down,
                'time_step': self.current_time,
                'initial_hedge_cost': initial_hedge_cost,
                'is_terminal': True
            }
        
        self.state = next_state
        return next_state, reward, done, info


# -------------------------
# Replay Buffer
# -------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([b.state for b in batch], dtype=np.float32)
        actions = np.array([b.action for b in batch], dtype=np.float32)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.array([b.next_state for b in batch], dtype=np.float32)
        dones = np.array([b.done for b in batch], dtype=np.float32)
        
        return (torch.from_numpy(states),
                torch.from_numpy(actions),
                torch.from_numpy(rewards).unsqueeze(-1),
                torch.from_numpy(next_states),
                torch.from_numpy(dones).unsqueeze(-1))
    
    def __len__(self):
        return len(self.buffer)


# -------------------------
# SAC Networks
# -------------------------
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6

class QNetwork(nn.Module):
    """Q-value network for SAC."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        input_dim = obs_dim + act_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        
        x = torch.cat([s, a], dim=-1)
        return self.network(x).squeeze(-1)


class GaussianPolicy(nn.Module):
    """Gaussian policy network with constrained actions."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, act_dim)
        self.log_std_head = nn.Linear(prev_dim, act_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for head in [self.mean_head, self.log_std_head]:
            nn.init.uniform_(head.weight, -1e-3, 1e-3)
            nn.init.zeros_(head.bias)
    
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        
        features = self.backbone(s)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with constraints."""
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        
        mean, log_std = self.forward(s)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Delta: [0, 1]
        delta = torch.sigmoid(x_t[:, 0:1])
        
        # B: [-50, 50]
        B = 50.0 * torch.tanh(x_t[:, 1:2])
        
        # Binary options: [-50, 50]
        b_up = 50.0 * torch.tanh(x_t[:, 2:3])
        b_down = 50.0 * torch.tanh(x_t[:, 3:4])
        
        action = torch.cat([delta, B, b_up, b_down], dim=-1)
        
        # Log probability with Jacobian corrections
        log_prob = normal.log_prob(x_t).sum(axis=-1, keepdim=True)
        
        # Jacobian for sigmoid (delta)
        log_prob -= torch.log(delta * (1 - delta) + EPS).sum(axis=-1, keepdim=True)
        
        # Jacobian for tanh transformations
        for i in [1, 2, 3]:
            tanh_val = torch.tanh(x_t[:, i:i+1])
            log_prob -= torch.log(50.0 * (1 - tanh_val.pow(2)) + EPS).sum(axis=-1, keepdim=True)
        
        return action, log_prob
    
    def deterministic_action(self, s: torch.Tensor) -> torch.Tensor:
        """Get deterministic action."""
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
            
        mean, _ = self.forward(s)
        
        delta = torch.sigmoid(mean[:, 0:1])
        B = 50.0 * torch.tanh(mean[:, 1:2])
        b_up = 50.0 * torch.tanh(mean[:, 2:3])
        b_down = 50.0 * torch.tanh(mean[:, 3:4])
        
        return torch.cat([delta, B, b_up, b_down], dim=-1)


# -------------------------
# SAC Agent
# -------------------------
class SACAgent:
    """SAC agent for binomial hedging with binary options."""
    
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True,
                 device: str = 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dims).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + 
                                     list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        if automatic_entropy_tuning:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                action = self.policy.deterministic_action(state)
        else:
            with torch.no_grad():
                action, _ = self.policy.sample(state)
        
        return action.squeeze(0).cpu().numpy()
    
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        if self.automatic_entropy_tuning:
            alpha = torch.exp(self.log_alpha)
        else:
            alpha = self.alpha
        
        # Update Q-functions
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions).unsqueeze(-1)
            q2_next = self.q2_target(next_states, next_actions).unsqueeze(-1)
            q_next_min = torch.min(q1_next, q2_next)
            
            q_target = rewards + (1 - dones) * self.gamma * (q_next_min - alpha * next_log_probs)
        
        q1_pred = self.q1(states, actions).unsqueeze(-1)
        q2_pred = self.q2(states, actions).unsqueeze(-1)
        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + 
                                       list(self.q2.parameters()), 1.0)
        self.q_optimizer.step()
        
        # Update policy
        new_actions, new_log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions).unsqueeze(-1)
        q2_new = self.q2(states, new_actions).unsqueeze(-1)
        q_new_min = torch.min(q1_new, q2_new)
        
        policy_loss = (alpha * new_log_probs - q_new_min).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update temperature
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update targets
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            'alpha': alpha.item() if self.automatic_entropy_tuning else alpha
        }
    
    def soft_update(self, source: nn.Module, target: nn.Module):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + 
                                  (1 - self.tau) * target_param.data)


# -------------------------
# Training Function
# -------------------------
def train_binomial_sac(
    T: int = 2,
    u: float = 1.2,
    d: float = 0.8,
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.05,
    episodes: int = 20000,
    batch_size: int = 256,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
):
    """Train SAC agent for binomial option hedging with binary options."""
    
    env = MultiStepBinomialEnv(S0=S0, K=K, r=r, T=T, u=u, d=d, seed=42)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Multi-Step Binomial SAC with Binary Options")
        print(f"{'='*80}")
        print(f"Environment: S0={S0}, K={K}, r={r}, T={T} steps")
        print(f"Up factor: {u}, Down factor: {d}")
        print(f"Risk-neutral prob (up): {env.p:.6f}")
        print(f"Binary option prices: up={env.binary_up_price:.6f}, down={env.binary_down_price:.6f}")
        print(f"Constraints: delta ∈ [0,1], B ∈ [-50,50], binaries ∈ [-50,50]")
        print(f"Initial hedge cost must be positive and ≤ 50")
        print(f"{'='*80}\n")
    
    agent = SACAgent(
        obs_dim=env.state_dim,
        act_dim=env.action_dim,
        hidden_dims=[256, 256],
        device=device
    )
    
    replay_buffer = ReplayBuffer(capacity=100000)
    
    episode_rewards = []
    episode_pnls = []
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            if len(replay_buffer) >= batch_size and episode > 50:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)
            
            state = next_state
            
            if done:
                if 'pnl' in info:
                    episode_pnls.append(info['pnl'])
                break
        
        episode_rewards.append(episode_reward)
        
        if verbose and episode % 2500 == 0:
            eval_pnls = []
            eval_costs = []
            eval_deltas = []
            eval_binaries = []
            
            for _ in range(100):
                state = env.reset()
                initial_cost = None
                
                while True:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = env.step(action)
                    
                    if initial_cost is None and info.get('time_step') == 1:
                        initial_cost = info.get('initial_hedge_cost', 0)
                        eval_deltas.append(info.get('delta', 0))
                        eval_binaries.append((info.get('b_up', 0), info.get('b_down', 0)))
                    
                    state = next_state
                    
                    if done:
                        eval_pnls.append(info.get('pnl', 0))
                        if initial_cost is not None:
                            eval_costs.append(initial_cost)
                        break
            
            print(f"Episode {episode:5d}")
            print(f"  Mean PnL: {np.mean(eval_pnls):.4f} ± {np.std(eval_pnls):.4f}")
            print(f"  Mean initial cost: {np.mean(eval_costs):.4f}")
            print(f"  Mean delta: {np.mean(eval_deltas):.4f}")
            if eval_binaries:
                mean_b_up = np.mean([b[0] for b in eval_binaries])
                mean_b_down = np.mean([b[1] for b in eval_binaries])
                print(f"  Mean binaries: b_up={mean_b_up:.4f}, b_down={mean_b_down:.4f}")
            print()
    
    # Final evaluation
    if verbose:
        print(f"\n{'='*80}")
        print("Final Evaluation (1000 episodes)")
        print(f"{'='*80}")
        
        final_pnls = []
        final_costs = []
        final_portfolios = []
        final_payoffs = []
        final_actions = []
        
        for _ in range(1000):
            state = env.reset()
            initial_action = None
            
            while True:
                action = agent.select_action(state, evaluate=True)
                
                if initial_action is None:
                    initial_action = action.copy()
                
                next_state, reward, done, info = env.step(action)
                state = next_state
                
                if done:
                    final_pnls.append(info['pnl'])
                    final_portfolios.append(info['portfolio_value'])
                    final_payoffs.append(info['option_payoff'])
                    if initial_action is not None:
                        final_actions.append(initial_action)
                    if 'initial_hedge_cost' in info:
                        final_costs.append(info['initial_hedge_cost'])
                    break
        
        print(f"\n*** BEST HEDGE SUMMARY ***")
        print(f"  Mean PnL: {np.mean(final_pnls):.6f}")
        print(f"  Std PnL: {np.std(final_pnls):.6f}")
        print(f"  Mean initial hedge cost: {np.mean(final_costs):.4f}")
        print(f"  Mean portfolio value: {np.mean(final_portfolios):.4f}")
        print(f"  Mean option payoff: {np.mean(final_payoffs):.4f}")
        
        if final_actions:
            best_idx = np.argmin(np.abs(final_pnls))
            best_action = final_actions[best_idx]
            print(f"\n  Best single hedge:")
            print(f"    δ = {best_action[0]:.6f}")
            print(f"    B = {best_action[1]:.6f}")
            print(f"    b_up = {best_action[2]:.6f}")
            print(f"    b_down = {best_action[3]:.6f}")
            print(f"    PnL = {final_pnls[best_idx]:.6f}")
            print(f"    Initial cost = {final_costs[best_idx]:.4f}")
        
        print(f"{'='*80}")
    
    return agent, env


if __name__ == "__main__":
    print("Training Multi-Step Binomial SAC with Binary Options...")
    
    agent, env = train_binomial_sac(
        T=2,
        u=1.2,
        d=0.8,
        S0=100.0,
        K=100.0,
        r=0.05,
        episodes=20000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    print("\nTraining completed!")