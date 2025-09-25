"""
Multi-Step SAC (Soft Actor-Critic) for N-nomial Option Hedging
Perfect Market: No interest rates, no transaction costs
Configurable time steps and n-nomial outcomes
Completely unbiased implementation with entropy regularization
FIXED: Removed BatchNorm to avoid single-sample training issues
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# Multi-Step N-nomial Environment
# -------------------------
class MultiStepNnomialEnv:
    """Multi-step N-nomial environment for dynamic option hedging."""
    
    def __init__(self, 
                 S0: float,
                 K: float,
                 T: int,  # Number of time steps
                 prices_per_step: List[float],  # Possible prices at each step
                 probabilities: List[float],    # Probabilities for price moves
                 n_outcomes: int,               # Number of outcomes per step
                 seed: int = 0):
        
        self.S0 = float(S0)
        self.K = float(K)
        self.T = T  # Total time steps
        self.dt = 1.0 / T  # Time per step
        
        # Price evolution parameters
        self.prices_per_step = [float(p) for p in prices_per_step]
        self.probabilities = np.array(probabilities, dtype=np.float32)
        self.n_outcomes = n_outcomes
        
        # Validation
        assert len(self.prices_per_step) == n_outcomes, "Must have n_outcomes prices"
        assert len(self.probabilities) == n_outcomes, "Must have n_outcomes probabilities"
        assert abs(self.probabilities.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"
        
        self.np_rng = np.random.RandomState(seed)
        
        # State: [current_price, time_remaining, moneyness, portfolio_value, option_intrinsic_value]
        self.state_dim = 5
        
        # Initialize episode variables
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_time = 0
        self.current_price = self.S0
        self.portfolio_value = 0.0  # Will be set by first action
        
        # Calculate initial option intrinsic value
        option_intrinsic = max(self.current_price - self.K, 0.0)
        
        state = np.array([
            self.current_price / 100.0,  # Normalized current price
            (self.T - self.current_time) / self.T,  # Normalized time remaining
            self.current_price / self.K,  # Moneyness
            0.0,  # Initial portfolio value (normalized)
            option_intrinsic / 100.0  # Normalized option intrinsic value
        ], dtype=np.float32)
        
        self.state = state
        return self.state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a rebalancing step in the environment."""
        delta = float(action[0])  # New hedge ratio
        B = float(action[1])      # New cash position
        
        # Update portfolio value based on action
        self.portfolio_value = delta * self.current_price + B
        
        # Advance time
        self.current_time += 1
        done = (self.current_time >= self.T)
        
        if not done:
            # Sample next price based on current price and n-nomial model
            price_multiplier_idx = self.np_rng.choice(self.n_outcomes, p=self.probabilities)
            price_multiplier = self.prices_per_step[price_multiplier_idx]
            
            # Update price
            new_price = self.current_price * price_multiplier
            self.current_price = new_price
            
            # Calculate option intrinsic value at new price and time
            option_intrinsic = max(self.current_price - self.K, 0.0)
            
            # Intermediate reward: negative rebalancing cost
            position_change = abs(delta - getattr(self, 'previous_delta', 0.5))
            rebalancing_cost = -0.001 * position_change  # Reduced cost
            reward = rebalancing_cost
            
            # Store previous delta for next step
            self.previous_delta = delta
            
            # Next state
            next_state = np.array([
                self.current_price / 100.0,
                (self.T - self.current_time) / self.T,
                self.current_price / self.K,
                self.portfolio_value / 100.0,
                option_intrinsic / 100.0
            ], dtype=np.float32)
            
            info = {
                'hedging_error': 0.0,  # Not applicable for intermediate steps
                'portfolio_value': self.portfolio_value,
                'option_payoff': option_intrinsic,
                'current_price': self.current_price,
                'delta': delta,
                'B': B,
                'time_step': self.current_time,
                'rebalancing_cost': rebalancing_cost
            }
            
        else:
            # Terminal step - calculate final hedging error
            option_payoff = max(self.current_price - self.K, 0.0)
            final_portfolio = delta * self.current_price + B
            
            hedging_error = final_portfolio - option_payoff
            
            # Terminal reward: negative squared hedging error (scaled)
            reward = -(hedging_error ** 2) / 100.0  # Scaled for better learning
            
            # Terminal state (all zeros to indicate episode end)
            next_state = np.zeros(self.state_dim, dtype=np.float32)
            
            info = {
                'hedging_error': hedging_error,
                'portfolio_value': final_portfolio,
                'option_payoff': option_payoff,
                'current_price': self.current_price,
                'delta': delta,
                'B': B,
                'time_step': self.current_time,
                'squared_error': hedging_error ** 2,
                'is_terminal': True
            }
        
        self.state = next_state
        return next_state, reward, done, info
    
    def get_theoretical_price_path_stats(self) -> Dict[str, float]:
        """Get statistics about possible price paths for analysis."""
        expected_multiplier = np.sum(self.probabilities * self.prices_per_step)
        expected_final_price = self.S0 * (expected_multiplier ** self.T)
        
        log_multipliers = np.log(self.prices_per_step)
        expected_log_return = np.sum(self.probabilities * log_multipliers)
        var_log_return = np.sum(self.probabilities * (log_multipliers - expected_log_return) ** 2)
        
        return {
            'expected_final_price': expected_final_price,
            'expected_log_return_per_step': expected_log_return,
            'variance_log_return_per_step': var_log_return,
            'implied_volatility': np.sqrt(var_log_return * self.T)
        }


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
# SAC Networks (Fixed - No BatchNorm)
# -------------------------
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6

class QNetwork(nn.Module):
    """Q-value network for SAC (twin networks) - No BatchNorm."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        input_dim = obs_dim + act_dim
        
        # Build network without BatchNorm
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm instead of BatchNorm
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights without bias."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller initial weights
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
    """Gaussian policy network for SAC with entropy regularization - No BatchNorm."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        # Build network without BatchNorm
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm instead of BatchNorm
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, act_dim)
        self.log_std_head = nn.Linear(prev_dim, act_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights without bias toward solution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Very small weights for output heads to start unbiased
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
        """Sample action with reparameterization trick and apply constraints."""
        mean, log_std = self.forward(s)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply constraints for hedging - more flexible ranges
        # Delta: sigmoid for [0, 1] but allow wider initial exploration
        delta = torch.sigmoid(x_t[:, 0:1])
        
        # B: tanh with scaling for wider range
        B = 50.0 * torch.tanh(x_t[:, 1:2])  # Range [-50, 50] initially
        
        action = torch.cat([delta, B], dim=-1)
        
        # Calculate log probability with Jacobian correction
        log_prob = normal.log_prob(x_t).sum(axis=-1, keepdim=True)
        
        # Jacobian correction for sigmoid (delta)
        log_prob -= torch.log(delta * (1 - delta) + EPS).sum(axis=-1, keepdim=True)
        
        # Jacobian correction for tanh (B) 
        tanh_B_raw = torch.tanh(x_t[:, 1:2])
        log_prob -= torch.log((1 - tanh_B_raw.pow(2)) * 50 + EPS).sum(axis=-1, keepdim=True)
        
        return action, log_prob
    
    def deterministic_action(self, s: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean with constraints applied)."""
        mean, _ = self.forward(s)
        
        # Apply same constraints as in sample()
        delta = torch.sigmoid(mean[:, 0:1])
        B = 50.0 * torch.tanh(mean[:, 1:2])
        
        return torch.cat([delta, B], dim=-1)


# -------------------------
# Multi-Step SAC Agent
# -------------------------
class MultiStepSACAgent:
    """SAC agent with entropy regularization for multi-step option hedging."""
    
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
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + 
                                     list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -float(act_dim)  # Standard heuristic
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action from policy."""
        state = torch.FloatTensor(state).to(self.device)
        
        if evaluate:
            # Use deterministic action for evaluation
            with torch.no_grad():
                action = self.policy.deterministic_action(state)
        else:
            # Sample stochastic action for exploration
            with torch.no_grad():
                action, _ = self.policy.sample(state)
        
        return action.squeeze(0).cpu().numpy()
    
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Update SAC networks with entropy regularization."""
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current alpha value
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
        
        # Update temperature (alpha)
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            'alpha': alpha.item() if self.automatic_entropy_tuning else alpha,
            'q_value': q1_pred.mean().item()
        }
    
    def soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + 
                                  (1 - self.tau) * target_param.data)


# -------------------------
# Training Function
# -------------------------
def train_multistep_sac(
    T: int = 3,                    # Number of time steps
    n_outcomes: int = 3,           # Number of outcomes per step  
    prices_per_step: List[float] = [0.9, 1.0, 1.1],  # Price multipliers
    probabilities: List[float] = [0.25, 0.5, 0.25],   # Probabilities
    S0: float = 100.0,
    K: float = 110.0,
    episodes: int = 20000,         # Episodes for SAC convergence
    batch_size: int = 256,         # SAC typically uses larger batches
    buffer_size: int = 100000,
    hidden_dims: List[int] = [256, 256],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    verbose: bool = True
):
    """Train SAC agent for multi-step n-nomial option hedging."""
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create environment
    env = MultiStepNnomialEnv(
        S0=S0, K=K, T=T,
        prices_per_step=prices_per_step,
        probabilities=probabilities,
        n_outcomes=n_outcomes,
        seed=seed
    )
    
    # Get theoretical analysis
    price_stats = env.get_theoretical_price_path_stats()
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Multi-Step SAC Training for {T}-Step {n_outcomes}-nomial Option Hedging")
        print(f"Perfect Market - Entropy-Regularized Dynamic Rebalancing")
        print(f"{'='*80}")
        print(f"Environment: S0={S0}, K={K}, T={T} steps")
        print(f"Price multipliers per step: {prices_per_step}")
        print(f"Probabilities: {probabilities}")
        print(f"Expected final price: {price_stats['expected_final_price']:.2f}")
        print(f"Implied volatility: {price_stats['implied_volatility']:.4f}")
        print(f"{'='*80}\n")
    
    # Create agent
    agent = MultiStepSACAgent(
        obs_dim=env.state_dim,
        act_dim=2,
        hidden_dims=hidden_dims,
        device=device
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    hedging_errors = []
    
    # Training loop
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action (stochastic for exploration)
            action = agent.select_action(state, evaluate=False)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Update agent (start learning after some experience)
            if len(replay_buffer) >= batch_size and episode > 50:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)
            
            state = next_state
            
            # Track terminal hedging error
            if done:
                if 'hedging_error' in info:
                    hedging_errors.append(abs(info['hedging_error']))
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Logging
        if verbose and episode % 2500 == 0:
            # Evaluate current policy
            eval_rewards = []
            eval_errors = []
            
            for _ in range(10):  # Multiple evaluation episodes
                state = env.reset()
                eval_reward = 0
                
                while True:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = env.step(action)
                    eval_reward += reward
                    state = next_state
                    
                    if done:
                        if 'hedging_error' in info:
                            eval_errors.append(abs(info['hedging_error']))
                        break
                
                eval_rewards.append(eval_reward)
            
            avg_reward = np.mean(eval_rewards)
            avg_error = np.mean(eval_errors) if eval_errors else 0
            recent_training_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            # Get recent alpha value
            if len(replay_buffer) >= batch_size and episode > 50:
                dummy_batch = replay_buffer.sample(batch_size)
                update_info = agent.update(dummy_batch)
                current_alpha = update_info['alpha']
            else:
                current_alpha = 0.2
            
            print(f"Episode {episode:5d}")
            print(f"  Training reward: {recent_training_reward:.4f}")
            print(f"  Eval reward: {avg_reward:.4f}")
            print(f"  Avg hedging error: {avg_error:.4f}")
            print(f"  Alpha (entropy weight): {current_alpha:.4f}")
            print(f"  Replay buffer size: {len(replay_buffer)}")
            print()
    
    # Final evaluation
    if verbose:
        print(f"\n{'='*80}")
        print("Final Evaluation")
        print(f"{'='*80}")
        
        final_rewards = []
        final_errors = []
        final_paths = []
        
        # Run multiple evaluation episodes
        for eval_ep in range(100):
            state = env.reset()
            episode_path = []
            episode_reward = 0
            
            while True:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                episode_path.append({
                    'time_step': info.get('time_step', 0),
                    'price': info.get('current_price', 0),
                    'delta': info.get('delta', 0),
                    'B': info.get('B', 0),
                    'portfolio': info.get('portfolio_value', 0)
                })
                
                state = next_state
                
                if done:
                    if 'hedging_error' in info:
                        final_errors.append(info['hedging_error'])
                    break
            
            final_rewards.append(episode_reward)
            if eval_ep < 3:  # Store first 3 paths for analysis
                final_paths.append(episode_path)
        
        print(f"Final Performance (100 episodes):")
        print(f"  Mean reward: {np.mean(final_rewards):.6f}")
        print(f"  Std reward: {np.std(final_rewards):.6f}")
        print(f"  Mean |hedging error|: {np.mean(np.abs(final_errors)):.6f}")
        print(f"  Std |hedging error|: {np.std(np.abs(final_errors)):.6f}")
        print(f"  Max |hedging error|: {np.max(np.abs(final_errors)):.6f}")
        print(f"  MSE: {np.mean(np.array(final_errors)**2):.6f}")
        
        print(f"\nSample Episode Path:")
        for step in final_paths[0]:
            print(f"  Step {step['time_step']}: Price={step['price']:.2f}, "
                  f"Δ={step['delta']:.4f}, B={step['B']:.2f}, "
                  f"Portfolio={step['portfolio']:.2f}")
        
        print(f"{'='*80}")
    
    return agent, env, episode_rewards, hedging_errors


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    
    # Configure your multi-step n-nomial model
    T = 3  # Number of time steps
    n_outcomes = 3  # Trinomial per step
    prices_per_step = [0.9, 1.0, 1.1]  # Down, stay, up
    probabilities = [0.33, 0.33, 0.34]  # Probabilities for each outcome
    
    print("Training Multi-Step SAC for N-nomial Option Hedging...")
    print(f"Configuration: {T} steps, {n_outcomes}-nomial per step")
    print(f"Price multipliers: {prices_per_step}")
    print(f"Probabilities: {probabilities}")
    
    agent, env, rewards, errors = train_multistep_sac(
        T=T,
        n_outcomes=n_outcomes,
        prices_per_step=prices_per_step,
        probabilities=probabilities,
        S0=100.0,
        K=110.0,
        episodes=20000,
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )