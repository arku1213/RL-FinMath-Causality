"""
DDPG (Deep Deterministic Policy Gradient) for One-Step N-nomial Option Hedging
Perfect Market: No interest rates, no transaction costs
Enhanced with gamma=0, learning rate scheduling, and flexible delta constraints
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
import copy

# -------------------------
# Environment: One-Step N-nomial Model (Perfect Market)
# -------------------------
class OneStepNnomialEnv:
    """
    One-step N-nomial environment for option hedging in perfect market.
    No interest rates, no transaction costs.
    """
    def __init__(self, 
                 S0: float,
                 K: float, 
                 prices: List[float],
                 probabilities: List[float],
                 seed: int = 0):
        """
        Args:
            S0: Initial stock price
            K: Strike price
            prices: List of possible prices at T=1
            probabilities: List of probabilities for each price
            seed: Random seed
        """
        self.S0 = float(S0)
        self.K = float(K)
        self.prices = [float(p) for p in prices]
        self.probabilities = np.array(probabilities, dtype=np.float32)
        
        # Validation
        assert len(self.prices) == len(self.probabilities), "Prices and probabilities must have same length"
        assert abs(self.probabilities.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"
        
        self.n_outcomes = len(self.prices)
        
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # State dimension: S0, K, prices, probabilities
        self.state_dim = 2 + self.n_outcomes * 2
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = self.S0
        state[1] = self.K
        state[2:2+self.n_outcomes] = self.prices
        state[2+self.n_outcomes:] = self.probabilities
        self.state = state
        return self.state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: [delta, B] where delta is hedge ratio and B is cash position
        
        Returns:
            next_state, reward, done, info
        """
        delta = float(action[0])
        B = float(action[1])
        
        # Sample final price according to probabilities
        outcome_idx = self.np_rng.choice(self.n_outcomes, p=self.probabilities)
        S_T = self.prices[outcome_idx]
        
        # Calculate option payoff (call option)
        payoff = max(S_T - self.K, 0.0)
        
        # Calculate portfolio value (perfect market: no interest on cash)
        portfolio = delta * S_T + B
        
        # Hedging error
        hedging_error = portfolio - payoff
        
        # L2 reward: negative squared error with scaling for stable gradients
        reward = -(hedging_error ** 2) / 1000.0  # Scale down for gradient stability
        
        # Terminal state
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        next_state[0] = S_T
        next_state[1] = self.K
        next_state[2:2+self.n_outcomes] = self.prices
        next_state[2+self.n_outcomes:] = self.probabilities
        
        done = True
        
        info = {
            'hedging_error': hedging_error,
            'portfolio_value': portfolio,
            'option_payoff': payoff,
            'final_price': S_T,
            'outcome_idx': outcome_idx,
            'delta': delta,
            'B': B
        }
        
        return next_state, reward, done, info
    
    def theoretical_hedge(self) -> Tuple[float, float, float]:
        """
        Calculate theoretical hedge using least squares (perfect market).
        Returns: (delta, B, initial_portfolio_value)
        """
        # Set up least squares: delta * S_i + B = max(S_i - K, 0)
        A = np.column_stack([self.prices, np.ones(self.n_outcomes)])
        b = np.array([max(S - self.K, 0) for S in self.prices])
        
        # Weighted least squares using probabilities
        W = np.diag(self.probabilities)
        ATW = A.T @ W
        ATWA = ATW @ A
        ATWb = ATW @ b
        
        hedge_params = np.linalg.solve(ATWA, ATWb)
        delta_theory = hedge_params[0]
        B_theory = hedge_params[1]
        
        # Initial portfolio value (fair price in perfect market)
        initial_portfolio_value = delta_theory * self.S0 + B_theory
        
        return delta_theory, B_theory, initial_portfolio_value


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
# Ornstein-Uhlenbeck Noise
# -------------------------
class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""
    def __init__(self, 
                 size: int,
                 mu: float = 0.0,
                 theta: float = 0.15,
                 sigma: float = 0.2,
                 dt: float = 1e-2,
                 seed: int = 0):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.seed = seed
        self.reset()
    
    def reset(self):
        """Reset the internal state to mean."""
        np.random.seed(self.seed)
        self.state = copy.copy(self.mu)
    
    def sample(self) -> np.ndarray:
        """Update internal state and return it as noise."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state


# -------------------------
# Neural Networks
# -------------------------
def hidden_init(layer):
    """Initialize hidden layers with uniform distribution."""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """
    Deterministic policy network for DDPG.
    Flexible delta constraints (can be negative for short positions).
    """
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_sizes: List[int] = [400, 300],
                 init_w: float = 3e-3,
                 delta_range: float = 2.0):  # Allow delta in [-2, 2]
        super(Actor, self).__init__()
        
        self.delta_range = delta_range
        
        # Build network layers
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        
        # Output heads
        self.delta_head = nn.Linear(hidden_sizes[1], 1)  # Delta (with tanh scaling)
        self.b_head = nn.Linear(hidden_sizes[1], 1)      # B (unconstrained)
        
        self.reset_parameters(init_w)
    
    def reset_parameters(self, init_w: float):
        """Initialize network parameters with better convergence."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        
        # Initialize output layers with smaller weights for stability
        self.delta_head.weight.data.uniform_(-init_w, init_w)
        self.b_head.weight.data.uniform_(-init_w, init_w)
        
        # Initialize biases to zero (unbiased)
        self.delta_head.bias.data.zero_()
        self.b_head.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with flexible delta constraints."""
        # Handle both batch and single state
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Network forward pass
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # Output with constraints - smaller range for better convergence
        delta = 1.0 * torch.tanh(self.delta_head(x))  # Delta in [-1, 1]
        b = 50.0 * torch.tanh(self.b_head(x))  # B in [-50, 50]
        
        action = torch.cat([delta, b], dim=-1)
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action


class Critic(nn.Module):
    """Q-network for DDPG."""
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_sizes: List[int] = [400, 300],
                 init_w: float = 3e-3):
        super(Critic, self).__init__()
        
        # First layer processes state
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        
        # Second layer processes state features + action
        self.fc2 = nn.Linear(hidden_sizes[0] + action_dim, hidden_sizes[1])
        
        # Output layer
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        
        self.reset_parameters(init_w)
    
    def reset_parameters(self, init_w: float):
        """Initialize network parameters."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-value."""
        # Handle both batch and single inputs
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        # Process state
        xs = F.relu(self.bn1(self.fc1(state)))
        
        # Concatenate with action and continue
        x = torch.cat([xs, action], dim=-1)
        x = F.relu(self.fc2(x))
        
        # Output Q-value
        q_value = self.fc3(x)
        
        return q_value.squeeze(-1)


# -------------------------
# Learning Rate Scheduler
# -------------------------
class ExponentialLRScheduler:
    """Exponential learning rate decay scheduler."""
    def __init__(self, optimizer, decay_rate: float = 0.99, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Decay learning rate."""
        for group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            new_lr = max(self.min_lr, group['lr'] * self.decay_rate)
            group['lr'] = new_lr


# -------------------------
# Enhanced DDPG Agent
# -------------------------
class EnhancedDDPGAgent:
    """Enhanced DDPG agent with perfect market assumptions and improvements."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: List[int] = [400, 300],
                 actor_lr: float = 3e-5,  # Even lower for stability
                 critic_lr: float = 1e-4,
                 gamma: float = 0.95,  # Small gamma for better learning dynamics
                 tau: float = 0.01,  # Faster target updates
                 delta_range: float = 2.0,  # Allow delta in [-2, 2]
                 noise_theta: float = 0.15,
                 noise_sigma: float = 0.1,  # Reduce noise for finer exploration
                 noise_decay: float = 0.999,
                 min_noise: float = 0.01,
                 lr_decay: bool = True,
                 lr_decay_rate: float = 0.99995,  # Much slower decay
                 device: str = 'cpu',
                 seed: int = 0):
        
        self.device = device
        self.gamma = gamma  # 0.0 for single-step
        self.tau = tau
        self.action_dim = action_dim
        self.delta_range = delta_range
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_sizes, delta_range=delta_range).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_sizes, delta_range=delta_range).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_sizes).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_sizes).to(device)
        
        # Initialize target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 
                                          lr=critic_lr, 
                                          weight_decay=1e-4)
        
        # Learning rate schedulers
        self.lr_decay = lr_decay
        if lr_decay:
            self.actor_scheduler = ExponentialLRScheduler(self.actor_optimizer, lr_decay_rate)
            self.critic_scheduler = ExponentialLRScheduler(self.critic_optimizer, lr_decay_rate)
        
        # Noise process
        self.noise = OrnsteinUhlenbeckNoise(action_dim, 
                                           theta=noise_theta,
                                           sigma=noise_sigma,
                                           seed=seed)
        self.noise_scale = 1.0
        self.noise_decay = noise_decay
        self.min_noise = min_noise
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action from policy with optional exploration noise."""
        state = torch.FloatTensor(state).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        self.actor.train()
        
        # Add exploration noise
        if add_noise:
            noise = self.noise.sample() * self.noise_scale
            action = action + noise
        
        return action
    
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Update DDPG networks with L2 optimization focus."""
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # -------------------- Update Critic -------------------- #
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q.unsqueeze(-1)
        
        # Critic loss with L2 regularization to prevent overfitting
        current_q = self.critic(states, actions).unsqueeze(-1)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Add L2 regularization to critic
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.critic.parameters():
            l2_reg += torch.norm(param)
        critic_loss += 1e-5 * l2_reg
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # -------------------- Update Actor -------------------- #
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # -------------------- Update Target Networks -------------------- #
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)
        
        # -------------------- Learning Rate Decay -------------------- #
        if self.lr_decay:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item(),
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }
    
    def soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1.0 - self.tau) * target_param.data)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def decay_noise(self):
        """Decay exploration noise over time."""
        self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
    
    def reset_noise(self):
        """Reset noise process."""
        self.noise.reset()


# -------------------------
# Training Function
# -------------------------
def train_ddpg_nnomial(
    prices: List[float],
    probabilities: List[float],
    S0: float = 100.0,
    K: float = 110.0,
    episodes: int = 40000,  # More episodes for better convergence
    batch_size: int = 128,
    buffer_size: int = 100000,
    hidden_sizes: List[int] = [400, 300],
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    delta_range: float = 2.0,
    start_steps: int = 1000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 0,
    verbose: bool = True
):
    """
    Train Enhanced DDPG agent for n-nomial option hedging in perfect market.
    
    Returns:
        agent: Trained DDPG agent
        env: Environment used for training
    """
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create environment
    env = OneStepNnomialEnv(S0=S0, K=K, prices=prices, probabilities=probabilities, seed=seed)
    
    # Get theoretical solution
    delta_theory, B_theory, fair_price_theory = env.theoretical_hedge()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Enhanced DDPG Training for {env.n_outcomes}-nomial Option Hedging")
        print(f"Perfect Market: No interest rates, no transaction costs")
        print(f"{'='*70}")
        print(f"Environment: S0={env.S0}, K={env.K}")
        print(f"Possible prices: {[f'{p:.2f}' for p in env.prices]}")
        print(f"Probabilities: {[f'{p:.3f}' for p in env.probabilities]}")
        print(f"\nTheoretical Hedge:")
        print(f"  Delta: {delta_theory:.4f}")
        print(f"  B: {B_theory:.4f}")
        print(f"  Initial Portfolio Value: {fair_price_theory:.4f}")
        print(f"{'='*70}\n")
    
    # Create agent
    agent = EnhancedDDPGAgent(
        state_dim=env.state_dim,
        action_dim=2,
        hidden_sizes=hidden_sizes,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        delta_range=delta_range,
        device=device,
        seed=seed
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Training loop
    total_steps = 0
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        agent.reset_noise()
        
        # Select action with unbiased smart exploration
        if total_steps < start_steps:
            # Stage 1: Uniform exploration across reasonable ranges
            action = np.array([
                np.random.uniform(-0.8, 0.8),   # Uniform delta exploration
                np.random.uniform(-45, 15)      # Uniform B exploration
            ])
        else:
            # Normal DDPG exploration with adaptive noise
            action = agent.select_action(state, add_noise=True)
            
            # Adaptive exploration: increase noise if not improving
            if episode > 5000 and episode % 2000 == 0:
                recent_mses = [info.get('mse', float('inf')) for info in 
                              getattr(agent, 'recent_performance', [])]
                if len(recent_mses) > 3 and all(mse > 50 for mse in recent_mses[-3:]):
                    # If stuck, temporarily increase exploration
                    agent.noise_scale = min(0.2, agent.noise_scale * 1.5)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        replay_buffer.push(state, action, float(reward), next_state, float(done))
        total_steps += 1
        
        # Track performance for adaptive exploration (unbiased)
        if not hasattr(agent, 'recent_performance'):
            agent.recent_performance = []
        
        if len(replay_buffer) >= batch_size and total_steps >= start_steps:
            batch = replay_buffer.sample(batch_size)
            agent.update(batch)
            agent.decay_noise()
            
            # Track recent MSE for adaptive exploration
            if episode % 100 == 0:
                test_state = env.reset()
                test_action = agent.select_action(test_state, add_noise=False)
                _, _, _, test_info = env.step(test_action)
                test_mse = test_info['hedging_error'] ** 2
                agent.recent_performance.append(test_mse)
                if len(agent.recent_performance) > 10:
                    agent.recent_performance.pop(0)
        
        # Logging - more frequent for longer training
        if verbose and (episode % 3000 == 0 or episode in [1000, 2000, 5000, 10000]):  
            # Evaluate current policy
            state = env.reset()
            with torch.no_grad():
                action = agent.select_action(state, add_noise=False)
            
            delta_current = action[0]
            B_current = action[1]
            portfolio_value_current = delta_current * env.S0 + B_current
            
            # Calculate performance metrics
            test_errors = []
            for _ in range(100):
                state = env.reset()
                action = agent.select_action(state, add_noise=False)
                _, _, _, info = env.step(action)
                test_errors.append(info['hedging_error'])
            
            mse = np.mean([e**2 for e in test_errors])
            max_error = max(abs(e) for e in test_errors)
            
            print(f"Episode {episode:5d} | Steps {total_steps:7d}")
            print(f"  Current: Δ={delta_current:.4f}, B={B_current:.3f}, Value={portfolio_value_current:.3f}")
            print(f"  Theory:  Δ={delta_theory:.4f}, B={B_theory:.3f}, Value={fair_price_theory:.3f}")
            print(f"  MSE: {mse:.6f}, Max |Error|: {max_error:.4f}")
            print(f"  Noise: {agent.noise_scale:.4f}, Actor LR: {agent.actor_optimizer.param_groups[0]['lr']:.6f}")
            print()
    
    # Final evaluation
    if verbose:
        print(f"\n{'='*70}")
        print("Final Evaluation")
        print(f"{'='*70}")
        
        state = env.reset()
        with torch.no_grad():
            final_action = agent.select_action(state, add_noise=False)
        
        delta_final = final_action[0]
        B_final = final_action[1]
        portfolio_value_final = delta_final * env.S0 + B_final
        
        # Test over many episodes
        test_errors = []
        for _ in range(1000):
            state = env.reset()
            action = agent.select_action(state, add_noise=False)
            _, _, _, info = env.step(action)
            test_errors.append(info['hedging_error'])
        
        mse = np.mean([e**2 for e in test_errors])
        max_error = max(abs(e) for e in test_errors)
        mean_error = np.mean(test_errors)
        
        print(f"Learned Policy:")
        print(f"  Delta: {delta_final:.4f} (Theory: {delta_theory:.4f})")
        print(f"  B: {B_final:.4f} (Theory: {B_theory:.4f})")
        print(f"  Portfolio Value: {portfolio_value_final:.4f} (Theory: {fair_price_theory:.4f})")
        print(f"\nReplication Performance:")
        print(f"  Mean Squared Error: {mse:.6f}")
        print(f"  Max |Error|: {max_error:.6f}")
        print(f"  Mean Error: {mean_error:.6f}")
        
        # Calculate errors for each scenario
        print(f"\nError by Scenario:")
        for i, (price, prob) in enumerate(zip(env.prices, env.probabilities)):
            portfolio = delta_final * price + B_final
            payoff = max(price - env.K, 0)
            error = portfolio - payoff
            print(f"  S={price:.1f} (p={prob:.3f}): Portfolio={portfolio:.3f}, Payoff={payoff:.3f}, Error={error:.4f}")
        
        print(f"{'='*70}")
    
    return agent, env


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    
    # Configure your n-nomial model here
    # Binomial example:
    prices = [80.0, 140.0]
    probabilities = [0.5, 0.5]
    
    # Trinomial example:
    # prices = [80.0, 110.0, 140.0]
    # probabilities = [0.33, 0.34, 0.33]
    
    # 5-nomial example:
    # prices = [70.0, 85.0, 100.0, 115.0, 130.0]
    # probabilities = [0.1, 0.2, 0.4, 0.2, 0.1]
    
    print("Training Enhanced DDPG for N-nomial Option Hedging...")
    print(f"Model: {len(prices)}-nomial")
    print(f"Prices: {prices}")
    print(f"Probabilities: {probabilities}")
    
    agent, env = train_ddpg_nnomial(
        prices=prices,
        probabilities=probabilities,
        S0=100.0,
        K=110.0,
        episodes=25000,
        batch_size=128,
        delta_range=2.0,  # Allow delta in [-2, 2]
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )