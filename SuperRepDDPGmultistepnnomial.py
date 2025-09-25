"""
Multi-Step DDPG (Deep Deterministic Policy Gradient) for N-nomial Option Hedging
Perfect Market: No interest rates, no transaction costs
Configurable time steps and n-nomial outcomes
Completely unbiased implementation
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

# -------------------------
# Multi-Step N-nomial Environment (same as TD3)
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
            rebalancing_cost = -0.01 * position_change
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
            reward = -(hedging_error ** 2) / 1000.0
            
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
# Replay Buffer (same as TD3)
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
# Ornstein-Uhlenbeck Noise for Exploration (DDPG-specific)
# -------------------------
class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for correlated exploration noise."""
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
        self.state = np.copy(self.mu)
    
    def sample(self) -> np.ndarray:
        """Update internal state and return it as noise."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state


# -------------------------
# DDPG Networks
# -------------------------
class Actor(nn.Module):
    """Actor network for DDPG multi-step hedging."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [400, 300]):
        super(Actor, self).__init__()
        
        # Build network layers
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[1], action_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with no bias toward solution."""
        # Hidden layers
        for layer in [self.fc1, self.fc2]:
            fan_in = layer.weight.data.size()[0]
            lim = 1. / np.sqrt(fan_in)
            layer.weight.data.uniform_(-lim, lim)
            layer.bias.data.uniform_(-lim, lim)
        
        # Output layer - small weights for stability, zero bias
        init_w = 3e-3
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.zero_()  # No bias toward any solution
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with action constraints."""
        # Handle both batch and single state
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Network forward pass
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        raw_output = self.output_layer(x)
        
        # Apply constraints for hedging actions
        # Delta: sigmoid for [0, 1]
        delta = torch.sigmoid(raw_output[:, 0:1])
        
        # B: tanh with wider range for multi-step
        B = 100.0 * torch.tanh(raw_output[:, 1:2])  # Range [-100, 100]
        
        action = torch.cat([delta, B], dim=-1)
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action


class Critic(nn.Module):
    """Critic network for DDPG Q-value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [400, 300]):
        super(Critic, self).__init__()
        
        # First layer processes state
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        # Second layer processes state features + action
        self.fc2 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly."""
        # First layer
        fan_in = self.fc1.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        self.fc1.weight.data.uniform_(-lim, lim)
        self.fc1.bias.data.uniform_(-lim, lim)
        
        # Second layer
        fan_in = self.fc2.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        self.fc2.weight.data.uniform_(-lim, lim)
        self.fc2.bias.data.uniform_(-lim, lim)
        
        # Output layer
        init_w = 3e-3
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-value."""
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
# Multi-Step DDPG Agent
# -------------------------
class MultiStepDDPGAgent:
    """DDPG agent optimized for multi-step option hedging."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [400, 300],
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,  # Standard discounting for multi-step
                 tau: float = 1e-3,    # DDPG standard target update rate
                 noise_theta: float = 0.15,
                 noise_sigma: float = 0.2,
                 noise_decay: float = 0.9999,
                 min_noise: float = 0.01,
                 device: str = 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(device)
        
        # Initialize target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 
                                          lr=critic_lr, 
                                          weight_decay=1e-4)
        
        # Ornstein-Uhlenbeck noise process for exploration
        self.noise = OrnsteinUhlenbeckNoise(action_dim, 
                                           theta=noise_theta,
                                           sigma=noise_sigma)
        self.noise_scale = 1.0
        self.noise_decay = noise_decay
        self.min_noise = min_noise
    
    def select_action(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
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
            
            # Clip action to valid range
            action[0] = np.clip(action[0], 0.0, 1.0)    # Delta
            action[1] = np.clip(action[1], -100.0, 100.0)  # B
        
        return action
    
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Update DDPG networks."""
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # -------------------- Update Critic -------------------- #
        # Get predicted next actions and Q values from target models
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q.unsqueeze(-1)
        
        # Compute critic loss
        current_q = self.critic(states, actions).unsqueeze(-1)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # -------------------- Update Actor -------------------- #
        # Compute actor loss (negative Q-value)
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
        
        # Decay noise
        self.decay_noise()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item(),
            'noise_scale': self.noise_scale
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
def train_multistep_ddpg(
    T: int = 3,                    # Number of time steps
    n_outcomes: int = 3,           # Number of outcomes per step  
    prices_per_step: List[float] = [0.9, 1.0, 1.1],  # Price multipliers
    probabilities: List[float] = [0.25, 0.5, 0.25],   # Probabilities
    S0: float = 100.0,
    K: float = 110.0,
    episodes: int = 25000,         # More episodes for DDPG convergence
    batch_size: int = 128,
    buffer_size: int = 100000,
    hidden_dims: List[int] = [400, 300],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    verbose: bool = True
):
    """Train DDPG agent for multi-step n-nomial option hedging."""
    
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
        print(f"Multi-Step DDPG Training for {T}-Step {n_outcomes}-nomial Option Hedging")
        print(f"Perfect Market - Dynamic Rebalancing with Ornstein-Uhlenbeck Exploration")
        print(f"{'='*80}")
        print(f"Environment: S0={S0}, K={K}, T={T} steps")
        print(f"Price multipliers per step: {prices_per_step}")
        print(f"Probabilities: {probabilities}")
        print(f"Expected final price: {price_stats['expected_final_price']:.2f}")
        print(f"Implied volatility: {price_stats['implied_volatility']:.4f}")
        print(f"{'='*80}\n")
    
    # Create agent
    agent = MultiStepDDPGAgent(
        state_dim=env.state_dim,
        action_dim=2,
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
        agent.reset_noise()  # Reset OU noise for each episode
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action with noise
            action = agent.select_action(state, add_noise=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Update agent (start after some exploration)
            if len(replay_buffer) >= batch_size and episode > 100:
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
        if verbose and episode % 3000 == 0:
            # Evaluate current policy
            eval_rewards = []
            eval_errors = []
            
            for _ in range(10):  # Multiple evaluation episodes
                state = env.reset()
                eval_reward = 0
                
                while True:
                    action = agent.select_action(state, add_noise=False)
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
            
            print(f"Episode {episode:5d}")
            print(f"  Training reward: {recent_training_reward:.4f}")
            print(f"  Eval reward: {avg_reward:.4f}")
            print(f"  Avg hedging error: {avg_error:.4f}")
            print(f"  Noise scale: {agent.noise_scale:.4f}")
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
                action = agent.select_action(state, add_noise=False)
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
    probabilities = [0.25, 0.5, 0.25]  # Probabilities for each outcome
    
    print("Training Multi-Step DDPG for N-nomial Option Hedging...")
    print(f"Configuration: {T} steps, {n_outcomes}-nomial per step")
    print(f"Price multipliers: {prices_per_step}")
    print(f"Probabilities: {probabilities}")
    
    agent, env, rewards, errors = train_multistep_ddpg(
        T=T,
        n_outcomes=n_outcomes,
        prices_per_step=prices_per_step,
        probabilities=probabilities,
        S0=100.0,
        K=110.0,
        episodes=25000,
        batch_size=128,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )