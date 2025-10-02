"""
PPO (Proximal Policy Optimization) for Multi-Step N-nomial Option Hedging
Perfect Market: No interest rates, no transaction costs
Unbiased implementation with proper temporal structure for dynamic hedging
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple, Dict, Optional
from collections import deque

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# Multi-Step N-nomial Environment
# -------------------------
class MultiStepNnomialEnv:
    """
    Multi-step N-nomial environment for option hedging.
    Perfect market: no interest rates, no transaction costs.
    Stock follows multiplicative n-nomial tree over multiple time steps.
    """
    
    def __init__(self, 
                 initial_price: float = 100.0,
                 strike: float = 110.0,
                 time_steps: int = 3,
                 price_multipliers: List[float] = [0.8, 1.1, 1.4],
                 probabilities: List[float] = [0.33, 0.34, 0.33]):
        """
        Args:
            initial_price: Starting stock price
            strike: Option strike price
            time_steps: Number of rebalancing periods until expiry
            price_multipliers: Multiplicative factors for each price movement
            probabilities: Probability of each price movement per step
        """
        self.S0 = initial_price
        self.K = strike
        self.T = time_steps
        self.multipliers = np.array(price_multipliers)
        self.probs = np.array(probabilities)
        self.n_states = len(price_multipliers)
        
        # Validate inputs
        assert len(price_multipliers) == len(probabilities), "Multipliers and probabilities must have same length"
        assert abs(sum(probabilities) - 1.0) < 1e-6, "Probabilities must sum to 1"
        assert all(p > 0 for p in probabilities), "All probabilities must be positive"
        assert all(m > 0 for m in price_multipliers), "All multipliers must be positive"
        
        # State variables
        self.current_price = self.S0
        self.time_remaining = self.T
        self.portfolio_stock = 0.0  # Delta position
        self.portfolio_cash = 0.0   # Cash position
        self.total_rebalancing_cost = 0.0
        
        # Episode tracking
        self.episode_prices = []
        self.episode_actions = []
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_price = self.S0
        self.time_remaining = self.T
        self.portfolio_stock = 0.0
        self.portfolio_cash = 0.0
        self.total_rebalancing_cost = 0.0
        self.episode_prices = [self.current_price]
        self.episode_actions = []
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        State = [normalized_price, time_remaining, current_delta, moneyness, portfolio_value]
        """
        normalized_price = self.current_price / self.S0
        normalized_time = self.time_remaining / self.T
        moneyness = self.current_price / self.K
        portfolio_value = self.portfolio_stock * self.current_price + self.portfolio_cash
        normalized_portfolio = portfolio_value / self.S0
        
        return np.array([
            normalized_price,
            normalized_time, 
            self.portfolio_stock,
            moneyness,
            normalized_portfolio
        ], dtype=np.float32)
    
    def step(self, action: Tuple[float, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step.
        Args:
            action: (target_delta, cash_adjustment) - target portfolio position
        Returns:
            next_state, reward, done, info
        """
        target_delta, cash_adj = action
        
        # Calculate rebalancing cost (small penalty for excessive trading)
        rebalancing_cost = 0.001 * abs(target_delta - self.portfolio_stock)
        self.total_rebalancing_cost += rebalancing_cost
        
        # Update portfolio positions
        self.portfolio_stock = target_delta
        self.portfolio_cash += cash_adj
        
        # Store action
        self.episode_actions.append((target_delta, cash_adj))
        
        # Advance time
        self.time_remaining -= 1
        
        # Check if episode is done
        done = (self.time_remaining <= 0)
        
        if not done:
            # Generate next price movement
            movement_idx = np.random.choice(self.n_states, p=self.probs)
            self.current_price *= self.multipliers[movement_idx]
            self.episode_prices.append(self.current_price)
            
            # Intermediate reward: small penalty for rebalancing cost
            reward = -rebalancing_cost
            
        else:
            # Terminal reward: hedging error
            final_portfolio_value = self.portfolio_stock * self.current_price + self.portfolio_cash
            option_payoff = max(0, self.current_price - self.K)
            hedging_error = final_portfolio_value - option_payoff
            
            # L2 error penalty
            reward = -(hedging_error ** 2) - self.total_rebalancing_cost
        
        # Prepare info
        info = {
            'current_price': self.current_price,
            'time_remaining': self.time_remaining,
            'portfolio_value': self.portfolio_stock * self.current_price + self.portfolio_cash,
            'rebalancing_cost': rebalancing_cost
        }
        
        if done:
            info['final_hedging_error'] = hedging_error
            info['option_payoff'] = max(0, self.current_price - self.K)
            info['episode_prices'] = self.episode_prices.copy()
            info['episode_actions'] = self.episode_actions.copy()
        
        return self._get_state(), reward, done, info

# -------------------------
# PPO Neural Networks
# -------------------------
class PolicyNetwork(nn.Module):
    """Policy network for continuous action space (delta, cash_adjustment)."""
    
    def __init__(self, state_dim: int = 5, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log_std for Gaussian policy
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small values for unbiased start."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action mean and log_std.
        """
        features = self.shared_layers(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Create Gaussian distribution
        dist = Normal(mean, std)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob

class ValueNetwork(nn.Module):
    """Value network for state value estimation."""
    
    def __init__(self, state_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value."""
        return self.network(state).squeeze(-1)

# -------------------------
# PPO Agent
# -------------------------
class PPOAgent:
    """PPO agent for multi-step option hedging."""
    
    def __init__(self,
                 state_dim: int = 5,
                 action_dim: int = 2,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 clip_ratio: float = 0.2,
                 c1: float = 0.5,  # Value function coefficient
                 c2: float = 0.01,  # Entropy coefficient
                 max_grad_norm: float = 0.5,
                 hidden_dim: int = 128):
        
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # Experience storage
        self.reset_storage()
    
    def reset_storage(self):
        """Reset experience storage for new episode batch."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Get action from policy network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = self.policy.get_action_and_log_prob(state_tensor)
            value = self.value(state_tensor)
        
        action = action.cpu().numpy().squeeze()
        log_prob = log_prob.cpu().numpy().item()
        value = value.cpu().numpy().item()
        
        return action, log_prob, value
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, log_prob: float,
                        reward: float, value: float, done: bool):
        """Store experience for batch learning."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_value_est = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_value_est = self.values[i + 1]
            
            delta = self.rewards[i] + self.gamma * next_value_est * next_non_terminal - self.values[i]
            gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        return advantages, returns
    
    def update(self, n_epochs: int = 4, batch_size: int = 64):
        """Update policy and value networks using PPO."""
        if len(self.states) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae()
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy outputs
                mean, log_std = self.policy(batch_states)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Policy loss (PPO clipping)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                current_values = self.value(batch_states)
                value_loss = F.mse_loss(current_values, batch_returns)
                
                # Total loss
                total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                
                # Update policy
                self.policy_optimizer.zero_grad()
                policy_update_loss = policy_loss - self.c2 * entropy
                policy_update_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        # Reset storage after update
        self.reset_storage()
        
        return {
            'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0,
            'value_loss': total_value_loss / n_updates if n_updates > 0 else 0,
            'entropy': total_entropy / n_updates if n_updates > 0 else 0
        }

# -------------------------
# Training Function
# -------------------------
def train_ppo_hedging(env: MultiStepNnomialEnv,
                      agent: PPOAgent,
                      n_episodes: int = 10000,
                      update_frequency: int = 20,
                      log_frequency: int = 500):
    """Train PPO agent on multi-step hedging environment."""
    
    episode_rewards = deque(maxlen=100)
    episode_hedging_errors = deque(maxlen=100)
    best_avg_reward = float('-inf')
    
    print(f"Training PPO on {env.T}-step {env.n_states}-nomial Option Hedging")
    print(f"S0={env.S0}, K={env.K}, Multipliers={env.multipliers}, Probs={env.probs}")
    print("=" * 80)
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        
        # Run episode
        while True:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, info = env.step(tuple(action))
            
            agent.store_experience(state, action, log_prob, reward, value, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                episode_rewards.append(total_reward)
                if 'final_hedging_error' in info:
                    episode_hedging_errors.append(abs(info['final_hedging_error']))
                break
        
        # Update agent periodically
        if (episode + 1) % update_frequency == 0:
            metrics = agent.update()
        
        # Logging
        if (episode + 1) % log_frequency == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_error = np.mean(episode_hedging_errors) if episode_hedging_errors else 0
            
            print(f"Episode {episode + 1:6d} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Avg |Error|: {avg_error:8.2f} | "
                  f"Final Price: {info.get('current_price', 0):6.1f}")
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                print(f"  → New best average reward: {best_avg_reward:.2f}")
    
    return agent

# -------------------------
# Evaluation Function
# -------------------------
def evaluate_agent(env: MultiStepNnomialEnv, agent: PPOAgent, n_eval_episodes: int = 1000):
    """Evaluate trained agent performance."""
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    eval_rewards = []
    eval_errors = []
    final_prices = []
    
    for _ in range(n_eval_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # Use deterministic policy (mean action)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                mean, _ = agent.policy(state_tensor)
                action = mean.cpu().numpy().squeeze()
            
            state, reward, done, info = env.step(tuple(action))
            total_reward += reward
            
            if done:
                eval_rewards.append(total_reward)
                if 'final_hedging_error' in info:
                    eval_errors.append(info['final_hedging_error'])
                    final_prices.append(info['current_price'])
                break
    
    # Statistics
    print(f"Evaluation over {n_eval_episodes} episodes:")
    print(f"  Mean Reward: {np.mean(eval_rewards):.4f} ± {np.std(eval_rewards):.4f}")
    print(f"  Mean |Hedging Error|: {np.mean(np.abs(eval_errors)):.4f}")
    print(f"  Std Hedging Error: {np.std(eval_errors):.4f}")
    print(f"  Max |Error|: {np.max(np.abs(eval_errors)):.4f}")
    print(f"  L2 Error (MSE): {np.mean(np.array(eval_errors)**2):.4f}")
    
    # Sample episode demonstration
    print("\nSample Episode Demonstration:")
    state = env.reset()
    episode_data = []
    
    step = 0
    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, _ = agent.policy(state_tensor)
            action = mean.cpu().numpy().squeeze()
        
        delta, cash_adj = action
        print(f"  Step {step}: Price=${env.current_price:.1f}, Delta={delta:.3f}, Cash_Adj={cash_adj:.3f}")
        
        state, reward, done, info = env.step(tuple(action))
        step += 1
        
        if done:
            portfolio_value = info['portfolio_value']
            option_payoff = info['option_payoff']
            hedging_error = info['final_hedging_error']
            
            print(f"  Final: Price=${info['current_price']:.1f}")
            print(f"         Portfolio Value=${portfolio_value:.3f}")
            print(f"         Option Payoff=${option_payoff:.3f}")
            print(f"         Hedging Error=${hedging_error:.3f}")
            break
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'mean_abs_error': np.mean(np.abs(eval_errors)),
        'mse': np.mean(np.array(eval_errors)**2),
        'std_error': np.std(eval_errors)
    }

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Configuration - Modify these for different n-nomial models
    config = {
        'initial_price': 100.0,
        'strike': 110.0,
        'time_steps': 3,  # Multi-step: 3 rebalancing periods
        
        # Trinomial example (modify as needed)
        'price_multipliers': [0.8, 1.1, 1.4],  # down, middle, up movements
        'probabilities': [0.33, 0.34, 0.33],   # corresponding probabilities
        
        # Alternative examples (uncomment to use):
        # Binomial: 'price_multipliers': [0.8, 1.4], 'probabilities': [0.5, 0.5]
        # 5-nomial: 'price_multipliers': [0.7, 0.85, 1.0, 1.15, 1.3], 'probabilities': [0.2, 0.2, 0.2, 0.2, 0.2]
    }
    
    # Create environment and agent
    env = MultiStepNnomialEnv(**config)
    agent = PPOAgent(
        state_dim=5,
        action_dim=2,
        lr=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_ratio=0.2,
        hidden_dim=128
    )
    
    # Train agent
    trained_agent = train_ppo_hedging(
        env=env,
        agent=agent,
        n_episodes=15000,
        update_frequency=20,
        log_frequency=1000
    )
    
    # Evaluate performance
    evaluation_results = evaluate_agent(env, trained_agent, n_eval_episodes=1000)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)