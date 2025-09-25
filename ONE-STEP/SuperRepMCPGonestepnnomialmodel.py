"""
Optimized MCPG (Monte Carlo Policy Gradient) for One-Step N-nomial Option Hedging
Perfect Market: No interest rates, no transaction costs
Completely unbiased implementation for fair comparison
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

# -------------------------
# Environment: Perfect Market N-nomial Model
# -------------------------
class OneStepNnomialEnv:
    """One-step N-nomial environment for option hedging in perfect market."""
    
    def __init__(self, 
                 S0: float,
                 K: float, 
                 prices: List[float],
                 probabilities: List[float],
                 seed: int = 0):
        
        self.S0 = float(S0)
        self.K = float(K)
        self.prices = [float(p) for p in prices]
        self.probabilities = np.array(probabilities, dtype=np.float32)
        
        # Validation
        assert len(self.prices) == len(self.probabilities), "Prices and probabilities must have same length"
        assert abs(self.probabilities.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"
        
        self.n_outcomes = len(self.prices)
        self.np_rng = np.random.RandomState(seed)
        
        # Enhanced state representation
        self.state_dim = 4 + self.n_outcomes * 2  # S0, K, S0/K, time + prices + probabilities
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = self.S0 / 100.0  # Normalized stock price
        state[1] = self.K / 100.0   # Normalized strike
        state[2] = self.S0 / self.K  # Moneyness ratio
        state[3] = 1.0              # Time to maturity
        state[4:4+self.n_outcomes] = np.array(self.prices) / 100.0  # Normalized prices
        state[4+self.n_outcomes:] = self.probabilities
        self.state = state
        return self.state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment."""
        delta = float(action[0])
        B = float(action[1])
        
        # Sample final price according to probabilities
        outcome_idx = self.np_rng.choice(self.n_outcomes, p=self.probabilities)
        S_T = self.prices[outcome_idx]
        
        # Calculate option payoff and portfolio value (perfect market)
        payoff = max(S_T - self.K, 0.0)
        portfolio = delta * S_T + B  # No interest on cash
        
        # Hedging error
        hedging_error = portfolio - payoff
        
        # Scaled L2 reward for better gradient flow
        reward = -(hedging_error ** 2) / 1000.0
        
        # Terminal state
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        next_state[0] = S_T / 100.0
        next_state[1] = self.K / 100.0
        next_state[2] = S_T / self.K
        next_state[3] = 0.0  # Time to maturity = 0
        next_state[4:4+self.n_outcomes] = np.array(self.prices) / 100.0
        next_state[4+self.n_outcomes:] = self.probabilities
        
        info = {
            'hedging_error': hedging_error,
            'portfolio_value': portfolio,
            'option_payoff': payoff,
            'final_price': S_T,
            'delta': delta,
            'B': B,
            'squared_error': hedging_error ** 2
        }
        
        return next_state, reward, True, info
    
    def theoretical_hedge(self) -> Tuple[float, float, float]:
        """Calculate theoretical hedge using least squares."""
        A = np.column_stack([self.prices, np.ones(self.n_outcomes)])
        b = np.array([max(S - self.K, 0) for S in self.prices])
        W = np.diag(self.probabilities)
        
        hedge_params = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
        delta_theory = hedge_params[0]
        B_theory = hedge_params[1]
        portfolio_value = delta_theory * self.S0 + B_theory
        
        return delta_theory, B_theory, portfolio_value


# -------------------------
# Optimized MCPG Agent
# -------------------------
class OptimizedMCPGAgent:
    """Monte Carlo Policy Gradient agent optimized for option hedging."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 policy_lr: float = 3e-4,
                 value_lr: float = 1e-3,
                 gamma: float = 0.95,  # Slight discounting for single-step
                 batch_size: int = 32,  # Batch episodes together
                 device: str = 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Policy network with proper architecture (no BatchNorm for single samples)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], action_dim * 2)  # Mean and log_std
        ).to(device)
        
        # Value network (also no BatchNorm for consistency)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        ).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr, weight_decay=1e-5)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=value_lr, weight_decay=1e-4)
        
        # Learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.9999)
        self.value_scheduler = optim.lr_scheduler.ExponentialLR(self.value_optimizer, gamma=0.9999)
        
        # Initialize networks with no bias
        self._initialize_weights()
        
        # Storage for batch training
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_count = 0
    
    def _initialize_weights(self):
        """Initialize networks with no bias toward solution."""
        for network in [self.policy_network, self.value_network]:
            for module in network.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # Final layer small weights for stability - NO BIAS toward solution
        final_policy_layer = self.policy_network[-1]
        nn.init.uniform_(final_policy_layer.weight, -1e-3, 1e-3)
        nn.init.zeros_(final_policy_layer.bias)
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """Select action using policy network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Handle batch norm in eval mode
        if evaluate:
            self.policy_network.eval()
        
        # Get policy parameters
        output = self.policy_network(state_tensor)
        mean = output[:, :2]
        log_std = output[:, 2:].clamp(-5, 0)  # Reasonable exploration range
        std = torch.exp(log_std)
        
        if evaluate:
            # Use mean action for evaluation
            raw_action = mean
            log_prob = None
            self.policy_network.train()
        else:
            # Sample action from distribution
            normal = torch.distributions.Normal(mean, std)
            raw_action = normal.rsample()
            log_prob = normal.log_prob(raw_action).sum(-1, keepdim=True)
        
        # Apply constraints - completely unbiased
        # Delta: sigmoid for [0, 1] range (natural for call options)
        delta = torch.sigmoid(raw_action[:, 0:1])
        # B: wide tanh range to cover theoretical value
        B = 60.0 * torch.tanh(raw_action[:, 1:2]) - 10.0  # Range [-70, 50]
        
        final_action = torch.cat([delta, B], dim=-1)
        action_np = final_action.squeeze(0).cpu().detach().numpy()
        
        return action_np, log_prob
    
    def store_episode_data(self, state: np.ndarray, action: np.ndarray, reward: float, log_prob: torch.Tensor):
        """Store data for current episode."""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        if log_prob is not None:
            self.episode_log_probs.append(log_prob)
    
    def update(self) -> Dict[str, float]:
        """Update policy using batched episodes for better stability."""
        self.episode_count += 1
        
        # Only update every batch_size episodes for better gradients
        if self.episode_count % self.batch_size != 0 or len(self.episode_rewards) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'episodes_in_batch': 0}
        
        # Convert episode data to tensors
        all_states = []
        all_log_probs = []
        all_returns = []
        
        episodes_in_batch = len(self.episode_rewards) // self.batch_size
        
        for ep_idx in range(episodes_in_batch):
            # Get data for this episode
            start_idx = ep_idx * 1  # Each episode has 1 step
            end_idx = start_idx + 1
            
            states = self.episode_states[start_idx:end_idx]
            log_probs = self.episode_log_probs[start_idx:end_idx]
            rewards = self.episode_rewards[start_idx:end_idx]
            
            # Calculate Monte Carlo return (single step, so just the reward)
            G = rewards[0]  # No discounting needed for single step
            
            all_states.extend(states)
            all_log_probs.extend(log_probs)
            all_returns.append(G)
        
        if len(all_states) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'episodes_in_batch': 0}
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(all_states)).to(self.device)
        log_probs_tensor = torch.cat(all_log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(all_returns).to(self.device)
        
        # Normalize returns for stability
        if returns_tensor.std() > 1e-8:
            returns_normalized = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        else:
            returns_normalized = returns_tensor
        
        # Calculate baseline (value function)
        values = self.value_network(states_tensor).squeeze()
        if len(values.shape) == 0:  # Handle single value case
            values = values.unsqueeze(0)
        
        # Advantages
        advantages = returns_normalized - values.detach()
        
        # Policy loss (negative because we want gradient ascent)
        policy_loss = -(log_probs_tensor.squeeze() * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns_normalized)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
        self.value_optimizer.step()
        self.value_scheduler.step()
        
        # Clear storage
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': policy_loss.item() + value_loss.item(),
            'avg_return': returns_tensor.mean().item(),
            'avg_advantage': advantages.mean().item(),
            'episodes_in_batch': episodes_in_batch,
            'policy_lr': self.policy_optimizer.param_groups[0]['lr']
        }


# -------------------------
# Training Function
# -------------------------
def train_optimized_mcpg(
    prices: List[float],
    probabilities: List[float],
    S0: float = 100.0,
    K: float = 110.0,
    episodes: int = 20000,  # More episodes needed for MCPG
    batch_size: int = 32,
    hidden_dims: List[int] = [256, 256],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    verbose: bool = True
):
    """Train optimized MCPG agent for n-nomial option hedging."""
    
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
        print(f"Optimized MCPG Training for {env.n_outcomes}-nomial Option Hedging")
        print(f"Perfect Market - Completely Unbiased Implementation")
        print(f"{'='*70}")
        print(f"Environment: S0={env.S0}, K={env.K}")
        print(f"Possible prices: {[f'{p:.2f}' for p in env.prices]}")
        print(f"Probabilities: {[f'{p:.3f}' for p in env.probabilities]}")
        print(f"\nTheoretical Hedge:")
        print(f"  Delta: {delta_theory:.4f}")
        print(f"  B: {B_theory:.4f}")
        print(f"  Portfolio Value: {fair_price_theory:.4f}")
        print(f"{'='*70}\n")
    
    # Create agent
    agent = OptimizedMCPGAgent(
        state_dim=env.state_dim,
        action_dim=2,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        device=device
    )
    
    # Training loop
    for episode in range(1, episodes + 1):
        state = env.reset()
        
        # Select action (with log probability for training)
        action, log_prob = agent.select_action(state, evaluate=False)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store episode data
        agent.store_episode_data(state, action, float(reward), log_prob)
        
        # Update agent (batched)
        update_info = agent.update()
        
        # Logging and evaluation
        if verbose and episode % 2500 == 0:
            # Evaluate current policy
            state = env.reset()
            eval_action, _ = agent.select_action(state, evaluate=True)
            
            delta_current = eval_action[0]
            B_current = eval_action[1]
            portfolio_value_current = delta_current * env.S0 + B_current
            
            # Calculate performance metrics
            test_errors = []
            for _ in range(100):
                state = env.reset()
                action, _ = agent.select_action(state, evaluate=True)
                _, _, _, info = env.step(action)
                test_errors.append(info['hedging_error'])
            
            mse = np.mean([e**2 for e in test_errors])
            max_error = max(abs(e) for e in test_errors)
            
            print(f"Episode {episode:5d}")
            print(f"  Current: Δ={delta_current:.4f}, B={B_current:.3f}, Value={portfolio_value_current:.3f}")
            print(f"  Theory:  Δ={delta_theory:.4f}, B={B_theory:.3f}, Value={fair_price_theory:.3f}")
            print(f"  MSE: {mse:.6f}, Max |Error|: {max_error:.4f}")
            if update_info['episodes_in_batch'] > 0:
                print(f"  Policy Loss: {update_info['policy_loss']:.4f}, Value Loss: {update_info['value_loss']:.4f}")
                print(f"  Policy LR: {update_info['policy_lr']:.6f}")
            print()
    
    # Final evaluation
    if verbose:
        print(f"\n{'='*70}")
        print("Final Evaluation")
        print(f"{'='*70}")
        
        state = env.reset()
        final_action, _ = agent.select_action(state, evaluate=True)
        delta_final = final_action[0]
        B_final = final_action[1]
        portfolio_value_final = delta_final * env.S0 + B_final
        
        # Comprehensive testing
        test_errors = []
        for _ in range(1000):
            state = env.reset()
            action, _ = agent.select_action(state, evaluate=True)
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
        
        # Show error breakdown by scenario
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
    
    print("Training Optimized MCPG for N-nomial Option Hedging...")
    print("Completely unbiased implementation for fair comparison")
    print(f"Model: {len(prices)}-nomial")
    print(f"Prices: {prices}")
    print(f"Probabilities: {probabilities}")
    
    agent, env = train_optimized_mcpg(
        prices=prices,
        probabilities=probabilities,
        S0=100.0,
        K=110.0,
        episodes=20000,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )