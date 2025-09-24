"""
MCPG (Monte Carlo Policy Gradient) for One-Step N-nomial Option Hedging
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
# Environment: One-Step N-nomial Model
# -------------------------
class OneStepNnomialEnv:
    """
    One-step N-nomial environment for option hedging.
    """
    def __init__(self, 
                 S0: float = 100.0,
                 K: float = 110.0, 
                 prices: Optional[List[float]] = None,
                 probabilities: Optional[List[float]] = None,
                 n_outcomes: int = 5,
                 volatility: float = 0.3,
                 risk_free_rate: float = 0.0,
                 seed: int = 0):
        
        self.S0 = float(S0)
        self.K = float(K)
        self.r = risk_free_rate
        
        # Generate or set prices and probabilities
        if prices is None:
            # Generate n equally spaced prices based on volatility
            min_return = -3 * volatility
            max_return = 3 * volatility
            returns = np.linspace(min_return, max_return, n_outcomes)
            self.prices = [S0 * np.exp(r) for r in returns]
        else:
            self.prices = [float(p) for p in prices]
            n_outcomes = len(self.prices)
        
        if probabilities is None:
            # Use discretized normal distribution probabilities
            returns = np.array([np.log(p/S0) for p in self.prices])
            weights = np.exp(-0.5 * ((returns - self.r) / volatility) ** 2)
            self.probabilities = weights / weights.sum()
        else:
            self.probabilities = np.array(probabilities, dtype=np.float32)
            assert abs(self.probabilities.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"
        
        self.n_outcomes = len(self.prices)
        assert len(self.prices) == len(self.probabilities), "Prices and probabilities must have same length"
        
        self.np_rng = np.random.RandomState(seed)
        
        # State: [S0, time_to_maturity, strike_price_ratio]
        self.state_dim = 3
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        state = np.array([self.S0, 1.0, self.K/self.S0], dtype=np.float32)
        self.state = state
        return self.state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: [delta, B] where delta is hedge ratio and B is bond position
        
        Returns:
            next_state, reward, done, info
        """
        delta = float(action[0])
        B = float(action[1])
        
        # Sample final price according to probabilities
        outcome_idx = self.np_rng.choice(self.n_outcomes, p=self.probabilities)
        S_T = self.prices[outcome_idx]
        
        # Calculate option payoff
        payoff = max(S_T - self.K, 0.0)
        
        # Calculate portfolio value
        portfolio = delta * S_T + B * (1 + self.r)
        
        # Hedging error
        err = portfolio - payoff
        
        # Reward is negative squared error
        reward = -(err ** 2)
        
        # Terminal state
        next_state = np.array([S_T, 0.0, self.K/self.S0], dtype=np.float32)
        done = True
        
        info = {
            'err': err,
            'portfolio': portfolio,
            'payoff': payoff,
            'final_price': S_T,
            'outcome_idx': outcome_idx,
            'delta': delta,
            'B': B,
            'squared_error': err ** 2
        }
        
        return next_state, reward, done, info
    
    def theoretical_hedge(self) -> Tuple[float, float, float]:
        """
        Calculate theoretical hedge ratio and fair price using least squares.
        """
        # Payoffs at maturity
        payoffs = np.array([max(S - self.K, 0) for S in self.prices])
        
        # Design matrix: [prices, ones]
        A = np.column_stack([self.prices, np.ones(self.n_outcomes)])
        
        # Weighted least squares using probabilities as weights
        W = np.diag(self.probabilities)
        try:
            ATW = A.T @ W
            ATWA = ATW @ A
            ATWb = ATW @ payoffs
            
            hedge_params = np.linalg.solve(ATWA, ATWb)
            delta_theory = hedge_params[0]
            B_theory = hedge_params[1]
            fair_price = delta_theory * self.S0 + B_theory
            
            return delta_theory, B_theory, fair_price
        except np.linalg.LinAlgError:
            # Fallback to expected value
            expected_payoff = np.sum(self.probabilities * payoffs)
            in_the_money = np.array([S > self.K for S in self.prices])
            if np.any(in_the_money):
                delta_theory = np.sum(self.probabilities[in_the_money])
            else:
                delta_theory = 0.0
            B_theory = expected_payoff - delta_theory * self.S0
            fair_price = expected_payoff / (1 + self.r)
            
            return delta_theory, B_theory, fair_price

# -------------------------
# MCPG Agent
# -------------------------
class MCPGAgent:
    """Monte Carlo Policy Gradient agent for option hedging."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [128, 128],
                 policy_lr: float = 1e-3,
                 value_lr: float = 1e-3,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        
        self.device = device
        self.gamma = gamma
        
        # Policy network (outputs mean and log_std for each action)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim * 2)  # Mean and log_std for each action
        ).to(device)
        
        # Value network for baseline
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        ).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=value_lr)
        
        # Storage for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action using policy network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get policy parameters
        output = self.policy_network(state_tensor)
        mean = output[:, :2]
        log_std = output[:, 2:].clamp(-20, 2)
        std = torch.exp(log_std)
        
        if evaluate:
            # Use mean action for evaluation
            action = mean
        else:
            # Sample action from distribution
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()
            
            # Store log probability for training
            log_prob = normal.log_prob(action).sum(-1, keepdim=True)
            self.log_probs.append(log_prob)
            self.states.append(state_tensor)
            self.actions.append(action)
        
        # Apply transformations
        delta = torch.sigmoid(action[:, 0:1])
        B = torch.tanh(action[:, 1:2]) * 200
        
        final_action = torch.cat([delta, B], dim=-1)
        
        return final_action.squeeze(0).cpu().detach().numpy()
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def update(self) -> Dict[str, float]:
        """Update policy using Monte Carlo returns."""
        if len(self.rewards) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}
        
        # Convert stored data to tensors
        states = torch.cat(self.states)
        log_probs = torch.cat(self.log_probs)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        
        # Calculate Monte Carlo returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate baseline (value function)
        values = self.value_network(states).squeeze()
        
        # Advantages
        advantages = returns - values.detach()
        
        # Policy loss (negative for gradient ascent)
        policy_loss = -(log_probs.squeeze() * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.value_optimizer.step()
        
        # Clear storage
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': policy_loss.item() + value_loss.item(),
            'avg_return': returns.mean().item(),
            'avg_advantage': advantages.mean().item()
        }

# -------------------------
# MCPG Training Function
# -------------------------
def train_mcpg_nnomial(
    n_outcomes: int = 5,
    episodes: int = 5000,
    hidden_dims: List[int] = [128, 128],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    verbose: bool = True
) -> Tuple[MCPGAgent, OneStepNnomialEnv, List[Dict]]:
    """
    Train MCPG agent for n-nomial option hedging.
    
    Returns:
        agent: Trained MCPG agent
        env: Environment used for training
        training_history: List of training statistics
    """
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = OneStepNnomialEnv(
        S0=100.0,
        K=110.0,
        n_outcomes=n_outcomes,
        volatility=0.3,
        seed=seed
    )
    
    # Get theoretical solution
    delta_theory, B_theory, fair_price_theory = env.theoretical_hedge()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"MCPG Training for {n_outcomes}-nomial Option Hedging")
        print(f"{'='*60}")
        print(f"Environment: S0={env.S0}, K={env.K}")
        print(f"Possible prices: {[f'{p:.2f}' for p in env.prices]}")
        print(f"Probabilities: {[f'{p:.3f}' for p in env.probabilities]}")
        print(f"\nTheoretical solution:")
        print(f"  Delta: {delta_theory:.4f}")
        print(f"  B: {B_theory:.4f}")
        print(f"  Fair Price: {fair_price_theory:.4f}")
        print(f"{'='*60}\n")
    
    # Create agent
    agent = MCPGAgent(
        state_dim=env.state_dim,
        action_dim=2,
        hidden_dims=hidden_dims,
        device=device
    )
    
    # Training loop
    training_history = []
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        
        # Select action (stores data for training)
        action = agent.select_action(state, evaluate=False)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store reward
        agent.store_reward(float(reward))
        
        # Update agent after each episode (Monte Carlo)
        update_info = agent.update()
        
        # Store training info
        training_info = {
            'episode': episode,
            'hedging_error': info['err'],
            'delta': info['delta'],
            'B': info['B'],
            **update_info
        }
        training_history.append(training_info)
        
        # Logging
        if verbose and episode % 1000 == 0:
            # Evaluate current policy
            state = env.reset()
            eval_action = agent.select_action(state, evaluate=True)
            
            delta_current = eval_action[0]
            B_current = eval_action[1]
            fair_price_current = delta_current * env.S0 + B_current
            
            # Calculate average hedging error
            errors = []
            for _ in range(100):
                state = env.reset()
                action = agent.select_action(state, evaluate=True)
                _, _, _, info = env.step(action)
                errors.append(info['err'])
            
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            
            print(f"Episode {episode:6d}")
            print(f"  Current: Δ={delta_current:.4f}, B={B_current:.3f}, Price={fair_price_current:.3f}")
            print(f"  Theory:  Δ={delta_theory:.4f}, B={B_theory:.3f}, Price={fair_price_theory:.3f}")
            print(f"  Avg Error: {avg_error:.4f} ± {std_error:.4f}")
            print(f"  Policy Loss: {update_info['policy_loss']:.4f}, Value Loss: {update_info['value_loss']:.4f}")
    
    # Final evaluation
    if verbose:
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}")
        
        state = env.reset()
        final_action = agent.select_action(state, evaluate=True)
        
        delta_final = final_action[0]
        B_final = final_action[1]
        fair_price_final = delta_final * env.S0 + B_final
        
        # Test over many episodes
        test_errors = []
        test_rewards = []
        for _ in range(1000):
            state = env.reset()
            action = agent.select_action(state, evaluate=True)
            _, reward, _, info = env.step(action)
            test_errors.append(info['err'])
            test_rewards.append(reward)
        
        print(f"Learned Policy:")
        print(f"  Delta: {delta_final:.4f} (Theory: {delta_theory:.4f})")
        print(f"  B: {B_final:.4f} (Theory: {B_theory:.4f})")
        print(f"  Fair Price: {fair_price_final:.4f} (Theory: {fair_price_theory:.4f})")
        print(f"\nPerformance (1000 episodes):")
        print(f"  Mean Hedging Error: {np.mean(test_errors):.6f}")
        print(f"  Std Hedging Error: {np.std(test_errors):.6f}")
        print(f"  Mean Reward: {np.mean(test_rewards):.4f}")
        print(f"  95% VaR of |Error|: {np.percentile(np.abs(test_errors), 95):.4f}")
        print(f"{'='*60}")
    
    return agent, env, training_history

# -------------------------
# Main Execution for MCPG
# -------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Example 1: 5-nomial model
    print("\nTraining MCPG for 5-nomial model...")
    agent_5, env_5, history_5 = train_mcpg_nnomial(
        n_outcomes=5,
        episodes=3000,
        device=device
    )
    
    # Example 2: 10-nomial model
    print("\n\nTraining MCPG for 10-nomial model...")
    agent_10, env_10, history_10 = train_mcpg_nnomial(
        n_outcomes=10,
        episodes=5000,
        device=device
    )
    
    # Example 3: Custom trinomial model
    print("\n\nTraining MCPG for custom trinomial model...")
    env_tri = OneStepNnomialEnv(
        S0=100.0,
        K=110.0,
        prices=[80.0, 100.0, 140.0],
        probabilities=[0.25, 0.5, 0.25],
        seed=42
    )
    
    agent_tri = MCPGAgent(
        state_dim=env_tri.state_dim,
        action_dim=2,
        device=device
    )
    
    # Quick training for trinomial
    for episode in range(2000):
        state = env_tri.reset()
        action = agent_tri.select_action(state, evaluate=False)
        next_state, reward, done, info = env_tri.step(action)
        agent_tri.store_reward(float(reward))
        agent_tri.update()
    
    # Evaluate trinomial
    state = env_tri.reset()
    action = agent_tri.select_action(state, evaluate=True)
    delta_theory, B_theory, fair_price_theory = env_tri.theoretical_hedge()
    
    print(f"\nTrinomial Model Results:")
    print(f"  Learned: Δ={action[0]:.4f}, B={action[1]:.4f}, Price={action[0] * env_tri.S0 + action[1]:.4f}")
    print(f"  Theory:  Δ={delta_theory:.4f}, B={B_theory:.4f}, Price={fair_price_theory:.4f}")