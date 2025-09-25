"""
Optimized TD3 (Twin Delayed Deep Deterministic Policy Gradient) for One-Step N-nomial Option Hedging
Perfect Market: No interest rates, no transaction costs
Optimized for better convergence and multi-step scalability
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
# Environment: One-Step N-nomial Model (Perfect Market)
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
        
        # Enhanced state representation for better learning
        self.state_dim = 4 + self.n_outcomes * 2  # S0, K, S0/K, time + prices + probabilities
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = self.S0 / 100.0  # Normalize stock price
        state[1] = self.K / 100.0   # Normalize strike
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
# Improved TD3 Networks
# -------------------------
class Actor(nn.Module):
    """Optimized Actor network with better architecture for hedging."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(Actor, self).__init__()
        
        # Build network with batch normalization for stability
        layers = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i == 0:  # Only first layer gets batch norm to avoid issues
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Light dropout for regularization
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
        
        # Better initialization for financial applications
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization + strategic bias."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize final layer with smaller weights for stability
        final_layer = self.network[-1]
        nn.init.uniform_(final_layer.weight, -1e-3, 1e-3)
        
        # Strategic bias initialization for faster convergence
        # For delta: sigmoid^(-1)(0.5) ≈ 0, so bias of 0 gives starting delta ≈ 0.5
        final_layer.bias[0].data.fill_(0.0)  # Delta starts around 0.5
        # For B: we want B ≈ -40, so solve: -10 + 60*tanh(x) = -40 → tanh(x) = -0.5 → x ≈ -0.55
        final_layer.bias[1].data.fill_(-0.55)  # B starts around -40
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with appropriate action constraints."""
        # Handle batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        raw_output = self.network(state)
        
        # Apply constraints for hedging actions
        # Delta: sigmoid to [0, 1] for call options (more natural than tanh)
        delta = torch.sigmoid(raw_output[:, 0:1])
        
        # B: wider range since theoretical B = -40
        B = 60.0 * torch.tanh(raw_output[:, 1:2]) - 10.0  # Range [-70, 50], centered at -10
        
        action = torch.cat([delta, B], dim=-1)
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action


class Critic(nn.Module):
    """Improved Twin Q-network with better architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(Critic, self).__init__()
        
        input_dim = state_dim + action_dim
        
        # Q1 network with batch normalization
        q1_layers = [nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            q1_layers.extend([
                nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        q1_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q1_network = nn.Sequential(*q1_layers)
        
        # Q2 network with batch normalization
        q2_layers = [nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            q2_layers.extend([
                nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        q2_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q2_network = nn.Sequential(*q2_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both Q-networks."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_network(x).squeeze(-1)
        q2 = self.q2_network(x).squeeze(-1)
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q1 value only."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        x = torch.cat([state, action], dim=-1)
        return self.q1_network(x).squeeze(-1)


# -------------------------
# Optimized TD3 Agent
# -------------------------
class OptimizedTD3Agent:
    """Optimized TD3 agent for precise financial optimization."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 actor_lr: float = 1e-4,  # Lower learning rate for stability
                 critic_lr: float = 3e-4,
                 gamma: float = 0.95,     # Slightly less than 1 for single-step
                 tau: float = 0.01,       # Faster target updates
                 policy_noise: float = 0.1,  # Reduced noise
                 noise_clip: float = 0.2,
                 policy_freq: int = 2,
                 device: str = 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_iterations = 0
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(device)
        
        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers with weight decay for regularization
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-4)
        
        # Learning rate schedulers - slower decay to maintain learning ability
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.99995)
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.99995)
        
        # Adaptive exploration - slower decay, higher minimum
        self.exploration_noise = policy_noise
        self.noise_decay = 0.9995  # Much slower decay
        self.min_noise = 0.05      # Higher minimum noise
    
    def select_action(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Select action from policy with optional exploration."""
        state = torch.FloatTensor(state).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        
        action = action.cpu().numpy()
        
        if add_noise:
            # Add exploration noise with decay
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = action + noise
            
            # Clip actions to valid ranges
            action[0] = np.clip(action[0], 0.0, 1.0)    # Delta for calls
            action[1] = np.clip(action[1], -70.0, 50.0)  # B in wider range
        
        return action
    
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Update TD3 networks with improvements."""
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Critic update
        with torch.no_grad():
            # Target policy smoothing
            next_actions = self.actor_target(next_states)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise)
            
            # Clip actions to valid ranges
            next_actions[:, 0] = torch.clamp(next_actions[:, 0], 0.0, 1.0)    # Delta
            next_actions[:, 1] = torch.clamp(next_actions[:, 1], -70.0, 50.0)  # B
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2).unsqueeze(-1)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        current_q1 = current_q1.unsqueeze(-1)
        current_q2 = current_q2.unsqueeze(-1)
        
        # Compute critic loss with L2 regularization
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Delayed policy update
        actor_loss = torch.tensor(0.0)
        if self.total_iterations % self.policy_freq == 0:
            # Actor loss
            actor_actions = self.actor(states)
            actor_q1 = self.critic.q1(states, actor_actions)
            actor_loss = -actor_q1.mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update target networks
            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.actor, self.actor_target)
            
            # Update learning rates
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        
        # Decay exploration noise
        self.exploration_noise = max(self.min_noise, self.exploration_noise * self.noise_decay)
        
        self.total_iterations += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss != 0 else 0.0,
            'q_value': current_q1.mean().item(),
            'exploration_noise': self.exploration_noise,
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }
    
    def soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + 
                                  (1 - self.tau) * target_param.data)


# -------------------------
# Training Function
# -------------------------
def train_optimized_td3(
    prices: List[float],
    probabilities: List[float],
    S0: float = 100.0,
    K: float = 110.0,
    episodes: int = 15000,
    batch_size: int = 128,  # Smaller batch size for better gradients
    buffer_size: int = 50000,
    hidden_dims: List[int] = [256, 256],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    verbose: bool = True
):
    """Train optimized TD3 agent for n-nomial option hedging."""
    
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
        print(f"Optimized TD3 Training for {env.n_outcomes}-nomial Option Hedging")
        print(f"Perfect Market - Optimized for Multi-step Scaling")
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
    agent = OptimizedTD3Agent(
        state_dim=env.state_dim,
        action_dim=2,
        hidden_dims=hidden_dims,
        device=device
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Training loop
    total_steps = 0
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        
        # Action selection with adaptive exploration
        action = agent.select_action(state, add_noise=True)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        replay_buffer.push(state, action, float(reward), next_state, float(done))
        total_steps += 1
        
        # Update agent
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            update_info = agent.update(batch)
        
        # Logging and evaluation
        if verbose and episode % 2500 == 0:
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
            if len(replay_buffer) >= batch_size:
                print(f"  Exploration: {agent.exploration_noise:.4f}, Actor LR: {agent.actor_optimizer.param_groups[0]['lr']:.6f}")
            print()
    
    # Final evaluation
    if verbose:
        print(f"\n{'='*70}")
        print("Final Evaluation")
        print(f"{'='*70}")
        
        state = env.reset()
        final_action = agent.select_action(state, add_noise=False)
        delta_final = final_action[0]
        B_final = final_action[1]
        portfolio_value_final = delta_final * env.S0 + B_final
        
        # Comprehensive testing
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
    
    print("Training Optimized TD3 for N-nomial Option Hedging...")
    print("Optimized for convergence and multi-step scalability")
    print(f"Model: {len(prices)}-nomial")
    print(f"Prices: {prices}")
    print(f"Probabilities: {probabilities}")
    
    agent, env = train_optimized_td3(
        prices=prices,
        probabilities=probabilities,
        S0=100.0,
        K=110.0,
        episodes=15000,
        batch_size=128,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )