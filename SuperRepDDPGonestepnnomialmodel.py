"""
DDPG (Deep Deterministic Policy Gradient) for One-Step N-nomial Option Hedging
Hedging a European call option with n possible price outcomes
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
# Environment: One-Step N-nomial Model (Same as SAC)
# -------------------------
class OneStepNnomialEnv:
    """
    One-step N-nomial environment for option hedging.
    The stock can move to n different prices with associated probabilities.
    """
    def __init__(self, 
                 S0: float = 100.0,
                 K: float = 110.0, 
                 prices: Optional[List[float]] = None,
                 probabilities: Optional[List[float]] = None,
                 n_outcomes: int = 5,
                 volatility: float = 0.3,
                 seed: int = 0):
        """
        Args:
            S0: Initial stock price
            K: Strike price
            prices: List of possible prices at T=1. If None, generates from volatility
            probabilities: List of probabilities for each price. If None, uses equal probabilities
            n_outcomes: Number of possible outcomes (used if prices is None)
            volatility: Used to generate price range if prices not provided
            seed: Random seed
        """
        self.S0 = float(S0)
        self.K = float(K)
        
        # Generate or set prices and probabilities
        if prices is None:
            # Generate n equally spaced prices based on volatility
            min_return = -2 * volatility
            max_return = 2 * volatility
            returns = np.linspace(min_return, max_return, n_outcomes)
            self.prices = [S0 * np.exp(r) for r in returns]
        else:
            self.prices = [float(p) for p in prices]
            n_outcomes = len(self.prices)
        
        if probabilities is None:
            # Use discretized normal distribution probabilities
            returns = np.array([np.log(p/S0) for p in self.prices])
            weights = np.exp(-0.5 * (returns / volatility) ** 2)
            self.probabilities = weights / weights.sum()
        else:
            self.probabilities = np.array(probabilities, dtype=np.float32)
            assert abs(self.probabilities.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"
        
        self.n_outcomes = len(self.prices)
        assert len(self.prices) == len(self.probabilities), "Prices and probabilities must have same length"
        
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # State dimension includes more information for n-nomial case
        self.state_dim = 2 + self.n_outcomes * 2  # S0, time_to_maturity, prices, probabilities
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = self.S0
        state[1] = 1.0  # Time to maturity
        state[2:2+self.n_outcomes] = self.prices
        state[2+self.n_outcomes:] = self.probabilities
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
        portfolio = delta * S_T + B
        
        # Hedging error
        err = portfolio - payoff
        
        # Reward is negative squared error
        reward = -(err ** 2)
        
        # Terminal state
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        next_state[0] = S_T
        next_state[1] = 0.0  # Time to maturity = 0
        next_state[2:2+self.n_outcomes] = self.prices
        next_state[2+self.n_outcomes:] = self.probabilities
        
        done = True
        
        info = {
            'err': err,
            'portfolio': portfolio,
            'payoff': payoff,
            'final_price': S_T,
            'outcome_idx': outcome_idx,
            'delta': delta,
            'B': B
        }
        
        return next_state, reward, done, info
    
    def theoretical_hedge(self) -> Tuple[float, float, float]:
        """
        Calculate theoretical hedge ratio and fair price using risk-neutral valuation.
        Returns: (delta, B, fair_price)
        """
        # Weighted least squares solution for overdetermined system
        A = np.column_stack([self.prices, np.ones(self.n_outcomes)])
        b = np.array([max(S - self.K, 0) for S in self.prices])
        
        W = np.diag(self.probabilities)
        ATW = A.T @ W
        ATWA = ATW @ A
        ATWb = ATW @ b
        
        hedge_params = np.linalg.solve(ATWA, ATWb)
        delta_theory = hedge_params[0]
        B_theory = hedge_params[1]
        fair_price = delta_theory * self.S0 + B_theory
        
        return delta_theory, B_theory, fair_price


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
        
        s = torch.from_numpy(states)
        a = torch.from_numpy(actions)
        r = torch.from_numpy(rewards).unsqueeze(-1)
        ns = torch.from_numpy(next_states)
        d = torch.from_numpy(dones).unsqueeze(-1)
        
        return s, a, r, ns, d
    
    def __len__(self):
        return len(self.buffer)


# -------------------------
# Ornstein-Uhlenbeck Noise for Exploration
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
# Neural Networks for DDPG
# -------------------------
def hidden_init(layer):
    """Initialize hidden layers with uniform distribution."""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """
    Deterministic policy network for DDPG.
    Maps states to actions with proper constraints.
    """
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_sizes: List[int] = [400, 300],
                 init_w: float = 3e-3):
        super(Actor, self).__init__()
        
        # Build network layers
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        
        # Separate output heads for delta and B
        self.delta_head = nn.Linear(hidden_sizes[1], 1)  # Will apply sigmoid
        self.b_head = nn.Linear(hidden_sizes[1], 1)     # Unconstrained
        
        self.reset_parameters(init_w)
    
    def reset_parameters(self, init_w: float):
        """Initialize network parameters."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        
        # Initialize output layers
        self.delta_head.weight.data.uniform_(-init_w, init_w)
        self.b_head.weight.data.uniform_(-init_w, init_w)
        
        # Initialize biases
        self.delta_head.bias.data.uniform_(-init_w, init_w)
        self.b_head.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper action constraints."""
        # Handle both batch and single state
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Network forward pass
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # Output with constraints
        delta = torch.sigmoid(self.delta_head(x))  # Constrain to [0, 1]
        b = self.b_head(x)  # Unconstrained
        
        action = torch.cat([delta, b], dim=-1)
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action


class Critic(nn.Module):
    """
    Q-network for DDPG.
    Maps (state, action) pairs to Q-values.
    """
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
# DDPG Agent
# -------------------------
class DDPGAgent:
    """Deep Deterministic Policy Gradient agent for option hedging."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: List[int] = [400, 300],
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 1e-3,
                 noise_theta: float = 0.15,
                 noise_sigma: float = 0.2,
                 noise_decay: float = 0.999,
                 min_noise: float = 0.01,
                 device: str = 'cpu',
                 seed: int = 0):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_sizes).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_sizes).to(device)
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
        
        # Noise process for exploration
        self.noise = OrnsteinUhlenbeckNoise(action_dim, 
                                           theta=noise_theta,
                                           sigma=noise_sigma,
                                           seed=seed)
        self.noise_scale = 1.0
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        
        # Action bounds for clipping
        self.action_low = np.array([0.0, -np.inf])  # Delta >= 0
        self.action_high = np.array([1.0, np.inf])  # Delta <= 1
    
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
            
            # Clip action to valid range
            action = np.clip(action, self.action_low, self.action_high)
        
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
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item()
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
    
    def save(self, filepath: str):
        """Save agent parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'noise_scale': self.noise_scale,
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.noise_scale = checkpoint['noise_scale']


# -------------------------
# Training Function
# -------------------------
def train_ddpg_nnomial(
    n_outcomes: int = 5,
    episodes: int = 20000,
    batch_size: int = 128,
    buffer_size: int = 100000,
    hidden_sizes: List[int] = [400, 300],
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    start_steps: int = 1000,
    update_every: int = 1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 0,
    verbose: bool = True
) -> Tuple[DDPGAgent, OneStepNnomialEnv, List[Dict]]:
    """
    Train DDPG agent for n-nomial option hedging.
    
    Args:
        n_outcomes: Number of possible price outcomes
        episodes: Number of training episodes
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        hidden_sizes: Hidden layer sizes for networks
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        start_steps: Number of random steps before training
        update_every: Update networks every N steps
        device: Device to use for training
        seed: Random seed
        verbose: Whether to print progress
    
    Returns:
        agent: Trained DDPG agent
        env: Environment used for training
        training_history: List of training statistics
    """
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
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
        print(f"DDPG Training for {n_outcomes}-nomial Option Hedging")
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
    state_dim = env.state_dim
    action_dim = 2  # [delta, B]
    
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        device=device,
        seed=seed
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Training loop
    training_history = []
    total_steps = 0
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        agent.reset_noise()
        
        # Select action (random for initial exploration)
        if total_steps < start_steps:
            action = np.array([
                np.random.uniform(0, 1),  # Random delta in [0, 1]
                np.random.uniform(-50, 50)  # Random B
            ])
        else:
            action = agent.select_action(state, add_noise=True)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        replay_buffer.push(state, action, float(reward), next_state, float(done))
        total_steps += 1
        
        # Update agent
        if len(replay_buffer) >= batch_size and total_steps >= start_steps:
            if total_steps % update_every == 0:
                batch = replay_buffer.sample(batch_size)
                update_info = agent.update(batch)
                
                # Decay noise
                agent.decay_noise()
                
                # Store training info
                training_info = {
                    'episode': episode,
                    'total_steps': total_steps,
                    'hedging_error': info['err'],
                    'delta': info['delta'],
                    'B': info['B'],
                    'noise_scale': agent.noise_scale,
                    **update_info
                }
                training_history.append(training_info)
        
        # Logging
        if verbose and episode % 2000 == 0:
            # Evaluate current policy
            state = env.reset()
            with torch.no_grad():
                action = agent.select_action(state, add_noise=False)
            
            delta_current = action[0]
            B_current = action[1]
            fair_price_current = delta_current * env.S0 + B_current
            
            # Calculate average performance over multiple samples
            test_episodes = 100
            errors = []
            rewards = []
            
            for _ in range(test_episodes):
                state = env.reset()
                action = agent.select_action(state, add_noise=False)
                _, reward, _, info = env.step(action)
                errors.append(info['err'])
                rewards.append(reward)
            
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            avg_reward = np.mean(rewards)
            
            print(f"Episode {episode:6d} | Steps {total_steps:8d}")
            print(f"  Current: Δ={delta_current:.4f}, B={B_current:.3f}, Price={fair_price_current:.3f}")
            print(f"  Theory:  Δ={delta_theory:.4f}, B={B_theory:.3f}, Price={fair_price_theory:.3f}")
            print(f"  Performance: Error={avg_error:.4f}±{std_error:.4f}, Reward={avg_reward:.3f}")
            print(f"  Noise Scale: {agent.noise_scale:.4f}")
            
            if len(training_history) > 100:
                recent_q = np.mean([h['q_value'] for h in training_history[-100:]])
                print(f"  Avg Q-value: {recent_q:.3f}")
    
    # Final evaluation
    if verbose:
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}")
        
        state = env.reset()
        with torch.no_grad():
            final_action = agent.select_action(state, add_noise=False)
        
        delta_final = final_action[0]
        B_final = final_action[1]
        fair_price_final = delta_final * env.S0 + B_final
        
        # Test over many episodes
        test_episodes = 1000
        test_errors = []
        test_rewards = []
        test_deltas = []
        test_Bs = []
        
        for _ in range(test_episodes):
            state = env.reset()
            action = agent.select_action(state, add_noise=False)
            _, reward, _, info = env.step(action)
            test_errors.append(info['err'])
            test_rewards.append(reward)
            test_deltas.append(info['delta'])
            test_Bs.append(info['B'])
        
        print(f"Learned Policy:")
        print(f"  Delta: {delta_final:.4f} (Theory: {delta_theory:.4f}, Error: {abs(delta_final-delta_theory):.4f})")
        print(f"  B: {B_final:.4f} (Theory: {B_theory:.4f}, Error: {abs(B_final-B_theory):.4f})")
        print(f"  Fair Price: {fair_price_final:.4f} (Theory: {fair_price_theory:.4f}, Error: {abs(fair_price_final-fair_price_theory):.4f})")
        
        print(f"\nPerformance ({test_episodes} test episodes):")
        print(f"  Mean Hedging Error: {np.mean(test_errors):.6f}")
        print(f"  Std Hedging Error: {np.std(test_errors):.6f}")
        print(f"  Mean Squared Error: {np.mean(np.array(test_errors)**2):.6f}")
        print(f"  Mean Reward: {np.mean(test_rewards):.4f}")
        print(f"  95% VaR of |Error|: {np.percentile(np.abs(test_errors), 95):.4f}")
        print(f"  99% VaR of |Error|: {np.percentile(np.abs(test_errors), 99):.4f}")
        print(f"{'='*60}\n")