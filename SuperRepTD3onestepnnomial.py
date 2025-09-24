"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) for One-Step N-nomial Option Hedging
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
# Replay Buffer for TD3
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
# TD3 Networks
# -------------------------
class Actor(nn.Module):
    """Actor network for TD3 with bounded actions."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(Actor, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize final layer weights to small values
        nn.init.uniform_(self.network[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.network[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Output raw actions
        raw_actions = self.network(state)
        
        # Apply transformations for bounded actions
        # Delta: sigmoid for [0, 1], B: tanh for [-1, 1] then scale
        delta = torch.sigmoid(raw_actions[:, 0:1])
        B = torch.tanh(raw_actions[:, 1:2]) * 200  # Scale bond position
        
        return torch.cat([delta, B], dim=-1)

class Critic(nn.Module):
    """Twin Q-network for TD3."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(Critic, self).__init__()
        
        # Q1 network
        q1_layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            q1_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        q1_layers.append(nn.Linear(prev_dim, 1))
        self.q1_network = nn.Sequential(*q1_layers)
        
        # Q2 network
        q2_layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            q2_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2_network = nn.Sequential(*q2_layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q1_network(x)

# -------------------------
# TD3 Agent
# -------------------------
class TD3Agent:
    """TD3 agent for option hedging."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 5e-3,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
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
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Select action from policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state)
        
        action = action.squeeze(0).cpu().numpy()
        
        if add_noise:
            # Add exploration noise
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = action + noise
            # Clip final action
            action[0] = np.clip(action[0], 0, 1)  # Delta in [0, 1]
            action[1] = np.clip(action[1], -200, 200)  # B in reasonable range
        
        return action
    
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Update TD3 networks."""
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        with torch.no_grad():
            # Select next action with target policy and add noise
            next_actions = self.actor_target(next_states)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(
                torch.tensor([0.0, -200.0], device=self.device),
                torch.tensor([1.0, 200.0], device=self.device))
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Delayed policy updates
        actor_loss = torch.tensor(0.0)
        if self.total_iterations % self.policy_freq == 0:
            # Compute actor loss
            actor_actions = self.actor(states)
            actor_q1 = self.critic.q1(states, actor_actions)
            actor_loss = -actor_q1.mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Update target networks
            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.actor, self.actor_target)
        
        self.total_iterations += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss != 0 else 0.0,
            'q_value': current_q1.mean().item()
        }
    
    def soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + 
                                  (1 - self.tau) * target_param.data)

# -------------------------
# TD3 Training Function
# -------------------------
def train_td3_nnomial(
    n_outcomes: int = 5,
    episodes: int = 10000,
    batch_size: int = 256,
    buffer_size: int = 100000,
    hidden_dims: List[int] = [256, 256],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    verbose: bool = True
) -> Tuple[TD3Agent, OneStepNnomialEnv, List[Dict]]:
    """
    Train TD3 agent for n-nomial option hedging.
    
    Returns:
        agent: Trained TD3 agent
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
        print(f"TD3 Training for {n_outcomes}-nomial Option Hedging")
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
    agent = TD3Agent(
        state_dim=env.state_dim,
        action_dim=2,
        hidden_dims=hidden_dims,
        device=device
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Training loop
    training_history = []
    total_steps = 0
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        
        # Sample action with exploration noise
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
            
            # Store training info
            training_info = {
                'episode': episode,
                'total_steps': total_steps,
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
            with torch.no_grad():
                action = agent.select_action(state, add_noise=False)
            
            delta_current = action[0]
            B_current = action[1]
            fair_price_current = delta_current * env.S0 + B_current
            
            # Calculate average hedging error
            errors = []
            for _ in range(100):
                state = env.reset()
                action = agent.select_action(state, add_noise=False)
                _, _, _, info = env.step(action)
                errors.append(info['err'])
            
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            
            print(f"Episode {episode:6d} | Steps {total_steps:8d}")
            print(f"  Current: Δ={delta_current:.4f}, B={B_current:.3f}, Price={fair_price_current:.3f}")
            print(f"  Theory:  Δ={delta_theory:.4f}, B={B_theory:.3f}, Price={fair_price_theory:.3f}")
            print(f"  Avg Error: {avg_error:.4f} ± {std_error:.4f}")
    
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
        test_errors = []
        test_rewards = []
        for _ in range(1000):
            state = env.reset()
            action = agent.select_action(state, add_noise=False)
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
# Main Execution for TD3
# -------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Example 1: 5-nomial model
    print("\nTraining TD3 for 5-nomial model...")
    agent_5, env_5, history_5 = train_td3_nnomial(
        n_outcomes=5,
        episodes=5000,
        batch_size=256,
        device=device
    )
    
    # Example 2: 10-nomial model
    print("\n\nTraining TD3 for 10-nomial model...")
    agent_10, env_10, history_10 = train_td3_nnomial(
        n_outcomes=10,
        episodes=8000,
        batch_size=256,
        device=device
    )
    
    # Example 3: Custom binomial model
    print("\n\nTraining TD3 for custom binomial model...")
    env_bi = OneStepNnomialEnv(
        S0=100.0,
        K=110.0,
        prices=[90.0, 120.0],  # Down, up
        probabilities=[0.4, 0.6],
        seed=42
    )
    
    agent_bi = TD3Agent(
        state_dim=env_bi.state_dim,
        action_dim=2,
        device=device
    )
    
    # Quick training for binomial
    replay_buffer = ReplayBuffer()
    for episode in range(3000):
        state = env_bi.reset()
        action = agent_bi.select_action(state, add_noise=True)
        next_state, reward, done, info = env_bi.step(action)
        replay_buffer.push(state, action, float(reward), next_state, float(done))
        
        if len(replay_buffer) >= 128:
            batch = replay_buffer.sample(128)
            agent_bi.update(batch)
    
    # Evaluate binomial
    state = env_bi.reset()
    action = agent_bi.select_action(state, add_noise=False)
    delta_theory, B_theory, fair_price_theory = env_bi.theoretical_hedge()
    
    print(f"\nBinomial Model Results:")
    print(f"  Learned: Δ={action[0]:.4f}, B={action[1]:.4f}, Price={action[0] * env_bi.S0 + action[1]:.4f}")
    print(f"  Theory:  Δ={delta_theory:.4f}, B={B_theory:.4f}, Price={fair_price_theory:.4f}")