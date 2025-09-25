"""
SAC (Soft Actor-Critic) for One-Step N-nomial Option Hedging
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
            # Using log-normal inspired distribution
            min_return = -2 * volatility
            max_return = 2 * volatility
            returns = np.linspace(min_return, max_return, n_outcomes)
            self.prices = [S0 * np.exp(r) for r in returns]
        else:
            self.prices = [float(p) for p in prices]
            n_outcomes = len(self.prices)
        
        if probabilities is None:
            # Use discretized normal distribution probabilities
            # More realistic than equal probabilities
            returns = np.array([np.log(p/S0) for p in self.prices])
            # Normal PDF weights
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
        # State includes: [S0, time_to_maturity, price1, ..., priceN, prob1, ..., probN]
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
        
        # Reward is negative squared error (we want to minimize hedging error)
        reward = -(err ** 2)
        
        # Alternative reward formulations for experimentation:
        # reward = -abs(err)  # L1 loss
        # reward = -err**2 - 0.01 * abs(delta)  # With regularization
        
        # Terminal state
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        next_state[0] = S_T
        next_state[1] = 0.0  # Time to maturity = 0
        # Keep price and probability information
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
        # Expected payoff under physical measure
        expected_payoff = sum(p * max(S - self.K, 0) 
                            for S, p in zip(self.prices, self.probabilities))
        
        # For risk-neutral hedging, we need to solve the replication equations
        # Portfolio value at each outcome: delta * S_i + B = max(S_i - K, 0)
        # This is overdetermined for n > 2, so we use least squares
        
        A = np.column_stack([self.prices, np.ones(self.n_outcomes)])
        b = np.array([max(S - self.K, 0) for S in self.prices])
        
        # Weighted least squares using probabilities as weights
        W = np.diag(self.probabilities)
        ATW = A.T @ W
        ATWA = ATW @ A
        ATWb = ATW @ b
        
        # Solve for [delta, B]
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
        
        # Convert to numpy arrays first, then to tensors
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
# Neural Networks
# -------------------------
def mlp(sizes: List[int], 
        activation=nn.ReLU, 
        output_activation=nn.Identity,
        dropout_rate: float = 0.0):
    """Build a multi-layer perceptron."""
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        
        if j < len(sizes) - 2:
            layers += [activation()]
            if dropout_rate > 0:
                layers += [nn.Dropout(dropout_rate)]
        else:
            layers += [output_activation()]
    
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q-value network for SAC."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [256, 256]):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + hidden_sizes + [1])
    
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([s, a], -1)).squeeze(-1)


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network with proper action constraints.
    Delta is constrained to [0, 1] using sigmoid transformation.
    B is unconstrained (can be positive or negative).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [256, 256]):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_sizes, nn.ReLU, nn.ReLU)
        self.mean_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], act_dim)
        
        # Initialize outputs for better initial exploration
        with torch.no_grad():
            # Initialize mean for delta around 0 (sigmoid(0) = 0.5)
            self.mean_head.bias[0] = 0.0
            # Initialize B mean around 0
            if act_dim > 1:
                self.mean_head.bias[1] = 0.0
            
            # Initialize log_std to reasonable values
            self.log_std_head.bias.data.fill_(-1.0)  # std ≈ 0.37
    
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(s)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std, log_std
    
    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy with proper transformations.
        Returns: (action, log_prob, mean, log_std)
        """
        mean, std, log_std = self.forward(s)
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()  # Reparameterization trick
        
        # Apply transformations
        # Delta: use sigmoid to constrain to [0, 1]
        delta_raw = raw_action[:, 0:1]
        delta = torch.sigmoid(delta_raw)
        
        # B: keep unconstrained
        B = raw_action[:, 1:2]
        
        action = torch.cat([delta, B], dim=-1)
        
        # Compute log probability with Jacobian correction
        log_prob = normal.log_prob(raw_action).sum(-1, keepdim=True)
        
        # Jacobian correction for sigmoid transformation
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        jacobian_correction = torch.log(delta * (1 - delta) + EPS)
        log_prob = log_prob + jacobian_correction
        
        return action, log_prob, mean, log_std
    
    def deterministic(self, s: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean action with transformations)."""
        mean, _, _ = self.forward(s)
        delta = torch.sigmoid(mean[:, 0:1])
        B = mean[:, 1:2]
        return torch.cat([delta, B], -1)


# -------------------------
# SAC Agent
# -------------------------
class SACAgent:
    """Soft Actor-Critic agent for option hedging."""
    
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden_sizes: List[int] = [256, 256],
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 5e-3,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True,
                 device: str = 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        self.q1 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + 
                                     list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action from policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                action = self.policy.deterministic(state)
            else:
                action, _, _, _ = self.policy.sample(state)
        
        return action.squeeze(0).cpu().numpy()
    
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Update SAC networks."""
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update Q-functions
        with torch.no_grad():
            next_actions, next_log_probs, _, _ = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions).unsqueeze(-1)
            q2_next = self.q2_target(next_states, next_actions).unsqueeze(-1)
            q_next_min = torch.min(q1_next, q2_next)
            
            if self.automatic_entropy_tuning:
                alpha = torch.exp(self.log_alpha)
            else:
                alpha = self.alpha
            
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
        new_actions, new_log_probs, _, _ = self.policy.sample(states)
        q1_new = self.q1(states, new_actions).unsqueeze(-1)
        q2_new = self.q2(states, new_actions).unsqueeze(-1)
        q_new_min = torch.min(q1_new, q2_new)
        
        if self.automatic_entropy_tuning:
            alpha = torch.exp(self.log_alpha)
        else:
            alpha = self.alpha
        
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
        
        # Soft update target networks
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            'alpha': alpha.item() if self.automatic_entropy_tuning else alpha
        }
    
    def soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + 
                                  (1 - self.tau) * target_param.data)


# -------------------------
# Training Function
# -------------------------
def train_sac_nnomial(
    n_outcomes: int = 5,
    episodes: int = 20000,
    batch_size: int = 256,
    buffer_size: int = 100000,
    hidden_sizes: List[int] = [256, 256],
    lr: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 0,
    verbose: bool = True
) -> Tuple[SACAgent, OneStepNnomialEnv, List[Dict]]:
    """
    Train SAC agent for n-nomial option hedging.
    
    Returns:
        agent: Trained SAC agent
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
        print(f"SAC Training for {n_outcomes}-nomial Option Hedging")
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
    obs_dim = env.state_dim
    act_dim = 2  # [delta, B]
    
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        lr=lr,
        device=device
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Training loop
    training_history = []
    total_steps = 0
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        
        # Sample action
        action = agent.select_action(state, evaluate=False)
        
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
        if verbose and episode % 2000 == 0:
            # Evaluate current policy
            state = env.reset()
            with torch.no_grad():
                action = agent.select_action(state, evaluate=True)
            
            delta_current = action[0]
            B_current = action[1]
            fair_price_current = delta_current * env.S0 + B_current
            
            # Calculate average hedging error over multiple samples
            errors = []
            for _ in range(100):
                state = env.reset()
                action = agent.select_action(state, evaluate=True)
                _, _, _, info = env.step(action)
                errors.append(info['err'])
            
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            
            print(f"Episode {episode:6d} | Steps {total_steps:8d}")
            print(f"  Current: Δ={delta_current:.4f}, B={B_current:.3f}, Price={fair_price_current:.3f}")
            print(f"  Theory:  Δ={delta_theory:.4f}, B={B_theory:.3f}, Price={fair_price_theory:.3f}")
            print(f"  Avg Error: {avg_error:.4f} ± {std_error:.4f}")
            if len(training_history) > 0:
                recent_alpha = np.mean([h['alpha'] for h in training_history[-100:]])
                print(f"  Alpha: {recent_alpha:.4f}")
    
    # Final evaluation
    if verbose:
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}")
        
        state = env.reset()
        with torch.no_grad():
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
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Example 1: 5-nomial model
    print("\nTraining SAC for 5-nomial model...")
    agent_5, env_5, history_5 = train_sac_nnomial(
        n_outcomes=5,
        episodes=10000,
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example 2: 10-nomial model
    print("\n\nTraining SAC for 10-nomial model...")
    agent_10, env_10, history_10 = train_sac_nnomial(
        n_outcomes=10,
        episodes=15000,
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example 3: Custom trinomial model (up, middle, down)
    print("\n\nTraining SAC for custom trinomial model...")
    env_tri = OneStepNnomialEnv(
        S0=100.0,
        K=110.0,
        prices=[80.0, 100.0, 140.0],  # Down, middle, up
        probabilities=[0.33, 0.34, 0.33],  # Custom probabilities
        seed=42
    )
    
    agent_tri = SACAgent(
        obs_dim=env_tri.state_dim,
        act_dim=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Quick training for trinomial
    replay_buffer = ReplayBuffer()
    for episode in range(5000):
        state = env_tri.reset()
        action = agent_tri.select_action(state)
        next_state, reward, done, info = env_tri.step(action)
        replay_buffer.push(state, action, float(reward), next_state, float(done))
        
        if len(replay_buffer) >= 128:
            batch = replay_buffer.sample(128)
            agent_tri.update(batch)
    
    # Evaluate trinomial
    state = env_tri.reset()
    action = agent_tri.select_action(state, evaluate=True)
    print(f"\nTrinomial Model Results:")
    print(f"  Delta: {action[0]:.4f}, B: {action[1]:.4f}")
    print(f"  Fair Price: {action[0] * env_tri.S0 + action[1]:.4f}")