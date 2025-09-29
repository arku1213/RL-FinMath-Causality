"""
Multi-Step SAC (Soft Actor-Critic) for N-nomial Option Hedging
Enhanced with better reward structure and state representation
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
# Multi-Step N-nomial Environment (Enhanced)
# -------------------------
class MultiStepNnomialEnv:
    """Multi-step N-nomial environment for dynamic option hedging with enhanced rewards."""
    
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
        
        # Enhanced state: [current_price, time_remaining, moneyness, portfolio_value, option_intrinsic_value, theoretical_fair_price]
        self.state_dim = 6
        
        # Calculate theoretical fair price for reference
        self.initial_fair_price = self._calculate_theoretical_fair_price()
        
        # Initialize episode variables
        self.reset()
    
    def _calculate_theoretical_fair_price(self) -> float:
        """Calculate theoretical option fair price using proper binomial valuation."""
        # Use exact binomial pricing formula
        u = max(self.prices_per_step)  # Up multiplier
        d = min(self.prices_per_step)  # Down multiplier
        p = self.probabilities[self.prices_per_step.index(u)]  # Probability of up move
        
        # Calculate option value using backward induction
        def binomial_option_price(S, T, K, u, d, p):
            # Build price tree
            prices = {}
            for i in range(T + 1):
                for j in range(i + 1):
                    prices[(i, j)] = S * (u ** j) * (d ** (i - j))
            
            # Terminal payoffs
            option_values = {}
            for j in range(T + 1):
                option_values[(T, j)] = max(prices[(T, j)] - K, 0)
            
            # Work backwards
            for i in range(T - 1, -1, -1):
                for j in range(i + 1):
                    option_values[(i, j)] = p * option_values[(i + 1, j + 1)] + (1 - p) * option_values[(i + 1, j)]
            
            return option_values[(0, 0)]
        
        return binomial_option_price(self.S0, self.T, self.K, u, d, p)
    
    def _get_current_theoretical_price(self, current_price: float, time_remaining: int) -> float:
        """Get theoretical price for current state using proper binomial pricing."""
        if time_remaining == 0:
            return max(current_price - self.K, 0.0)
        
        u = max(self.prices_per_step)
        d = min(self.prices_per_step)
        p = self.probabilities[self.prices_per_step.index(u)]
        
        # Calculate option value from current state
        def binomial_from_current(S, T, K, u, d, p):
            if T == 0:
                return max(S - K, 0)
            
            # One step ahead values
            up_price = S * u
            down_price = S * d
            up_value = binomial_from_current(up_price, T - 1, K, u, d, p)
            down_value = binomial_from_current(down_price, T - 1, K, u, d, p)
            
            return p * up_value + (1 - p) * down_value
        
        return binomial_from_current(current_price, time_remaining, self.K, u, d, p)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_time = 0
        self.current_price = self.S0
        self.portfolio_value = 0.0
        
        # Calculate initial values
        option_intrinsic = max(self.current_price - self.K, 0.0)
        theoretical_price = self._get_current_theoretical_price(self.current_price, self.T - self.current_time)
        
        state = np.array([
            self.current_price / 100.0,  # Normalized current price
            (self.T - self.current_time) / self.T,  # Normalized time remaining
            self.current_price / self.K,  # Moneyness
            0.0,  # Initial portfolio value (normalized)
            option_intrinsic / 100.0,  # Normalized option intrinsic value
            theoretical_price / 100.0   # Normalized theoretical fair price
        ], dtype=np.float32)
        
        self.state = state
        return self.state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a rebalancing step with enhanced reward structure."""
        delta = float(action[0])
        B = float(action[1])
        
        # Update portfolio value based on action
        self.portfolio_value = delta * self.current_price + B
        
        # Advance time
        self.current_time += 1
        done = (self.current_time >= self.T)
        
        # Always update price first, regardless of done status
        price_multiplier_idx = self.np_rng.choice(self.n_outcomes, p=self.probabilities)
        price_multiplier = self.prices_per_step[price_multiplier_idx]
        new_price = self.current_price * price_multiplier
        self.current_price = new_price
        
        if not done:
            # Intermediate step logic with enhanced rewards
            option_intrinsic = max(self.current_price - self.K, 0.0)
            theoretical_price = self._get_current_theoretical_price(self.current_price, self.T - self.current_time)
            
            # Enhanced rewards with stronger portfolio size constraints
            position_change = abs(delta - getattr(self, 'previous_delta', 0.5))
            rebalancing_cost = -0.001 * position_change
            
            # FIXED: Strong portfolio size penalty - penalize deviation from theoretical value
            portfolio_size_penalty = 0.0
            target_portfolio = theoretical_price
            portfolio_deviation = abs(self.portfolio_value - target_portfolio)
            
            if portfolio_deviation > target_portfolio * 0.5:  # More than 50% deviation
                portfolio_size_penalty = -0.1 * portfolio_deviation
            elif portfolio_deviation > target_portfolio * 0.2:  # More than 20% deviation  
                portfolio_size_penalty = -0.02 * portfolio_deviation
            
            # FIXED: Remove problematic penalties that caused over-hedging
            # Just use simple rebalancing cost + portfolio size constraint
            reward = rebalancing_cost + portfolio_size_penalty
            
            # Store previous delta for next step
            self.previous_delta = delta
            
            # Update portfolio value for new price
            new_portfolio = delta * self.current_price + B
            self.portfolio_value = new_portfolio
            
            # Next state with theoretical price
            next_state = np.array([
                self.current_price / 100.0,
                (self.T - self.current_time) / self.T,
                self.current_price / self.K,
                self.portfolio_value / 100.0,
                option_intrinsic / 100.0,
                theoretical_price / 100.0
            ], dtype=np.float32)
            
            info = {
                'hedging_error': 0.0,
                'portfolio_value': self.portfolio_value,
                'option_payoff': option_intrinsic,
                'current_price': self.current_price,
                'delta': delta,
                'B': B,
                'time_step': self.current_time,
                'rebalancing_cost': rebalancing_cost,
                'portfolio_size_penalty': portfolio_size_penalty,
                'theoretical_price': theoretical_price,
                'target_portfolio': target_portfolio
            }
            
        else:
            # Terminal step with enhanced terminal reward
            option_payoff = max(self.current_price - self.K, 0.0)
            final_portfolio = delta * self.current_price + B
            
            hedging_error = final_portfolio - option_payoff
            
            # Enhanced terminal reward
            base_terminal_reward = -(hedging_error ** 2) / 100.0
            
            # NEW: Additional penalty for completely missing the hedge
            missed_hedge_penalty = 0.0
            if option_payoff > 0 and final_portfolio <= 0:
                missed_hedge_penalty = -5.0  # Large penalty for completely missing ITM option
            elif option_payoff == 0 and final_portfolio < -10:
                missed_hedge_penalty = -1.0  # Penalty for large negative portfolio when option worthless
            
            reward = base_terminal_reward + missed_hedge_penalty
            
            # Terminal state
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
                'missed_hedge_penalty': missed_hedge_penalty,
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
            'implied_volatility': np.sqrt(var_log_return * self.T),
            'initial_fair_price': self.initial_fair_price
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
# SAC Networks (Enhanced for larger state)
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
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
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
    """Gaussian policy network for SAC with entropy regularization."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = [256, 256], strike_price: float = 110.0):
        super().__init__()
        self.K = strike_price

        # Build network
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, act_dim)
        self.log_std_head = nn.Linear(prev_dim, act_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for reasonable exploration."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize output heads for reasonable initial exploration
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
        """Sample action with constraints."""
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        
        mean, log_std = self.forward(s)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Delta: [0, 1] for call option hedging
        delta = torch.sigmoid(x_t[:, 0:1])
        
        # B: [-100, 100] for better flexibility
        B = 100.0 * torch.tanh(x_t[:, 1:2])
        
        action = torch.cat([delta, B], dim=-1)
        
        # Calculate log probability with Jacobian correction
        log_prob = normal.log_prob(x_t).sum(axis=-1, keepdim=True)
        
        # Jacobian correction for sigmoid (delta)
        log_prob -= torch.log(delta * (1 - delta) + EPS).sum(axis=-1, keepdim=True)
        
        # Jacobian correction for B transformation
        B_raw = torch.tanh(x_t[:, 1:2])
        log_prob -= torch.log(100.0 * (1 - B_raw.pow(2)) + EPS).sum(axis=-1, keepdim=True)
        
        return action, log_prob
    
    def deterministic_action(self, s: torch.Tensor) -> torch.Tensor:
        """Get deterministic action."""
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
            
        mean, _ = self.forward(s)
        
        # Apply same constraints as in sample()
        delta = torch.sigmoid(mean[:, 0:1])
        B = 100.0 * torch.tanh(mean[:, 1:2])
        
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
                 alpha: float = 0.5,  # Higher initial alpha for more exploration
                 automatic_entropy_tuning: bool = True,
                 device: str = 'cpu',
                 strike_price: float = 110.0):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dims, strike_price=strike_price).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + 
                                     list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Automatic entropy tuning with higher initial value
        if automatic_entropy_tuning:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action from policy."""
        state = torch.FloatTensor(state).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                action = self.policy.deterministic_action(state)
        else:
            with torch.no_grad():
                action, _ = self.policy.sample(state)
        
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
    T: int = 3,
    n_outcomes: int = 3,
    prices_per_step: List[float] = [0.9, 1.0, 1.1],
    probabilities: List[float] = [0.25, 0.5, 0.25],
    S0: float = 100.0,
    K: float = 110.0,
    episodes: int = 20000,
    batch_size: int = 256,
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
        print(f"Enhanced Multi-Step SAC Training for {T}-Step {n_outcomes}-nomial Option Hedging")
        print(f"{'='*80}")
        print(f"Environment: S0={S0}, K={K}, T={T} steps")
        print(f"Price multipliers per step: {prices_per_step}")
        print(f"Probabilities: {probabilities}")
        print(f"Implied volatility: {price_stats['implied_volatility']:.4f}")
        print(f"Theoretical initial fair price: ${price_stats['initial_fair_price']:.2f}")
        print(f"Enhanced rewards: Portfolio realism, Delta magnitude, Missed hedge penalties")
        print(f"{'='*80}\n")
    
    # Create agent
    agent = MultiStepSACAgent(
        obs_dim=env.state_dim,  # Now 6 instead of 5
        act_dim=2,
        hidden_dims=hidden_dims,
        device=device,
        strike_price=K
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
        
        # Enhanced logging
        if verbose and episode % 2500 == 0:
            # Evaluate current policy
            eval_rewards = []
            eval_errors = []
            eval_portfolios = []
            
            for _ in range(10):
                state = env.reset()
                eval_reward = 0
                initial_portfolio = None
                
                while True:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = env.step(action)
                    eval_reward += reward
                    
                    if initial_portfolio is None:
                        initial_portfolio = info.get('portfolio_value', 0)
                    
                    state = next_state
                    
                    if done:
                        if 'hedging_error' in info:
                            eval_errors.append(abs(info['hedging_error']))
                        break
                
                eval_rewards.append(eval_reward)
                if initial_portfolio is not None:
                    eval_portfolios.append(initial_portfolio)
            
            avg_reward = np.mean(eval_rewards)
            avg_error = np.mean(eval_errors) if eval_errors else 0
            avg_initial_portfolio = np.mean(eval_portfolios) if eval_portfolios else 0
            recent_training_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            # Get recent alpha value
            if len(replay_buffer) >= batch_size and episode > 50:
                dummy_batch = replay_buffer.sample(batch_size)
                update_info = agent.update(dummy_batch)
                current_alpha = update_info['alpha']
            else:
                current_alpha = 0.5
            
            print(f"Episode {episode:5d}")
            print(f"  Training reward: {recent_training_reward:.4f}")
            print(f"  Eval reward: {avg_reward:.4f}")
            print(f"  Avg hedging error: {avg_error:.4f}")
            print(f"  Avg initial portfolio: {avg_initial_portfolio:.2f} (target: ~{price_stats['initial_fair_price']:.2f})")
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
        final_initial_portfolios = []
        
        # Run multiple evaluation episodes
        for eval_ep in range(100):
            state = env.reset()
            episode_path = []
            episode_reward = 0
            initial_portfolio = None
            
            while True:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                if initial_portfolio is None:
                    initial_portfolio = info.get('portfolio_value', 0)
                
                episode_path.append({
                    'time_step': info.get('time_step', 0),
                    'price': info.get('current_price', 0),
                    'delta': info.get('delta', 0),
                    'B': info.get('B', 0),
                    'portfolio': info.get('portfolio_value', 0),
                    'theoretical_price': info.get('theoretical_price', 0)
                })
                
                state = next_state
                
                if done:
                    if 'hedging_error' in info:
                        final_errors.append(info['hedging_error'])
                    break
            
            final_rewards.append(episode_reward)
            if initial_portfolio is not None:
                final_initial_portfolios.append(initial_portfolio)
            if eval_ep < 3:  # Store first 3 paths for analysis
                final_paths.append(episode_path)
        
        print(f"Final Performance (100 episodes):")
        print(f"  Mean reward: {np.mean(final_rewards):.6f}")
        print(f"  Std reward: {np.std(final_rewards):.6f}")
        print(f"  Mean |hedging error|: {np.mean(np.abs(final_errors)):.6f}")
        print(f"  Std |hedging error|: {np.std(np.abs(final_errors)):.6f}")
        print(f"  Max |hedging error|: {np.max(np.abs(final_errors)):.6f}")
        print(f"  MSE: {np.mean(np.array(final_errors)**2):.6f}")
        
        if final_initial_portfolios:
            print(f"  Mean initial portfolio: {np.mean(final_initial_portfolios):.2f} (target: ~{price_stats['initial_fair_price']:.2f})")
            print(f"  Portfolio improvement: {(np.mean(final_initial_portfolios) / price_stats['initial_fair_price'] * 100):.1f}% of theoretical")
        
        print(f"\nSample Episode Path with Theoretical Comparison:")
        print(f"{'Step':>4} | {'Price':>6} | {'Delta':>6} | {'B':>7} | {'Portfolio':>9} | {'Theoretical':>11}")
        print("-" * 70)
        if final_paths:
            for step in final_paths[0]:
                print(f"{step['time_step']:4d} | "
                      f"${step['price']:5.2f} | "
                      f"{step['delta']:6.4f} | "
                      f"${step['B']:6.2f} | "
                      f"${step['portfolio']:8.2f} | "
                      f"${step.get('theoretical_price', 0):10.2f}")
        
        print(f"{'='*80}")
    
    return agent, env, episode_rewards, hedging_errors


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    
    # Configure your multi-step n-nomial model
    T = 4  # Number of time steps
    n_outcomes = 2  
    prices_per_step = [0.8, 1.2]  # Down, up
    probabilities = [0.5, 0.5]  # Probabilities for each outcome

    print("Training Enhanced Multi-Step SAC for N-nomial Option Hedging...")
    print(f"Configuration: {T} steps, {n_outcomes}-nomial per step")
    print(f"Price multipliers: {prices_per_step}")
    print(f"Probabilities: {probabilities}")
    print("Enhancements: Better rewards, theoretical price tracking, higher exploration")
    
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
    
    print(f"\nTraining completed!")
    print(f"Final results:")
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Final reward trend: {np.mean(rewards[-100:]):.4f}")
    if errors:
        print(f"  Final hedging error trend: {np.mean(errors[-100:]):.4f}")
    print(f"  Agent saved and ready for use.")