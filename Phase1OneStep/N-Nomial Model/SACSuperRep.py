"""
SAC (Soft Actor-Critic) for One-Step N-nomial Option Hedging
Perfect Market Assumptions: No interest rates, no transaction costs
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
    """One-step N-nomial environment for option hedging."""
    
    def __init__(self, S0: float, K: float, prices: List[float], probabilities: List[float], seed: int = 0):
        self.S0 = float(S0)
        self.K = float(K)
        self.prices = [float(p) for p in prices]
        self.probabilities = np.array(probabilities, dtype=np.float32)
        self.n_outcomes = len(self.prices)
        
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Simplified state: just include essential info
        self.state_dim = 2  # [S0, K] - the model parameters
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = np.array([self.S0, self.K], dtype=np.float32)
        return self.state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment."""
        delta = float(action[0])
        B = float(action[1])
        
        # Sample final price according to probabilities
        outcome_idx = self.np_rng.choice(self.n_outcomes, p=self.probabilities)
        S_T = self.prices[outcome_idx]
        
        # Calculate option payoff and portfolio value
        payoff = max(S_T - self.K, 0.0)
        portfolio = delta * S_T + B
        
        # Hedging error
        hedging_error = portfolio - payoff
        
        # Strong penalty for large errors - focus the learning
        reward = -(hedging_error ** 2) / 100.0
        
        # Simple terminal state
        next_state = np.array([S_T, self.K], dtype=np.float32)
        
        info = {
            'hedging_error': hedging_error,
            'portfolio_value': portfolio,
            'option_payoff': payoff,
            'final_price': S_T,
            'delta': delta,
            'B': B
        }
        
        return next_state, reward, True, info
    
    def theoretical_hedge(self) -> Tuple[float, float, float]:
        """Calculate theoretical hedge using least squares replication."""
        A = np.column_stack([self.prices, np.ones(self.n_outcomes)])
        b = np.array([max(S - self.K, 0) for S in self.prices])
        
        # Weighted least squares
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
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays first to avoid tensor creation warning
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
# Neural Networks
# -------------------------
def create_network(input_dim: int, output_dim: int, hidden_sizes: List[int] = [64, 64]):
    """Create a simple MLP network."""
    layers = []
    sizes = [input_dim] + hidden_sizes + [output_dim]
    
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q-value network for SAC."""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.q = create_network(obs_dim + act_dim, 1, [64, 64])
    
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([s, a], -1)).squeeze(-1)


class PolicyNetwork(nn.Module):
    """Policy network with no bias toward specific hedge values."""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = create_network(obs_dim, act_dim * 2, [64, 64])  # Mean and log_std
        
        # Neutral initialization - no assumptions about optimal values
        with torch.no_grad():
            # Small random weights, zero bias - let SAC learn everything from scratch
            self.net[-1].weight.data *= 0.01
            self.net[-1].bias.data.zero_()  # No bias toward any specific delta or B
            # Set log_std to allow reasonable exploration initially
            self.net[-1].bias.data[act_dim:] = -2.0  # log_std = -2, std ≈ 0.135
    
    def forward(self, s: torch.Tensor):
        output = self.net(s)
        mean = output[:, :2]  # [delta, B] - no constraints, let SAC learn
        log_std = output[:, 2:].clamp(-4, -0.5)  # Reasonable exploration range
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, s: torch.Tensor):
        mean, std = self.forward(s)
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob
    
    def deterministic(self, s: torch.Tensor):
        mean, _ = self.forward(s)
        return mean


# -------------------------
# Simple SAC Agent
# -------------------------
class SimpleSACAgent:
    """SAC agent with no biases - works for any n-nomial case."""
    
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 3e-4, device: str = 'cpu'):
        self.device = device
        self.gamma = 0.0  # No discounting for single-step
        self.tau = 0.005  # Standard target update rate
        
        # Networks
        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(device)
        self.policy = PolicyNetwork(obs_dim, act_dim).to(device)
        
        # Copy weights to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Equal learning rates for all networks
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Automatic entropy tuning - let SAC find the right exploration level
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action = self.policy.deterministic(state)
            else:
                action, _ = self.policy.sample(state)
        return action.squeeze(0).cpu().numpy()
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current alpha value
        current_alpha = torch.exp(self.log_alpha).detach()
        
        # Q-function updates
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions).unsqueeze(-1)
            q2_next = self.q2_target(next_states, next_actions).unsqueeze(-1)
            q_target = rewards + (1 - dones) * self.gamma * (torch.min(q1_next, q2_next) - current_alpha * next_log_probs)
        
        q1_pred = self.q1(states, actions).unsqueeze(-1)
        q2_pred = self.q2(states, actions).unsqueeze(-1)
        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.q_optimizer.step()
        
        # Policy update
        new_actions, new_log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions).unsqueeze(-1)
        q2_new = self.q2(states, new_actions).unsqueeze(-1)
        policy_loss = (current_alpha * new_log_probs - torch.min(q1_new, q2_new)).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Alpha (entropy) update
        alpha_loss = -(self.log_alpha * (new_log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update targets
        for source, target in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q_loss': q_loss.item(), 
            'policy_loss': policy_loss.item(), 
            'alpha_loss': alpha_loss.item(),
            'alpha': current_alpha.item()
        }


# -------------------------
# Training Function
# -------------------------
def train_sac_nnomial(
    prices: List[float],
    probabilities: List[float],
    S0: float = 100.0,
    K: float = 110.0,
    episodes: int = 50000,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 0,
    verbose: bool = True
):
    """Train SAC agent for n-nomial option hedging."""
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = OneStepNnomialEnv(S0=S0, K=K, prices=prices, probabilities=probabilities, seed=seed)
    
    # Get theoretical solution
    delta_theory, B_theory, fair_price_theory = env.theoretical_hedge()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SAC Training for {env.n_outcomes}-nomial Option Hedging")
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
    agent = SimpleSACAgent(obs_dim=env.state_dim, act_dim=2, lr=lr, device=device)
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=20000)
    
    # Training loop
    for episode in range(1, episodes + 1):
        state = env.reset()
        action = agent.select_action(state, evaluate=False)
        next_state, reward, done, info = env.step(action)
        
        replay_buffer.push(state, action, float(reward), next_state, float(done))
        
        # Update agent
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            agent.update(batch)
    
    # Final evaluation
    if verbose:
        print(f"\n{'='*70}")
        print("Final Evaluation")
        print(f"{'='*70}")
        
        # Test final policy
        state = env.reset()
        final_action = agent.select_action(state, evaluate=True)
        delta_final, B_final = final_action[0], final_action[1]
        portfolio_value_final = delta_final * env.S0 + B_final
        
        # Calculate replication errors
        errors = []
        for S_i in env.prices:
            portfolio = delta_final * S_i + B_final
            payoff = max(S_i - env.K, 0)
            errors.append(portfolio - payoff)
        
        mse = np.mean([e**2 for e in errors])
        max_error = max(abs(e) for e in errors)
        
        print(f"Learned Policy:")
        print(f"  Delta: {delta_final:.4f} (Theory: {delta_theory:.4f})")
        print(f"  B: {B_final:.4f} (Theory: {B_theory:.4f})")
        print(f"  Portfolio Value: {portfolio_value_final:.4f} (Theory: {fair_price_theory:.4f})")
        print(f"\nReplication Performance:")
        print(f"  MSE: {mse:.6f}")
        print(f"  Max |Error|: {max_error:.6f}")
        
        print(f"\nError by Scenario:")
        for i, (price, prob, error) in enumerate(zip(env.prices, env.probabilities, errors)):
            portfolio = delta_final * price + B_final
            payoff = max(price - env.K, 0)
            print(f"  S={price:.1f} (p={prob:.3f}): Portfolio={portfolio:.3f}, Payoff={payoff:.3f}, Error={error:.4f}")
        
        print(f"{'='*70}")
    
    return agent, env


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    
    # Configure your n-nomial model here
    prices = [80.0, 140.0]
    probabilities = [0.5, 0.5]
    
    # Trinomial example:
    # prices = [80.0, 110.0, 140.0]
    # probabilities = [0.33, 0.34, 0.33]
    
    print("Training SAC for N-nomial Option Hedging...")
    print(f"Model: {len(prices)}-nomial")
    print(f"Prices: {prices}")
    print(f"Probabilities: {probabilities}")
    
    agent, env = train_sac_nnomial(
        prices=prices,
        probabilities=probabilities,
        S0=100.0,
        K=110.0,
        episodes=50000,
        batch_size=128,
        lr=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )