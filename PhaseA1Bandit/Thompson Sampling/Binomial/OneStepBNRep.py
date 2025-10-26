import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

class BinomialOptionEnvironment:
    """One-step binomial model for option pricing"""
    def __init__(self, S0=100, K=100, r=0.05, T=1.0, u=1.2, d=0.8):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.u = u
        self.d = d
        
        self.q = (np.exp(r * T) - d) / (u - d)
        self.Su = S0 * u
        self.Sd = S0 * d
        self.Cu = max(self.Su - K, 0)
        self.Cd = max(self.Sd - K, 0)
        self.C0 = np.exp(-r * T) * (self.q * self.Cu + (1 - self.q) * self.Cd)
        
    def get_context(self):
        """Get context (static in this case)"""
        return np.array([self.S0, self.K], dtype=np.float32)
    
    def evaluate_hedge(self, delta, B):
        """Evaluate hedge on both scenarios"""
        pnls = []
        
        for step_type in [1, 0]:
            if step_type == 1:
                ST = self.Su
                CT = self.Cu
            else:
                ST = self.Sd
                CT = self.Cd
            
            portfolio_value = delta * ST + B * np.exp(self.r * self.T)
            pnl = portfolio_value - CT
            pnls.append(pnl)
        
        pnl_up, pnl_down = pnls
        
        # Hedging error
        hedging_error = (pnl_up ** 2 + pnl_down ** 2) / 2.0
        
        # Constraint: 0 < delta*S0 + B <= 50
        hedge_price = delta * self.S0 + B
        constraint_penalty = 0
        
        if hedge_price <= 0:
            constraint_penalty = 200 * ((hedge_price - 1) ** 2)
        elif hedge_price > 50:
            constraint_penalty = 100 * ((hedge_price - 50) ** 2)
        
        reward = -(hedging_error + constraint_penalty) / 100.0
        
        return reward, pnl_up, pnl_down, hedge_price


class ContextualBanditNetwork(nn.Module):
    """Neural network for contextual bandit (NO BIAS)"""
    def __init__(self, context_dim=2, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(context_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, 2, bias=False)  # [delta, B]
        
    def forward(self, context):
        x = torch.relu(self.fc1(context))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        
        # Delta in [0, 1]
        delta = torch.sigmoid(output[:, 0])
        
        # B - allow full range
        B = output[:, 1] * 10  # Linear output
        
        return delta, B


class LinUCBBandit:
    """
    LinUCB for Continuous Actions
    
    Instead of discrete arms, we:
    1. Sample candidate actions
    2. For each, predict reward using linear model: r = θᵀ·[context, action]
    3. Add UCB bonus for uncertainty
    4. Pick action with highest UCB
    """
    def __init__(self, context_dim=2, action_dim=2, alpha=1.0):
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.alpha = alpha  # Exploration parameter
        
        # Single linear model: reward = θᵀ·[context, action]
        feature_dim = context_dim + action_dim
        self.A = np.identity(feature_dim)
        self.b = np.zeros(feature_dim)
        
        self.history = []
        
    def features(self, context, delta, B):
        """Combine context and action into feature vector"""
        return np.concatenate([context, [delta, B]])
    
    def predict_reward(self, context, delta, B):
        """Predict reward with uncertainty"""
        x = self.features(context, delta, B)
        
        # Estimate parameters
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        
        # Predicted reward
        pred_reward = theta @ x
        
        # Uncertainty (UCB bonus)
        uncertainty = self.alpha * np.sqrt(x @ A_inv @ x)
        
        # UCB = prediction + exploration bonus
        ucb = pred_reward + uncertainty
        
        return ucb, pred_reward, uncertainty
    
    def select_action(self, context, n_samples=200):
        """Sample actions and pick best by UCB"""
        best_ucb = -float('inf')
        best_delta, best_B = None, None
        
        for _ in range(n_samples):
            # Sample random action respecting constraints
            delta = np.random.uniform(0, 1)
            B_min = -delta * 100 + 0.1  # Constraint: delta*S0 + B > 0
            B_max = 50 - delta * 100     # Constraint: delta*S0 + B <= 50
            B = np.random.uniform(max(-80, B_min), min(50, B_max))
            
            # Compute UCB
            ucb, pred, unc = self.predict_reward(context, delta, B)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_delta = delta
                best_B = B
        
        return best_delta, best_B
    
    def update(self, context, delta, B, reward):
        """Update linear model with observed (context, action, reward)"""
        x = self.features(context, delta, B)
        
        # Standard LinUCB update
        self.A += np.outer(x, x)
        self.b += reward * x
        
        self.history.append((delta, B, reward))


class EpsilonGreedyBandit:
    """
    Epsilon-Greedy with Neural Network
    Simple but effective
    """
    def __init__(self, env, learning_rate=1e-3):
        self.env = env
        self.network = ContextualBanditNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.history = []
        self.best_reward = -float('inf')
        self.best_action = None
        
    def select_action(self, context, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            # Random exploration (respecting constraints)
            delta = np.random.uniform(0, 1)
            B_min = -delta * self.env.S0 + 0.1
            B_max = 50 - delta * self.env.S0
            B = np.random.uniform(B_min, B_max)
            return delta, B
        else:
            # Exploit: use network
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            with torch.no_grad():
                delta, B = self.network(context_tensor)
            return delta.item(), B.item()
    
    def update(self, context, delta, B, reward):
        """Update network using good experiences"""
        self.history.append((delta, B, reward))
        
        # Keep best action
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_action = (delta, B)
        
        # Only update on good experiences
        if reward > -1.0:  # Only learn from decent hedges
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            target_delta = torch.FloatTensor([delta])
            target_B = torch.FloatTensor([B])
            
            # Forward
            pred_delta, pred_B = self.network(context_tensor)
            
            # MSE loss
            loss = (pred_delta - target_delta) ** 2 + ((pred_B - target_B) / 50) ** 2
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()


class ThompsonSamplingBandit:
    """
    Thompson Sampling for continuous actions
    UNBIASED: Uniform prior (maximum entropy)
    """
    def __init__(self):
        # Store all observed (action, reward) pairs
        self.observations = []
        
        # NO PRIOR BELIEF - we'll sample uniformly until we have data
        self.n_updates = 0
        
    def select_action(self, context):
        """Sample from posterior (or uniform if no data)"""
        if len(self.observations) < 10:
            # Pure uniform exploration at start (unbiased)
            delta = np.random.uniform(0, 1)
            B = np.random.uniform(-80, 50)
        else:
            # After collecting data, sample from empirical distribution
            # Weight by softmax of rewards
            deltas = [obs[0] for obs in self.observations]
            Bs = [obs[1] for obs in self.observations]
            rewards = np.array([obs[2] for obs in self.observations])
            
            # Softmax weights (higher reward = higher probability)
            # Use log-sum-exp trick for numerical stability
            rewards_shifted = rewards - rewards.max()  # Shift for stability
            exp_rewards = np.exp(rewards_shifted / 0.5)  # Temperature = 0.5
            
            # Handle potential numerical issues
            if np.any(np.isnan(exp_rewards)) or np.any(np.isinf(exp_rewards)):
                # Fallback to uniform if numerical issues
                weights = np.ones(len(self.observations)) / len(self.observations)
            else:
                weights = exp_rewards / exp_rewards.sum()
            
            # Sample from weighted distribution + noise for exploration
            idx = np.random.choice(len(self.observations), p=weights)
            delta = deltas[idx] + np.random.normal(0, 0.05)
            B = Bs[idx] + np.random.normal(0, 2.0)
            
            # Clip to valid range
            delta = np.clip(delta, 0, 1)
            B = np.clip(B, -80, 50)
        
        return delta, B
    
    def update(self, context, delta, B, reward):
        """Store observation (no parametric assumption)"""
        self.observations.append((delta, B, reward))
        self.n_updates += 1
        
        # Keep only recent observations to prevent memory explosion
        if len(self.observations) > 1000:
            # Remove worst 20%
            self.observations.sort(key=lambda x: x[2])
            self.observations = self.observations[200:]


def train_bandit(bandit, env, algorithm_name, n_rounds=5000, print_every=500):
    """Generic training loop for any bandit"""
    print(f"\n{'='*60}")
    print(f"Training: {algorithm_name}")
    print(f"{'='*60}\n")
    
    context = env.get_context()
    rewards_history = []
    
    # Epsilon decay for epsilon-greedy
    epsilon_start = 0.5
    epsilon_end = 0.05
    epsilon = epsilon_start
    epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / n_rounds)
    
    for round_num in range(n_rounds):
        # Select action
        if isinstance(bandit, EpsilonGreedyBandit):
            delta, B = bandit.select_action(context, epsilon=epsilon)
            epsilon *= epsilon_decay
        else:
            delta, B = bandit.select_action(context)
        
        # Get reward
        reward, pnl_up, pnl_down, hedge_price = env.evaluate_hedge(delta, B)
        
        # Update bandit
        bandit.update(context, delta, B, reward)
        
        rewards_history.append(reward)
        
        # Print progress
        if (round_num + 1) % print_every == 0:
            recent_reward = np.mean(rewards_history[-print_every:])
            print(f"Round {round_num + 1}/{n_rounds}")
            print(f"  Avg Reward: {recent_reward:.4f}")
            if isinstance(bandit, EpsilonGreedyBandit):
                print(f"  Epsilon: {epsilon:.4f}")
                if bandit.best_action:
                    print(f"  Best so far: δ={bandit.best_action[0]:.4f}, B={bandit.best_action[1]:.4f}")
    
    return bandit, rewards_history


def evaluate_bandit(bandit, env, algorithm_name, n_samples=1000):
    """Evaluate final performance"""
    context = env.get_context()
    
    best_reward = -float('inf')
    best_delta, best_B = None, None
    
    for _ in range(n_samples):
        if isinstance(bandit, EpsilonGreedyBandit):
            delta, B = bandit.select_action(context, epsilon=0)  # No exploration
        else:
            delta, B = bandit.select_action(context)
        
        reward, pnl_up, pnl_down, hedge_price = env.evaluate_hedge(delta, B)
        
        if reward > best_reward:
            best_reward = reward
            best_delta = delta
            best_B = B
            best_pnl_up = pnl_up
            best_pnl_down = pnl_down
            best_hedge_price = hedge_price
    
    print(f"\n{'='*60}")
    print(f"{algorithm_name} - FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best from {n_samples} samples:")
    print(f"  δ = {best_delta:.6f}")
    print(f"  B = {best_B:.6f}")
    print(f"  Hedge price: {best_hedge_price:.4f}")
    print(f"  P&L Up: {best_pnl_up:.6f}, Down: {best_pnl_down:.6f}")
    print(f"  Total error: {abs(best_pnl_up) + abs(best_pnl_down):.6f}")
    print(f"  Reward: {best_reward:.6f}")
    
    return best_delta, best_B, best_reward


if __name__ == "__main__":
    env = BinomialOptionEnvironment(
        S0=100, K=100, r=0.05, T=1.0, u=1.2, d=0.8
    )
    
    print("="*60)
    print("CONTEXTUAL BANDIT ALGORITHMS")
    print("="*60)
    print("Testing three bandit approaches:")
    print("  1. Epsilon-Greedy (neural network)")
    print("  2. Thompson Sampling (Bayesian)")
    print("  3. LinUCB (linear + UCB)")
    print("="*60)
    
    # Theoretical
    delta_theory = (env.Cu - env.Cd) / (env.Su - env.Sd)
    B_theory = np.exp(-env.r * env.T) * (env.u * env.Cd - env.d * env.Cu) / (env.u - env.d)
    print(f"\nTheoretical: δ={delta_theory:.6f}, B={B_theory:.6f}")
    
    # Test each algorithm
    results = {}
    
    # 1. Epsilon-Greedy
    bandit1 = EpsilonGreedyBandit(env, learning_rate=5e-3)
    train_bandit(bandit1, env, "Epsilon-Greedy", n_rounds=5000, print_every=500)
    delta1, B1, reward1 = evaluate_bandit(bandit1, env, "Epsilon-Greedy", n_samples=1000)
    results['Epsilon-Greedy'] = (delta1, B1, reward1)
    
    # 2. Thompson Sampling
    bandit2 = ThompsonSamplingBandit()
    train_bandit(bandit2, env, "Thompson Sampling", n_rounds=5000, print_every=500)
    delta2, B2, reward2 = evaluate_bandit(bandit2, env, "Thompson Sampling", n_samples=1000)
    results['Thompson Sampling'] = (delta2, B2, reward2)
    
    # 3. LinUCB
    bandit3 = LinUCBBandit(context_dim=2, alpha=0.5)
    train_bandit(bandit3, env, "LinUCB", n_rounds=5000, print_every=500)
    delta3, B3, reward3 = evaluate_bandit(bandit3, env, "LinUCB", n_samples=1000)
    results['LinUCB'] = (delta3, B3, reward3)
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON TO THEORY")
    print("="*60)
    for name, (delta, B, reward) in results.items():
        print(f"\n{name}:")
        print(f"  Δδ = {abs(delta - delta_theory):.6f}")
        print(f"  ΔB = {abs(B - B_theory):.6f}")
        print(f"  Reward = {reward:.6f}")
    
    # Best algorithm
    best_algo = max(results.items(), key=lambda x: x[1][2])
    print(f"\n🏆 Best Algorithm: {best_algo[0]} (Reward: {best_algo[1][2]:.6f})")