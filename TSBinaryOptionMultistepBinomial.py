"""
Thompson Sampling for Multi-Step Binomial Option Hedging with Binary Options
Adapted from trinomial version for T-step binomial trees
"""

import numpy as np
from itertools import product


def calculate_binomial_option_price(S0, K, r, T, u, d):
    """Calculate theoretical binomial option price using Monte Carlo (risk-neutral expectation)"""
    dt = 1.0 / T
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
    q = 1 - p
    
    # Generate all terminal states
    expected_payoff = 0.0
    for n_ups in range(T + 1):
        n_downs = T - n_ups
        
        # Terminal stock price
        S_T = S0 * (u ** n_ups) * (d ** n_downs)
        
        # Option payoff
        payoff = max(S_T - K, 0)
        
        # Binomial probability of this path
        from math import comb
        prob = comb(T, n_ups) * (p ** n_ups) * (q ** n_downs)
        
        expected_payoff += prob * payoff
    
    # Discount back to present
    return np.exp(-r * T) * expected_payoff


class MultiStepBinomialEnvironment:
    """
    Multi-step binomial model for option hedging with binary options
    
    For T steps, there are 2^T possible terminal states.
    We need to find a hedge that super-replicates across ALL paths.
    """
    
    def __init__(self, S0, K, r, T, u, d):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.dt = 1.0 / T
        
        self.u = u
        self.d = d
        
        # Risk-neutral probability
        self.p = (np.exp(r * self.dt) - d) / (u - d)
        self.q = 1 - self.p
        
        print(f"Risk-neutral probabilities: p_up={self.p:.6f}, p_down={self.q:.6f}")
        
        # Binary option prices (discounted probabilities)
        self.binary_up_price = np.exp(-r * self.dt) * self.p
        self.binary_down_price = np.exp(-r * self.dt) * self.q
        
        print(f"Binary option prices: up={self.binary_up_price:.6f}, down={self.binary_down_price:.6f}")
        
        # Generate all possible paths and terminal states
        self.paths = list(product([0, 1], repeat=T))  # 0=down, 1=up
        self.n_paths = len(self.paths)
        
        print(f"Number of paths: {self.n_paths}")
        
        # Calculate terminal prices and payoffs for each path
        self.S_T = np.zeros(self.n_paths)
        self.C_T = np.zeros(self.n_paths)
        self.path_probs = np.zeros(self.n_paths)
        
        for i, path in enumerate(self.paths):
            n_ups = sum(path)
            n_downs = T - n_ups
            
            # Terminal stock price
            self.S_T[i] = S0 * (u ** n_ups) * (d ** n_downs)
            
            # Option payoff
            self.C_T[i] = max(self.S_T[i] - K, 0)
            
            # Path probability
            self.path_probs[i] = (self.p ** n_ups) * (self.q ** n_downs)
        
        # Theoretical option price
        self.C0_theoretical = calculate_binomial_option_price(S0, K, r, T, u, d)
        self.C0_monte_carlo = np.exp(-r * T) * np.sum(self.path_probs * self.C_T)
        
        print(f"Theoretical option price: {self.C0_theoretical:.6f}")
        print(f"Monte Carlo check: {self.C0_monte_carlo:.6f}")
        
        # Discount factor
        self.discount_factor = np.exp(-r * T)
    
    def evaluate_hedge(self, delta, B, b_up, b_down):
        """
        Evaluate a static hedge using stock, bond, and binary options.
        
        The hedge is set at t=0 and held to maturity.
        Binary options pay at each step along the path.
        
        Args:
            delta: shares of stock
            B: bond position
            b_up: units of up-binary (pays 1 for each up move)
            b_down: units of down-binary (pays 1 for each down move)
        """
        
        # Initial hedge cost
        hedge_cost = (delta * self.S0 + B + 
                     b_up * self.binary_up_price * self.T +  # Can use up-binary at each step
                     b_down * self.binary_down_price * self.T)  # Can use down-binary at each step
        
        # Calculate portfolio value at maturity for each path
        portfolio_values = np.zeros(self.n_paths)
        
        for i, path in enumerate(self.paths):
            n_ups = sum(path)
            n_downs = self.T - n_ups
            
            # Portfolio at maturity:
            # - Stock position: delta * S_T
            # - Bond: B * e^(rT)
            # - Binary payoffs: b_up pays for each up move, b_down for each down
            portfolio_values[i] = (delta * self.S_T[i] + 
                                  B * np.exp(self.r * self.T) + 
                                  b_up * n_ups + 
                                  b_down * n_downs)
        
        # Super-replication gaps (portfolio - option payoff)
        gaps = portfolio_values - self.C_T
        min_gap = np.min(gaps)
        
        # Check constraint satisfaction
        is_feasible = min_gap >= -1e-6  # Allow tiny numerical error
        cost_positive = hedge_cost > 1e-6
        
        # Reward function
        if not cost_positive:
            # Negative cost = arbitrage, heavily penalize
            reward = -1e6 + hedge_cost * 1e3
        elif not is_feasible:
            # Constraint violation
            penalty = 1000 * (min_gap ** 2)
            reward = -hedge_cost - penalty
        else:
            # Feasible solution: minimize cost + excess
            # Penalize excess more heavily to encourage tighter hedges
            avg_gap = np.mean(gaps)
            max_gap = np.max(gaps)
            reward = -hedge_cost - 1.0 * avg_gap - 0.5 * max_gap
        
        return reward, hedge_cost, gaps, min_gap, is_feasible


class ThompsonSamplingBandit:
    """Thompson Sampling for continuous actions in hedging"""
    
    def __init__(self, T):
        self.T = T
        self.observations = []
        self.n_updates = 0
        
    def select_action(self):
        """Sample action from posterior (or uniform initially)"""
        if len(self.observations) < 100:  # Initial exploration
            delta = np.random.uniform(0, 1.5)
            B = np.random.uniform(-80, 50)
            b_up = np.random.uniform(-50, 50)
            b_down = np.random.uniform(-50, 50)
        else:
            # Sample from weighted distribution of past observations
            deltas = [obs[0] for obs in self.observations]
            Bs = [obs[1] for obs in self.observations]
            b_ups = [obs[2] for obs in self.observations]
            b_downs = [obs[3] for obs in self.observations]
            rewards = np.array([obs[4] for obs in self.observations])
            
            # Softmax weights
            rewards_shifted = rewards - rewards.max()
            exp_rewards = np.exp(rewards_shifted / 0.5)
            
            if np.any(np.isnan(exp_rewards)) or np.any(np.isinf(exp_rewards)):
                weights = np.ones(len(self.observations)) / len(self.observations)
            else:
                weights = exp_rewards / exp_rewards.sum()
            
            # Sample and add noise
            idx = np.random.choice(len(self.observations), p=weights)
            delta = deltas[idx] + np.random.normal(0, 0.05)
            B = Bs[idx] + np.random.normal(0, 2.0)
            b_up = b_ups[idx] + np.random.normal(0, 2.0)
            b_down = b_downs[idx] + np.random.normal(0, 2.0)
            
            # Clip to valid range
            delta = np.clip(delta, 0, 1.5)
            B = np.clip(B, -80, 50)
            b_up = np.clip(b_up, -50, 50)
            b_down = np.clip(b_down, -50, 50)
        
        return delta, B, b_up, b_down
    
    def update(self, delta, B, b_up, b_down, reward):
        """Store observation"""
        self.observations.append((delta, B, b_up, b_down, reward))
        self.n_updates += 1
        
        # Keep only best 1000 observations to prevent memory explosion
        if len(self.observations) > 1000:
            self.observations.sort(key=lambda x: x[4])  # Sort by reward
            self.observations = self.observations[200:]  # Keep top 80%


def train_thompson_sampling(env, n_rounds=10000, print_every=1000):
    """Train Thompson Sampling for binomial hedging"""
    print("\n" + "="*80)
    print("MULTI-STEP BINOMIAL HEDGING WITH BINARY OPTIONS")
    print("="*80)
    print(f"Parameters: S0={env.S0}, K={env.K}, r={env.r}, T={env.T} steps")
    print(f"Up factor: {env.u}, Down factor: {env.d}")
    print(f"Number of terminal states: {env.n_paths}")
    print(f"Theoretical option price: {env.C0_theoretical:.6f}")
    print("\nInstruments: Stock (δ), Bond (B), Binary-Up (b_up), Binary-Down (b_down)")
    print("Objective: Find cheapest super-replicating hedge")
    print("Constraint: Portfolio ≥ Option payoff across ALL paths")
    print("="*80 + "\n")
    
    bandit = ThompsonSamplingBandit(env.T)
    
    rewards_history = []
    costs_history = []
    feasible_count = 0
    best_feasible_cost = float('inf')
    best_feasible_hedge = None
    
    for round_num in range(n_rounds):
        # Select action
        delta, B, b_up, b_down = bandit.select_action()
        
        # Evaluate
        reward, cost, gaps, min_gap, is_feasible = env.evaluate_hedge(delta, B, b_up, b_down)
        
        # Track best feasible solution
        if is_feasible:
            feasible_count += 1
            if cost < best_feasible_cost:
                best_feasible_cost = cost
                best_feasible_hedge = (delta, B, b_up, b_down, gaps, min_gap)
        
        # Update bandit
        bandit.update(delta, B, b_up, b_down, reward)
        
        rewards_history.append(reward)
        costs_history.append(cost)
        
        # Print progress
        if (round_num + 1) % print_every == 0:
            recent_reward = np.mean(rewards_history[-print_every:])
            recent_cost = np.mean(costs_history[-print_every:])
            feasible_rate = feasible_count / (round_num + 1) * 100
            
            print(f"Round {round_num + 1}/{n_rounds}")
            print(f"  Avg Reward: {recent_reward:.4f}")
            print(f"  Avg Cost: {recent_cost:.4f}")
            print(f"  Feasible rate: {feasible_rate:.1f}%")
            
            if best_feasible_hedge:
                d, b, bu, bd, _, mg = best_feasible_hedge
                print(f"  Best feasible hedge:")
                print(f"    δ={d:.4f}, B={b:.4f}, b_up={bu:.4f}, b_down={bd:.4f}")
                print(f"    Cost={best_feasible_cost:.4f} (vs theoretical {env.C0_theoretical:.4f})")
                print(f"    Min gap={mg:.6f}")
            else:
                print(f"  No feasible solution found yet")
            print()
    
    return bandit, best_feasible_hedge, best_feasible_cost


def evaluate_final_solution(bandit, env, n_samples=5000):
    """Find best hedge from learned policy"""
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80 + "\n")
    
    best_feasible_cost = float('inf')
    best_solution = None
    all_feasible = []
    
    for _ in range(n_samples):
        delta, B, b_up, b_down = bandit.select_action()
        reward, cost, gaps, min_gap, is_feasible = env.evaluate_hedge(delta, B, b_up, b_down)
        
        if is_feasible:
            all_feasible.append((delta, B, b_up, b_down, cost, gaps, min_gap))
            if cost < best_feasible_cost:
                best_feasible_cost = cost
                best_solution = (delta, B, b_up, b_down, cost, gaps, min_gap)
    
    if best_solution:
        delta, B, b_up, b_down, cost, gaps, min_gap = best_solution
        
        print(f"Found {len(all_feasible)} feasible solutions from {n_samples} samples\n")
        print("*** BEST SUPER-REPLICATING HEDGE ***")
        print(f"  δ (stock)     = {delta:.6f}")
        print(f"  B (bond)      = {B:.6f}")
        print(f"  b_up (binary) = {b_up:.6f}")
        print(f"  b_down (bin.) = {b_down:.6f}")
        print(f"\n  Hedge cost: {cost:.6f}")
        
        # Cost breakdown
        stock_cost = delta * env.S0
        bond_cost = B
        binary_cost = (b_up * env.binary_up_price * env.T + 
                      b_down * env.binary_down_price * env.T)
        
        print(f"\n  Cost breakdown:")
        print(f"    Stock:    {stock_cost:.6f}")
        print(f"    Bond:     {bond_cost:.6f}")
        print(f"    Binaries: {binary_cost:.6f}")
        
        # Verification across all paths
        print(f"\n  Super-replication verification (showing first 5 paths):")
        for i in range(min(5, env.n_paths)):
            path_str = ''.join(['U' if x else 'D' for x in env.paths[i]])
            n_ups = sum(env.paths[i])
            n_downs = env.T - n_ups
            
            portfolio_val = (delta * env.S_T[i] + 
                           B * np.exp(env.r * env.T) + 
                           b_up * n_ups + 
                           b_down * n_downs)
            
            status = "✓" if gaps[i] >= -1e-6 else "✗"
            print(f"    Path {path_str}: Portfolio={portfolio_val:.4f} ≥ Payoff={env.C_T[i]:.4f} {status} (gap={gaps[i]:.4f})")
        
        if env.n_paths > 5:
            print(f"    ... ({env.n_paths - 5} more paths)")
        
        print(f"\n  Min gap across all paths: {min_gap:.6f}")
        print(f"  Mean gap: {np.mean(gaps):.6f}")
        print(f"  Max gap: {np.max(gaps):.6f}")
        
        print(f"\n  Theoretical option price: {env.C0_theoretical:.6f}")
        print(f"  Your hedge cost:          {cost:.6f}")
        print(f"  Super-replication premium: {cost - env.C0_theoretical:.6f} ({(cost/env.C0_theoretical - 1)*100:.2f}%)")
        
        print("\n" + "="*80)
    else:
        print("No feasible solution found!")
        print("Try increasing n_rounds or adjusting action bounds.")
    
    return best_solution


if __name__ == "__main__":
    # T=2 Binomial model (your original problem)
    env = MultiStepBinomialEnvironment(
        S0=100.0,
        K=100.0,
        r=0.05,
        T=2,
        u=1.2,
        d=0.8
    )
    
    # Train Thompson Sampling
    print("Training Thompson Sampling...")
    bandit, best_training, best_cost_training = train_thompson_sampling(
        env, 
        n_rounds=20000,  # More rounds for better convergence
        print_every=2000
    )
    
    # Final evaluation
    best_solution = evaluate_final_solution(bandit, env, n_samples=5000)