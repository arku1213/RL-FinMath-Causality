import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TrinomialOptionEnvironment:
    """One-step trinomial model for option pricing with super-replication"""
    def __init__(self, S0=100, K=100, r=0.05, T=1.0, u=1.2, d=0.8, m=1.0):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.u = u    # Up factor
        self.m = m    # Middle factor (usually 1.0 = no change)
        self.d = d    # Down factor
        
        # Stock prices at maturity
        self.Su = S0 * u
        self.Sm = S0 * m
        self.Sd = S0 * d
        
        # Option payoffs (call option)
        self.Cu = max(self.Su - K, 0)
        self.Cm = max(self.Sm - K, 0)
        self.Cd = max(self.Sd - K, 0)
        
        # Risk-neutral probabilities (assuming they sum to 1)
        # For trinomial, we need: pu + pm + pd = 1 and match first two moments
        # Simplified: equal probabilities for now (can be made more sophisticated)
        self.pu = 0.33
        self.pm = 0.34
        self.pd = 0.33
        
        # Theoretical option price (risk-neutral valuation)
        self.C0 = np.exp(-r * T) * (self.pu * self.Cu + self.pm * self.Cm + self.pd * self.Cd)
        
    def get_context(self):
        """Get context (static in this case)"""
        return np.array([self.S0, self.K], dtype=np.float32)
    
    def evaluate_super_replication(self, delta, B):
        """
        Evaluate super-replication hedge
        
        Super-replication: Portfolio value ≥ Option payoff in ALL scenarios
        
        Objective: Minimize hedge cost
        Subject to: V_T(ω) ≥ C_T(ω) for all ω ∈ {up, middle, down}
        
        Returns reward based on:
        1. How much hedge costs (want to minimize)
        2. Whether super-replication constraint is satisfied
        """
        
        # Hedge cost at t=0
        hedge_cost = delta * self.S0 + B
        
        # Portfolio values at maturity for each scenario
        V_up = delta * self.Su + B * np.exp(self.r * self.T)
        V_mid = delta * self.Sm + B * np.exp(self.r * self.T)
        V_down = delta * self.Sd + B * np.exp(self.r * self.T)
        
        # Super-replication gaps (positive = good, negative = violates constraint)
        gap_up = V_up - self.Cu
        gap_mid = V_mid - self.Cm
        gap_down = V_down - self.Cd
        
        # Check if super-replication is satisfied
        min_gap = min(gap_up, gap_mid, gap_down)
        
        # Reward function for super-replication:
        # We want to minimize cost while satisfying constraints
        
        if min_gap >= 0:
            # Constraint satisfied: reward = -cost + small bonus for tightness
            # Tighter super-replication (smaller gaps) is better
            avg_gap = (gap_up + gap_mid + gap_down) / 3.0
            reward = -hedge_cost - 0.1 * avg_gap  # Penalize both cost and excess coverage
        else:
            # Constraint violated: heavy penalty
            penalty = 1000 * (min_gap ** 2)  # Quadratic penalty for violation
            reward = -hedge_cost - penalty
        
        # Additional constraint penalties
        constraint_penalty = 0
        
        # Constraint: hedge cost should be positive (you pay for hedging)
        if hedge_cost <= 0:
            constraint_penalty += 1000 * ((hedge_cost - 1) ** 2)
        
        # Constraint: reasonable upper bound on cost
        if hedge_cost > 50:
            constraint_penalty += 100 * ((hedge_cost - 50) ** 2)
        
        reward -= constraint_penalty / 100.0
        
        return reward, hedge_cost, gap_up, gap_mid, gap_down, min_gap


class ThompsonSamplingBandit:
    """
    Thompson Sampling for continuous actions
    UNBIASED: Uniform prior (maximum entropy)
    """
    def __init__(self):
        # Store all observed (action, reward) pairs
        self.observations = []
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
        """Store observation (non-parametric)"""
        self.observations.append((delta, B, reward))
        self.n_updates += 1
        
        # Keep only recent observations to prevent memory explosion
        if len(self.observations) > 1000:
            # Remove worst 20%
            self.observations.sort(key=lambda x: x[2])
            self.observations = self.observations[200:]


def train_thompson_sampling(env, n_rounds=5000, print_every=500):
    """Train Thompson Sampling for super-replication"""
    print("="*60)
    print("TRINOMIAL SUPER-REPLICATION WITH THOMPSON SAMPLING")
    print("="*60)
    print(f"Stock: S0={env.S0}, Strike: K={env.K}")
    print(f"Moves: Up={env.Su}, Middle={env.Sm}, Down={env.Sd}")
    print(f"Payoffs: Cu={env.Cu}, Cm={env.Cm}, Cd={env.Cd}")
    print(f"Theoretical option price: {env.C0:.4f}")
    print("\nObjective: Find cheapest super-replicating portfolio")
    print("Constraint: Portfolio value ≥ Option payoff in ALL scenarios")
    print("="*60)
    print()
    
    bandit = ThompsonSamplingBandit()
    context = env.get_context()
    
    rewards_history = []
    costs_history = []
    best_feasible_cost = float('inf')
    best_feasible_hedge = None
    
    for round_num in range(n_rounds):
        # Select action
        delta, B = bandit.select_action(context)
        
        # Evaluate
        reward, cost, gap_up, gap_mid, gap_down, min_gap = env.evaluate_super_replication(delta, B)
        
        # Track best feasible solution (satisfies constraints)
        if min_gap >= 0 and cost < best_feasible_cost:
            best_feasible_cost = cost
            best_feasible_hedge = (delta, B, gap_up, gap_mid, gap_down)
        
        # Update bandit
        bandit.update(context, delta, B, reward)
        
        rewards_history.append(reward)
        costs_history.append(cost)
        
        # Print progress
        if (round_num + 1) % print_every == 0:
            recent_reward = np.mean(rewards_history[-print_every:])
            recent_cost = np.mean(costs_history[-print_every:])
            
            print(f"Round {round_num + 1}/{n_rounds}")
            print(f"  Avg Reward: {recent_reward:.4f}")
            print(f"  Avg Cost: {recent_cost:.4f}")
            if best_feasible_hedge:
                d, b, gu, gm, gd = best_feasible_hedge
                print(f"  Best feasible: δ={d:.4f}, B={b:.4f}, Cost={best_feasible_cost:.4f}")
                print(f"    Gaps: Up={gu:.4f}, Mid={gm:.4f}, Down={gd:.4f}")
            else:
                print(f"  No feasible solution found yet")
    
    return bandit, best_feasible_hedge, best_feasible_cost


def evaluate_final_solution(bandit, env, n_samples=2000):
    """Find best super-replicating hedge from learned policy"""
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    context = env.get_context()
    
    best_feasible_cost = float('inf')
    best_solution = None
    all_feasible = []
    
    for _ in range(n_samples):
        delta, B = bandit.select_action(context)
        reward, cost, gap_up, gap_mid, gap_down, min_gap = env.evaluate_super_replication(delta, B)
        
        # Check if feasible
        if min_gap >= 0:
            all_feasible.append((delta, B, cost, gap_up, gap_mid, gap_down))
            if cost < best_feasible_cost:
                best_feasible_cost = cost
                best_solution = (delta, B, cost, gap_up, gap_mid, gap_down)
    
    if best_solution:
        delta, B, cost, gap_up, gap_mid, gap_down = best_solution
        
        print(f"\nFound {len(all_feasible)} feasible solutions from {n_samples} samples")
        print(f"\n*** BEST SUPER-REPLICATING HEDGE ***")
        print(f"  δ = {delta:.6f}")
        print(f"  B = {B:.6f}")
        print(f"  Hedge cost: {cost:.4f}")
        print(f"\nSuper-replication verification:")
        print(f"  Up scenario:   Portfolio={delta * env.Su + B * np.exp(env.r * env.T):.4f} ≥ Payoff={env.Cu:.4f} ✓ (gap={gap_up:.4f})")
        print(f"  Mid scenario:  Portfolio={delta * env.Sm + B * np.exp(env.r * env.T):.4f} ≥ Payoff={env.Cm:.4f} ✓ (gap={gap_mid:.4f})")
        print(f"  Down scenario: Portfolio={delta * env.Sd + B * np.exp(env.r * env.T):.4f} ≥ Payoff={env.Cd:.4f} ✓ (gap={gap_down:.4f})")
        print(f"\n  All constraints satisfied: {min(gap_up, gap_mid, gap_down) >= 0}")
        print(f"  Cost vs theoretical option price: {cost:.4f} vs {env.C0:.4f}")
        print(f"  Super-replication premium: {cost - env.C0:.4f} ({(cost/env.C0 - 1)*100:.2f}%)")
    else:
        print("\nNo feasible super-replicating hedge found!")
        print("This may indicate the problem is infeasible or needs more exploration.")
    
    return best_solution


if __name__ == "__main__":
    # Create trinomial environment
    env = TrinomialOptionEnvironment(
        S0=100,   # Initial stock price
        K=100,    # Strike price
        r=0.05,   # Risk-free rate
        T=1.0,    # Time to maturity
        u=1.2,    # Up factor
        m=1.0,    # Middle factor (no change)
        d=0.8     # Down factor
    )
    
    # Train Thompson Sampling
    bandit, best_training, best_cost_training = train_thompson_sampling(
        env, 
        n_rounds=20000, 
        print_every=1000
    )
    
    # Final evaluation
    best_solution = evaluate_final_solution(bandit, env, n_samples=2000)
    
    print("\n" + "="*60)
    print("KEY INSIGHT: Super-Replication")
    print("="*60)
    print("Unlike perfect replication (P&L=0), super-replication:")
    print("  • Portfolio dominates option in ALL states")
    print("  • Costs more than theoretical fair value")
    print("  • Provides safety margin against model error")
    print("  • Finds cheapest dominating portfolio")
    print("="*60)