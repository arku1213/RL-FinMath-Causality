import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations
from scipy.optimize import minimize

def calculate_risk_neutral_probabilities(factors, r, T):
    """
    Calculate risk-neutral probabilities for given factors using optimization
    
    Constraints:
    1. Sum to 1: Σ pᵢ = 1
    2. Match expected return: Σ pᵢ × factorsᵢ = e^(rT)
    3. Non-negative: All pᵢ ≥ 0
    
    Objective: Minimize deviation from uniform distribution (maximum entropy)
    """
    factors = np.array(factors)
    n = len(factors)
    target_return = np.exp(r * T)
    
    def objective(probs):
        # Minimize deviation from uniform probabilities (maximum entropy)
        uniform_prob = 1.0 / n
        return np.sum((probs - uniform_prob)**2)
    
    def constraint_sum(probs):
        return np.sum(probs) - 1.0
    
    def constraint_return(probs):
        return np.sum(probs * factors) - target_return
    
    # Starting guess: uniform probabilities
    x0 = np.ones(n) / n
    
    constraints = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'eq', 'fun': constraint_return}
    ]
    
    bounds = [(0, 1) for _ in range(n)]
    
    result = minimize(objective, x0, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        print(f"Warning: Optimization failed. Using uniform probabilities.")
        return np.ones(n) / n

class NnomialOptionEnvironment:
    """Generalized N-nomial model for option pricing with super-replication"""
    def __init__(self, S0, K, r, T, factors, probabilities=None):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        
        # Convert factors to numpy array
        self.factors = np.array(factors)
        self.N = len(self.factors)
        
        # Calculate risk-neutral probabilities if not provided
        if probabilities is None:
            print("Calculating risk-neutral probabilities...")
            self.probabilities = calculate_risk_neutral_probabilities(factors, r, T)
            print(f"Risk-neutral probabilities: {self.probabilities}")
            print(f"Expected return check: {np.sum(self.probabilities * self.factors):.6f} vs target {np.exp(r*T):.6f}")
        else:
            self.probabilities = np.array(probabilities)
        
        # Validate inputs
        assert len(self.probabilities) == self.N, "Probabilities must match number of factors"
        assert abs(sum(self.probabilities) - 1.0) < 1e-10, "Probabilities must sum to 1"
        assert self.N >= 2, "Must have at least 2 scenarios"
        
        # Stock prices at maturity (vectorized)
        self.S_T = self.S0 * self.factors
        
        # Option payoffs (vectorized) - call option
        self.C_T = np.maximum(self.S_T - self.K, 0)
        
        # Theoretical option price (risk-neutral valuation)
        self.C0 = np.exp(-r * T) * np.sum(self.probabilities * self.C_T)
        
        # Store discount factor
        self.erT = np.exp(self.r * self.T)
        
    def get_context(self):
        """Get context (static in this case)"""
        return np.array([self.S0, self.K], dtype=np.float32)
    
    def evaluate_super_replication(self, delta, B):
        """
        Evaluate super-replication hedge for N scenarios
        
        Super-replication: Portfolio value ≥ Option payoff in ALL scenarios
        
        Objective: Minimize hedge cost
        Subject to: V_T(ω) ≥ C_T(ω) for all ω ∈ {1, 2, ..., N}
        
        Returns reward based on:
        1. How much hedge costs (want to minimize)
        2. Whether super-replication constraint is satisfied
        """
        
        # Hedge cost at t=0
        hedge_cost = delta * self.S0 + B
        
        # Portfolio values at maturity for ALL N scenarios (vectorized)
        V_T_scenarios = delta * self.S_T + B * self.erT
        
        # Super-replication gaps (vectorized)
        gaps = V_T_scenarios - self.C_T
        
        # Check if super-replication is satisfied across ALL scenarios
        min_gap = np.min(gaps)
        
        # Reward function for super-replication:
        # We want to minimize cost while satisfying constraints
        
        if min_gap >= 0:
            # Constraint satisfied: reward = -cost + small bonus for tightness
            # Tighter super-replication (smaller gaps) is better
            avg_gap = np.mean(gaps)
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
        
        return reward, hedge_cost, gaps, min_gap


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


def find_analytical_super_replication_price(env):
    """
    Find the analytical super-replication price by solving the active constraints
    
    For N scenarios, we have N constraints but only 2 variables (delta, B).
    The optimal solution lies at the intersection of exactly 2 active constraints.
    
    Strategy:
    1. Try all pairs of constraints
    2. Solve the 2x2 system for each pair
    3. Check if the solution satisfies all N constraints
    4. Return the feasible solution with minimum cost
    """
    
    best_cost = float('inf')
    best_solution = None
    
    # Try all pairs of constraints (scenarios)
    for i, j in combinations(range(env.N), 2):
        try:
            # Set up 2x2 system for scenarios i and j
            # delta * S_T[i] + B * erT = C_T[i]
            # delta * S_T[j] + B * erT = C_T[j]
            
            A = np.array([
                [env.S_T[i], env.erT],
                [env.S_T[j], env.erT]
            ])
            b = np.array([env.C_T[i], env.C_T[j]])
            
            # Solve the system
            delta, B = np.linalg.solve(A, b)
            
            # Calculate cost
            cost = delta * env.S0 + B
            
            # Check if this solution satisfies ALL N constraints
            V_T_all = delta * env.S_T + B * env.erT
            gaps_all = V_T_all - env.C_T
            
            # Must satisfy all constraints (with small numerical tolerance)
            if np.all(gaps_all >= -1e-10) and cost > 0:
                if cost < best_cost:
                    best_cost = cost
                    best_solution = {
                        'delta': delta,
                        'B': B, 
                        'cost': cost,
                        'active_constraints': [i, j],
                        'gaps': gaps_all,
                        'min_gap': np.min(gaps_all),
                        'portfolio_values': V_T_all
                    }
                    
        except np.linalg.LinAlgError:
            # Skip if system is singular (parallel constraints)
            continue
    
    return best_solution


def train_thompson_sampling(env, n_rounds=5000, print_every=500):
    """Train Thompson Sampling for super-replication"""
    print("="*60)
    print(f"{env.N}-NOMIAL SUPER-REPLICATION WITH THOMPSON SAMPLING")
    print("="*60)
    print(f"Stock: S0={env.S0}, Strike: K={env.K}")
    print(f"Factors: {env.factors}")
    print(f"Stock prices at maturity: {env.S_T}")
    print(f"Option payoffs: {env.C_T}")
    print(f"Probabilities: {env.probabilities}")
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
        reward, cost, gaps, min_gap = env.evaluate_super_replication(delta, B)
        
        # Track best feasible solution (satisfies constraints)
        if min_gap >= 0 and cost < best_feasible_cost:
            best_feasible_cost = cost
            best_feasible_hedge = (delta, B, gaps, min_gap)
        
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
                d, b, gaps, min_gap = best_feasible_hedge
                print(f"  Best feasible: δ={d:.4f}, B={b:.4f}, Cost={best_feasible_cost:.4f}")
                print(f"    Min gap: {min_gap:.4f}, Gaps: {gaps}")
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
        reward, cost, gaps, min_gap = env.evaluate_super_replication(delta, B)
        
        # Check if feasible
        if min_gap >= 0:
            all_feasible.append((delta, B, cost, gaps, min_gap))
            if cost < best_feasible_cost:
                best_feasible_cost = cost
                best_solution = (delta, B, cost, gaps, min_gap)
    
    if best_solution:
        delta, B, cost, gaps, min_gap = best_solution
        
        print(f"\nFound {len(all_feasible)} feasible solutions from {n_samples} samples")
        print(f"\n*** BEST SUPER-REPLICATING HEDGE ***")
        print(f"  δ = {delta:.6f}")
        print(f"  B = {B:.6f}")
        print(f"  Hedge cost: {cost:.4f}")
        print(f"\nSuper-replication verification:")
        portfolio_values = delta * env.S_T + B * env.erT
        for i in range(env.N):
            status = "✓" if gaps[i] >= 0 else "❌"
            print(f"  Scenario {i+1}: Portfolio={portfolio_values[i]:.4f} ≥ Payoff={env.C_T[i]:.4f} {status} (gap={gaps[i]:.4f})")
        
        print(f"\n  All constraints satisfied: {min_gap >= 0}")
        print(f"  Cost vs theoretical option price: {cost:.4f} vs {env.C0:.4f}")
        print(f"  Super-replication premium: {cost - env.C0:.4f} ({(cost/env.C0 - 1)*100:.2f}%)")
    else:
        print("\nNo feasible super-replicating hedge found!")
        print("This may indicate the problem is infeasible or needs more exploration.")
    
    return best_solution


if __name__ == "__main__":
    # Example: Trinomial model
    env = NnomialOptionEnvironment(
        S0=100, K=100, r=0.05, T=1.0,
        factors=[1.2, 1.0, 0.8],
        probabilities=[0.33, 0.34, 0.33]
    )
    
    # Train Thompson Sampling
    bandit, best_training, best_cost_training = train_thompson_sampling(
        env, n_rounds=10000, print_every=1000
    )
    
    # Final evaluation
    best_solution = evaluate_final_solution(bandit, env, n_samples=2000)
    
    # Calculate analytical optimum
    analytical = find_analytical_super_replication_price(env)
    
    if analytical:
        print(f"\n" + "="*60)
        print("ANALYTICAL OPTIMUM")
        print("="*60)
        print(f"  δ* = {analytical['delta']:.6f}")
        print(f"  B* = {analytical['B']:.6f}")
        print(f"  Cost* = {analytical['cost']:.4f}")
        print(f"  Active constraints: {analytical['active_constraints']}")
        print(f"  Min gap: {analytical['min_gap']:.6f}")
        
        if best_solution:
            delta, B, cost, gaps, min_gap = best_solution
            print(f"\n" + "="*60)
            print("THOMPSON SAMPLING vs ANALYTICAL COMPARISON")
            print("="*60)
            print(f"  Thompson Sampling: δ={delta:.6f}, B={B:.6f}, Cost={cost:.4f}")
            print(f"  Analytical Optimum: δ={analytical['delta']:.6f}, B={analytical['B']:.6f}, Cost={analytical['cost']:.4f}")
            print(f"  Error: δ={abs(delta-analytical['delta'])/abs(analytical['delta'])*100:.2f}%, B={abs(B-analytical['B'])/abs(analytical['B'])*100:.2f}%, Cost={abs(cost-analytical['cost'])/analytical['cost']*100:.2f}%")
    
