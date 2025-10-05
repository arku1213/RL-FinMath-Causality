import numpy as np
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
        uniform_prob = 1.0 / n
        return np.sum((probs - uniform_prob)**2)
    
    def constraint_sum(probs):
        return np.sum(probs) - 1.0
    
    def constraint_return(probs):
        return np.sum(probs * factors) - target_return
    
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


class NnomialBinaryOptionEnvironment:
    """
    N-nomial model with ONLY binary options (NO BOND) for replication
    
    Trading instruments:
    - Binary_i: b_i (number of binary options paying 1 if scenario i occurs)
    
    Complete market: N scenarios, N instruments (b1, b2, ..., bN)
    EXACTLY COMPLETE - unique solution exists
    
    NO BIAS: Uniform prior, no hardcoded preferences
    """
    def __init__(self, S0, K, r, T, factors):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        
        # Convert factors to numpy array
        self.factors = np.array(factors)
        self.N = len(self.factors)
        
        print(f"Initializing {self.N}-nomial model...")
        
        # Calculate risk-neutral probabilities
        print("Calculating risk-neutral probabilities...")
        self.probabilities = calculate_risk_neutral_probabilities(factors, r, T)
        print(f"Risk-neutral probabilities: {self.probabilities}")
        print(f"Expected return check: {np.sum(self.probabilities * self.factors):.6f} vs target {np.exp(r*T):.6f}")
        
        # Stock prices at maturity (for reference only - we don't trade stock)
        self.S_T = self.S0 * self.factors
        
        # Option payoffs - call option
        self.C_T = np.maximum(self.S_T - self.K, 0)
        
        # Theoretical option price (risk-neutral valuation)
        self.C0 = np.exp(-r * T) * np.sum(self.probabilities * self.C_T)
        
        # Binary option prices = discounted risk-neutral probabilities
        # Price to buy a binary that pays 1 in scenario i
        self.binary_prices = np.exp(-r * T) * self.probabilities
        
        # Analytical solution (UNIQUE in exactly complete market)
        self.b_target = self.C_T.copy()
        
        print(f"\nEnvironment Setup:")
        print(f"  S0 = {self.S0}, K = {self.K}, r = {self.r}, T = {self.T}")
        print(f"  Number of scenarios: {self.N}")
        print(f"  Factors: {self.factors}")
        print(f"  Stock prices at maturity: {self.S_T}")
        print(f"  Option payoffs: {self.C_T}")
        print(f"  Theoretical option price: {self.C0:.4f}")
        print(f"  Binary prices: {self.binary_prices}")
        print(f"  Unique analytical solution: b = {self.b_target}")
    
    def get_context(self):
        """Get context for bandit (static in this case)"""
        return np.array([self.S0, self.K], dtype=np.float32)
    
    def evaluate_replication(self, b_vector):
        """
        Evaluate how well a given b_vector replicates the option
        
        For EXACT replication in complete market:
        - Portfolio value = Option payoff in ALL scenarios
        - No bond, only binaries
        - Unique solution: b_i = C_i
        
        Args:
            b_vector: numpy array of shape (N,) with binary positions
        
        Returns:
            reward: Negative of squared replication error + cost penalty
            cost: Initial cost of the portfolio
            errors: Replication errors in each scenario
        """
        
        # Ensure b_vector is numpy array
        b_vector = np.array(b_vector)
        
        # Initial cost (no bond!)
        cost = np.sum(b_vector * self.binary_prices)
        
        # Portfolio values at maturity (no bond term!)
        # In scenario i, only binary_i pays 1, all others pay 0
        portfolio_values = b_vector.copy()
        
        # Replication errors (should be zero for perfect replication)
        errors = portfolio_values - self.C_T
        
        # Mean squared error (HEAVILY penalized)
        mse = np.mean(errors ** 2)
        
        # Max absolute error (also penalize worst-case)
        max_error = np.max(np.abs(errors))
        
        # Reward function for EXACT replication:
        # Heavily penalize any replication error
        replication_penalty = mse * 10000 + max_error * 5000
        
        # Small penalty for cost deviation from theoretical
        cost_deviation = abs(cost - self.C0) * 0.1
        
        reward = -(replication_penalty + cost_deviation)
        
        return reward, cost, errors, portfolio_values


class ThompsonSamplingBandit:
    """
    Thompson Sampling for continuous actions in N-nomial case
    Learning to find exact replication with binary options only (no bond)
    
    NO BIAS: Uniform exploration, no hardcoded preferences
    Uses adaptive exploration noise that decreases over time
    Scales initial exploration with dimensionality
    """
    def __init__(self, n_binaries):
        # Store all observed (action, reward) pairs
        self.observations = []
        self.n_updates = 0
        self.n_binaries = n_binaries
        # Scale initial exploration more aggressively for higher dimensions
        # Use n^2 scaling for curse of dimensionality
        self.initial_exploration = max(100, 30 * n_binaries)
        
    def select_action(self, context, exploration_factor=1.0):
        """
        Sample from posterior (or uniform if no data) - NO BIAS
        exploration_factor: scales noise (1.0 = full noise, 0.0 = no noise)
        
        Returns:
            b_vector: numpy array of shape (n_binaries,)
        """
        if len(self.observations) < self.initial_exploration:
            # Pure uniform exploration at start - NO BIAS
            # More exploration for higher dimensions
            b_vector = np.random.uniform(-10, 60, size=self.n_binaries)
        else:
            # After collecting data, sample from empirical distribution
            # Weight by softmax of rewards
            b_vectors = np.array([obs[0] for obs in self.observations])
            rewards = np.array([obs[1] for obs in self.observations])
            
            # Softmax weights (higher reward = higher probability)
            # Use lower temperature for more exploitation
            rewards_shifted = rewards - rewards.max()
            exp_rewards = np.exp(rewards_shifted / 0.05)
            
            # Handle potential numerical issues
            if np.any(np.isnan(exp_rewards)) or np.any(np.isinf(exp_rewards)):
                weights = np.ones(len(self.observations)) / len(self.observations)
            else:
                weights = exp_rewards / exp_rewards.sum()
            
            # Sample from weighted distribution + ADAPTIVE noise for exploration
            idx = np.random.choice(len(self.observations), p=weights)
            
            # Adaptive noise that decreases over time
            # Scale noise with dimensionality to maintain exploration in high dimensions
            noise_b = exploration_factor * 0.3  # Increased from 0.2
            
            b_vector = b_vectors[idx] + np.random.normal(0, noise_b, size=self.n_binaries)
            
            # Clip to reasonable range
            b_vector = np.clip(b_vector, -10, 60)
        
        return b_vector
    
    def update(self, context, b_vector, reward):
        """Store observation (non-parametric) - NO BIAS"""
        self.observations.append((b_vector.copy(), reward))
        self.n_updates += 1
        
        # Keep only best observations to focus on good regions
        if len(self.observations) > 2000:
            # Keep top 80% by reward
            self.observations.sort(key=lambda x: x[1], reverse=True)
            self.observations = self.observations[:1600]


def train_thompson_sampling(env, n_rounds=20000, print_every=2000):
    """Train Thompson Sampling for exact replication with adaptive exploration"""
    print("\n" + "="*60)
    print(f"THOMPSON SAMPLING FOR {env.N}-NOMIAL REPLICATION")
    print("="*60)
    print("Objective: Find exact replication using ONLY binaries (no bond)")
    print(f"Instruments: Binary_1, Binary_2, ..., Binary_{env.N}")
    print("Goal: Portfolio value = Option payoff in ALL scenarios")
    print("Unique solution: b_i = C_i for all i")
    print("Adaptive exploration: Noise decreases over time for convergence")
    print("="*60)
    print()
    
    bandit = ThompsonSamplingBandit(n_binaries=env.N)
    context = env.get_context()
    
    rewards_history = []
    costs_history = []
    errors_history = []
    best_solution = None
    best_error = float('inf')
    
    for round_num in range(n_rounds):
        # Adaptive exploration: decrease noise over time
        exploration_factor = max(0.1, 1.0 - (round_num / n_rounds))
        
        # Select action with adaptive noise
        b_vector = bandit.select_action(context, exploration_factor)
        
        # Evaluate
        reward, cost, errors, portfolio_values = env.evaluate_replication(b_vector)
        
        # Track best solution (lowest replication error)
        total_error = np.sum(np.abs(errors))
        if total_error < best_error:
            best_error = total_error
            best_solution = (b_vector.copy(), cost, errors.copy(), portfolio_values.copy())
        
        # Update bandit
        bandit.update(context, b_vector, reward)
        
        rewards_history.append(reward)
        costs_history.append(cost)
        errors_history.append(total_error)
        
        # Print progress
        if (round_num + 1) % print_every == 0:
            recent_reward = np.mean(rewards_history[-print_every:])
            recent_cost = np.mean(costs_history[-print_every:])
            recent_error = np.mean(errors_history[-print_every:])
            
            print(f"Round {round_num + 1}/{n_rounds}")
            print(f"  Exploration factor: {exploration_factor:.4f}")
            print(f"  Avg Reward: {recent_reward:.4f}")
            print(f"  Avg Cost: {recent_cost:.4f}")
            print(f"  Avg Total Error: {recent_error:.6f}")
            if best_solution:
                b_best, cost_best, errs_best, vals_best = best_solution
                print(f"  Best solution so far:")
                print(f"    b = {b_best}")
                print(f"    Cost={cost_best:.4f}, Total error: {best_error:.10f}")
    
    return bandit, best_solution


def evaluate_final_solution(bandit, env, n_samples=5000):
    """Find best replicating portfolio from learned policy"""
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    context = env.get_context()
    
    best_error = float('inf')
    best_solution = None
    all_solutions = []
    
    # Use very low exploration for final evaluation
    for _ in range(n_samples):
        b_vector = bandit.select_action(context, exploration_factor=0.05)
        reward, cost, errors, portfolio_values = env.evaluate_replication(b_vector)
        
        total_error = np.sum(np.abs(errors))
        all_solutions.append((b_vector.copy(), cost, errors.copy(), portfolio_values.copy(), total_error))
        
        if total_error < best_error:
            best_error = total_error
            best_solution = (b_vector.copy(), cost, errors.copy(), portfolio_values.copy())
    
    if best_solution:
        b_vector, cost, errors, portfolio_values = best_solution
        
        print(f"\n*** BEST REPLICATION STRATEGY ***")
        for i in range(env.N):
            print(f"  b{i+1} = {b_vector[i]:.8f}")
        print(f"  Initial cost: {cost:.8f}")
        
        print(f"\n  Cost breakdown:")
        for i in range(env.N):
            print(f"    Binary_{i+1}: {b_vector[i] * env.binary_prices[i]:.8f} ({b_vector[i]:.4f} units @ {env.binary_prices[i]:.4f})")
        
        print(f"\n  Replication verification:")
        for i in range(env.N):
            status = "✓" if abs(errors[i]) < 1e-4 else "❌"
            print(f"    Scenario {i+1}: Portfolio={portfolio_values[i]:.8f}, Payoff={env.C_T[i]:.8f}, Error={errors[i]:.10f} {status}")
        
        print(f"\n  Total absolute error: {np.sum(np.abs(errors)):.10f}")
        print(f"  Cost vs theoretical: {cost:.8f} vs {env.C0:.8f}")
        print(f"  Cost error: {abs(cost - env.C0):.10f}")
        
        # Show distribution of costs (only show best solutions)
        costs = [sol[1] for sol in all_solutions]
        errors_all = [sol[4] for sol in all_solutions]
        
        # Filter to only good solutions (error < 1.0)
        good_solutions = [(c, e) for c, e in zip(costs, errors_all) if e < 1.0]
        if good_solutions:
            good_costs = [c for c, e in good_solutions]
            print(f"\n  Cost statistics from {len(good_solutions)} good solutions (error < 1.0):")
            print(f"    Mean: {np.mean(good_costs):.8f}")
            print(f"    Std: {np.std(good_costs):.8f}")
            print(f"    Min: {np.min(good_costs):.8f}")
            print(f"    Max: {np.max(good_costs):.8f}")
        else:
            print(f"\n  All {n_samples} solutions had error >= 1.0")
    else:
        print("\nNo solution found!")
    
    return best_solution


def find_analytical_solution(env):
    """Calculate the analytical solution for comparison"""
    print("\n" + "="*60)
    print("ANALYTICAL SOLUTION")
    print("="*60)
    
    # In exactly complete market (no bond), unique solution
    b_analytical = env.C_T.copy()
    cost_analytical = np.sum(b_analytical * env.binary_prices)
    
    print("\nUnique Solution (no bond):")
    for i in range(env.N):
        print(f"  b{i+1} = {b_analytical[i]:.8f}")
    print(f"  Cost = {cost_analytical:.8f}")
    
    print(f"\nTheoretical option price: {env.C0:.8f}")
    print("Cost should equal theoretical price (no arbitrage)!")
    
    return b_analytical, cost_analytical


def main_nnomial():
    """
    Use Thompson Sampling to learn exact replication in the complete N-nomial market
    NO BOND - only binaries for unique solution
    
    User can specify any N and factors
    """
    
    print("="*60)
    print("N-NOMIAL OPTION REPLICATION WITH THOMPSON SAMPLING")
    print("="*60)
    print("Objective: Learn exact replication using RL (Thompson Sampling)")
    print("Instruments: Binary_1 + Binary_2 + ... + Binary_N (NO BOND)")
    print("Market: EXACTLY COMPLETE (N scenarios, N instruments)")
    print("UNIQUE SOLUTION: b_i = C_i for all i")
    print("NO BIAS: Uniform prior, no hardcoded preferences")
    print("="*60)
    print()
    
    # Example: 5-nomial model
    # User can change these parameters
    N = 10
    factors = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]  # 10 scenarios

    print(f"Running {N}-nomial example...")
    print()
    
    # Create environment
    env = NnomialBinaryOptionEnvironment(
        S0=100, 
        K=100, 
        r=0.05, 
        T=1.0,
        factors=factors
    )
    
    # Show analytical solution first
    b_ana, cost_ana = find_analytical_solution(env)
    
    # Train Thompson Sampling
    bandit, best_training = train_thompson_sampling(
        env, n_rounds=40000, print_every=2000
    )
    
    # Final evaluation
    best_solution = evaluate_final_solution(bandit, env, n_samples=5000)
    
    # Compare with analytical
    if best_solution:
        b_ts, cost_ts, _, _ = best_solution
        
        print("\n" + "="*60)
        print("THOMPSON SAMPLING vs ANALYTICAL")
        print("="*60)
        print("Thompson Sampling:")
        for i in range(N):
            print(f"  b{i+1} = {b_ts[i]:.6f}")
        print(f"  Cost = {cost_ts:.6f}")
        
        print("\nAnalytical:")
        for i in range(N):
            print(f"  b{i+1} = {b_ana[i]:.6f}")
        print(f"  Cost = {cost_ana:.6f}")
        
        print("\nErrors:")
        for i in range(N):
            print(f"  b{i+1} error: {abs(b_ts[i] - b_ana[i]):.8f}")
        print(f"  Cost error: {abs(cost_ts - cost_ana):.8f}")
        print("="*60)


if __name__ == "__main__":
    main_nnomial()