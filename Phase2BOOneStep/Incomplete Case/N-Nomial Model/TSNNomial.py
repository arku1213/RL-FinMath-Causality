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


class NnomialIncompleteEnvironment:
    """
    N-nomial INCOMPLETE model with pairwise pooled binary options (NO BOND)
    
    Pooling strategy: Adjacent pairs
    - If N is even: Binary_{1,2}, Binary_{3,4}, ..., Binary_{N-1,N}
    - If N is odd: Binary_{1,2}, Binary_{3,4}, ..., Binary_{N-2,N-1}, Binary_N
    
    Example for N=10:
    - Binary_{1,2}: Pays 1 if scenario 1 OR 2
    - Binary_{3,4}: Pays 1 if scenario 3 OR 4
    - ...
    - Binary_{9,10}: Pays 1 if scenario 9 OR 10
    
    Incomplete market: N scenarios, ceil(N/2) instruments
    SUPER-REPLICATION: Find cheapest portfolio where Portfolio >= Payoff everywhere
    """
    def __init__(self, S0, K, r, T, factors):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        
        # Convert factors to numpy array
        self.factors = np.array(factors)
        self.N = len(self.factors)
        
        print(f"Initializing {self.N}-nomial INCOMPLETE market...")
        
        # Calculate risk-neutral probabilities
        print("Calculating risk-neutral probabilities...")
        self.probabilities = calculate_risk_neutral_probabilities(factors, r, T)
        print(f"Risk-neutral probabilities: {self.probabilities}")
        print(f"Expected return check: {np.sum(self.probabilities * self.factors):.6f} vs target {np.exp(r*T):.6f}")
        
        # Stock prices at maturity
        self.S_T = self.S0 * self.factors
        
        # Option payoffs - call option
        self.C_T = np.maximum(self.S_T - self.K, 0)
        
        # Theoretical option price (risk-neutral valuation)
        self.C0 = np.exp(-r * T) * np.sum(self.probabilities * self.C_T)
        
        # PAIRWISE POOLED Binary option structure
        # Create pairs: (1,2), (3,4), (5,6), ..., and possibly one singleton
        self.pooling = []
        self.n_instruments = 0
        
        i = 0
        while i < self.N:
            if i + 1 < self.N:
                # Pair scenarios i and i+1
                self.pooling.append([i, i+1])
                i += 2
            else:
                # Odd scenario left over
                self.pooling.append([i])
                i += 1
        
        self.n_instruments = len(self.pooling)
        
        # Calculate binary prices for each pooled instrument
        # Binary_j pays 1 if any scenario in pool j occurs
        # Price = e^(-rT) × sum of probabilities in pool
        self.binary_prices = np.zeros(self.n_instruments)
        for j, pool in enumerate(self.pooling):
            prob_sum = sum(self.probabilities[i] for i in pool)
            self.binary_prices[j] = np.exp(-r * T) * prob_sum
        
        print(f"\nEnvironment Setup:")
        print(f"  S0 = {self.S0}, K = {self.K}, r = {self.r}, T = {self.T}")
        print(f"  Number of scenarios: {self.N}")
        print(f"  Number of instruments: {self.n_instruments}")
        print(f"  Factors: {self.factors}")
        print(f"  Stock prices at maturity: {self.S_T}")
        print(f"  Option payoffs: {self.C_T}")
        print(f"  Theoretical option price: {self.C0:.4f}")
        
        print(f"\n  INCOMPLETE MARKET - Pairwise Pooled Binaries:")
        for j, pool in enumerate(self.pooling):
            scenarios_str = ",".join(str(i+1) for i in pool)
            print(f"  Binary_{{{scenarios_str}}}: Pays 1 if scenario {' OR '.join(str(i+1) for i in pool)}, Price = {self.binary_prices[j]:.4f}")
        
        print(f"\n  SUPER-REPLICATION OBJECTIVE:")
        print(f"  Find CHEAPEST portfolio where Portfolio >= Payoff in ALL scenarios")
        print(f"  CONSTRAINT: Scenarios in same pool MUST have same portfolio value")
    
    def get_context(self):
        """Get context for bandit (static in this case)"""
        return np.array([self.S0, self.K], dtype=np.float32)
    
    def evaluate_replication(self, b_vector):
        """
        Evaluate super-replication for given binary positions
        
        Args:
            b_vector: numpy array of shape (n_instruments,) with binary positions
        
        Portfolio values at maturity:
        - For scenario i: value = b_j where j is the pool containing scenario i
        
        SUPER-REPLICATION: Portfolio >= Payoff in ALL scenarios
        
        Returns:
            reward: -cost if super-replicates, else heavy penalty
            cost: Initial cost of the portfolio
            errors: Replication errors in each scenario
        """
        
        # Ensure b_vector is numpy array
        b_vector = np.array(b_vector)
        
        # Initial cost (no bond!)
        cost = np.sum(b_vector * self.binary_prices)
        
        # Portfolio values at maturity
        # For each scenario, find which pool it belongs to, get that binary's value
        portfolio_values = np.zeros(self.N)
        for i in range(self.N):
            # Find which pool contains scenario i
            for j, pool in enumerate(self.pooling):
                if i in pool:
                    portfolio_values[i] = b_vector[j]
                    break
        
        # Replication errors
        errors = portfolio_values - self.C_T
        
        # SUPER-REPLICATION OBJECTIVE
        # Goal: Portfolio >= Payoff in ALL scenarios, minimize cost
        
        # Check for violations (portfolio < payoff)
        violations = np.maximum(0, self.C_T - portfolio_values)
        total_violation = np.sum(violations)
        
        # Reward function for super-replication:
        # 1. If portfolio super-replicates (>= payoff everywhere): reward = -cost (minimize cost)
        # 2. If violations exist: heavily penalize violations
        if total_violation < 1e-6:
            # Super-replication achieved, minimize cost
            reward = -cost
        else:
            # Penalize violations heavily, also consider cost
            reward = -10000 * total_violation - cost
        
        return reward, cost, errors, portfolio_values


class ThompsonSamplingBandit:
    """
    Thompson Sampling for N-nomial incomplete market
    Learning super-replication strategy
    
    NO BIAS: Uniform exploration, no hardcoded preferences
    """
    def __init__(self, n_instruments):
        # Store all observed (action, reward) pairs
        self.observations = []
        self.n_updates = 0
        self.n_instruments = n_instruments
        
        # Scale initial exploration with dimensionality
        self.initial_exploration = max(100, 20 * n_instruments)
        
    def select_action(self, context, exploration_factor=1.0):
        """
        Sample from posterior (or uniform if no data) - NO BIAS
        exploration_factor: scales noise (1.0 = full noise, 0.0 = no noise)
        
        Returns:
            b_vector: numpy array of shape (n_instruments,)
        """
        if len(self.observations) < self.initial_exploration:
            # Pure uniform exploration at start - NO BIAS
            b_vector = np.random.uniform(-10, 60, size=self.n_instruments)
        else:
            # After collecting data, sample from empirical distribution
            b_vectors = np.array([obs[0] for obs in self.observations])
            rewards = np.array([obs[1] for obs in self.observations])
            
            # Softmax weights (higher reward = higher probability)
            rewards_shifted = rewards - rewards.max()
            exp_rewards = np.exp(rewards_shifted / 0.05)
            
            # Handle potential numerical issues
            if np.any(np.isnan(exp_rewards)) or np.any(np.isinf(exp_rewards)):
                weights = np.ones(len(self.observations)) / len(self.observations)
            else:
                weights = exp_rewards / exp_rewards.sum()
            
            # Sample from weighted distribution + adaptive noise
            idx = np.random.choice(len(self.observations), p=weights)
            
            # Adaptive noise that decreases over time
            noise = exploration_factor * 0.3
            
            b_vector = b_vectors[idx] + np.random.normal(0, noise, size=self.n_instruments)
            
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


def train_thompson_sampling(env, n_rounds=30000, print_every=3000):
    """Train Thompson Sampling for super-replication in incomplete market"""
    print("\n" + "="*60)
    print(f"THOMPSON SAMPLING FOR {env.N}-NOMIAL SUPER-REPLICATION")
    print("="*60)
    print(f"Objective: Find CHEAPEST portfolio where Portfolio >= Payoff everywhere")
    print(f"Instruments: {env.n_instruments} pooled binaries (pairwise)")
    print(f"Scenarios: {env.N}, Instruments: {env.n_instruments}")
    print(f"Constraint: Scenarios in same pool must have SAME portfolio value")
    print(f"Goal: Minimize cost subject to super-replication constraint")
    print("="*60)
    print()
    
    bandit = ThompsonSamplingBandit(n_instruments=env.n_instruments)
    context = env.get_context()
    
    rewards_history = []
    costs_history = []
    violations_history = []
    best_solution = None
    best_cost = float('inf')
    
    for round_num in range(n_rounds):
        # Adaptive exploration: decrease noise over time
        exploration_factor = max(0.1, 1.0 - (round_num / n_rounds))
        
        # Select action
        b_vector = bandit.select_action(context, exploration_factor)
        
        # Evaluate
        reward, cost, errors, portfolio_values = env.evaluate_replication(b_vector)
        
        # Track violations
        violations = np.sum(np.maximum(0, env.C_T - portfolio_values))
        
        # Track best VALID solution (super-replicates with lowest cost)
        if violations < 1e-6:  # Valid super-replication
            if cost < best_cost:
                best_cost = cost
                best_solution = (b_vector.copy(), cost, errors.copy(), portfolio_values.copy())
        
        # Update bandit
        bandit.update(context, b_vector, reward)
        
        rewards_history.append(reward)
        costs_history.append(cost)
        violations_history.append(violations)
        
        # Print progress
        if (round_num + 1) % print_every == 0:
            recent_reward = np.mean(rewards_history[-print_every:])
            recent_cost = np.mean(costs_history[-print_every:])
            recent_violation = np.mean(violations_history[-print_every:])
            
            print(f"Round {round_num + 1}/{n_rounds}")
            print(f"  Exploration factor: {exploration_factor:.4f}")
            print(f"  Avg Reward: {recent_reward:.4f}")
            print(f"  Avg Cost: {recent_cost:.4f}")
            print(f"  Avg Violation: {recent_violation:.6f}")
            if best_solution:
                b_best, cost_best, _, _ = best_solution
                print(f"  Best VALID solution:")
                print(f"    b = {b_best[:3]}{'...' if len(b_best) > 3 else ''}")
                print(f"    Cost={cost_best:.4f} (super-replicates!)")
            else:
                print(f"  No valid super-replicating solution found yet")
    
    return bandit, best_solution


def evaluate_final_solution(bandit, env, n_samples=5000):
    """Find best super-replicating portfolio from learned policy"""
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    context = env.get_context()
    
    best_cost = float('inf')
    best_solution = None
    all_solutions = []
    
    # Use very low exploration for final evaluation
    for _ in range(n_samples):
        b_vector = bandit.select_action(context, exploration_factor=0.05)
        reward, cost, errors, portfolio_values = env.evaluate_replication(b_vector)
        
        violations = np.sum(np.maximum(0, env.C_T - portfolio_values))
        all_solutions.append((b_vector.copy(), cost, errors.copy(), portfolio_values.copy(), violations))
        
        # Only consider valid super-replicating solutions
        if violations < 1e-6:
            if cost < best_cost:
                best_cost = cost
                best_solution = (b_vector.copy(), cost, errors.copy(), portfolio_values.copy())
    
    if best_solution:
        b_vector, cost, errors, portfolio_values = best_solution
        
        print(f"\n*** BEST SUPER-REPLICATING STRATEGY ***")
        for j, pool in enumerate(env.pooling):
            scenarios_str = ",".join(str(i+1) for i in pool)
            print(f"  b_{{{scenarios_str}}} = {b_vector[j]:.8f}")
        print(f"  Initial cost: {cost:.8f}")
        
        print(f"\n  Cost breakdown:")
        for j, pool in enumerate(env.pooling):
            scenarios_str = ",".join(str(i+1) for i in pool)
            print(f"    Binary_{{{scenarios_str}}}: {b_vector[j] * env.binary_prices[j]:.8f} ({b_vector[j]:.4f} units @ {env.binary_prices[j]:.4f})")
        
        print(f"\n  Super-replication verification:")
        for i in range(env.N):
            violation = max(0, env.C_T[i] - portfolio_values[i])
            if violation < 1e-6:
                status = "✓ SUPER-REPLICATES"
            else:
                status = f"❌ VIOLATION: short by {violation:.4f}"
            print(f"    Scenario {i+1}: Portfolio={portfolio_values[i]:.4f}, Payoff={env.C_T[i]:.4f}, {status}")
        
        total_violation = np.sum(np.maximum(0, env.C_T - portfolio_values))
        print(f"\n  Total violation: {total_violation:.10f}")
        print(f"  Minimum cost found: {cost:.8f}")
        
        # Show statistics
        valid_solutions = [(s[1], s[4]) for s in all_solutions if s[4] < 1e-6]
        if valid_solutions:
            valid_costs = [c for c, v in valid_solutions]
            print(f"\n  Cost statistics from {len(valid_solutions)} valid solutions:")
            print(f"    Mean: {np.mean(valid_costs):.8f}")
            print(f"    Std: {np.std(valid_costs):.8f}")
            print(f"    Min: {np.min(valid_costs):.8f}")
            print(f"    Max: {np.max(valid_costs):.8f}")
        else:
            print(f"\n  WARNING: No valid super-replicating solutions in {n_samples} samples!")
    else:
        print("\n❌ NO VALID SUPER-REPLICATING SOLUTION FOUND!")
        print("Thompson Sampling failed to learn a portfolio that super-replicates.")
    
    return best_solution


def main_nnomial_incomplete():
    """
    Use Thompson Sampling for super-replication in N-nomial incomplete market
    NO BOND - only pairwise pooled binaries
    """
    
    print("="*60)
    print("N-NOMIAL INCOMPLETE MARKET - PAIRWISE POOLING")
    print("="*60)
    print("Objective: Find CHEAPEST super-replicating portfolio using RL")
    print("Pooling: Adjacent pairs (1-2, 3-4, 5-6, ...)")
    print("Market: INCOMPLETE (N scenarios, ceil(N/2) instruments)")
    print("Constraint: Portfolio >= Payoff in ALL scenarios")
    print("Goal: MINIMIZE COST subject to super-replication")
    print("NO BIAS: Uniform prior, no hardcoded preferences")
    print("="*60)
    print()
    
    # Example: 10-nomial incomplete
    N = 10
    factors = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
    
    print(f"Running {N}-nomial incomplete example...")
    print()
    
    # Create environment
    env = NnomialIncompleteEnvironment(
        S0=100, 
        K=100, 
        r=0.05, 
        T=1.0,
        factors=factors
    )
    
    # Train Thompson Sampling
    bandit, best_training = train_thompson_sampling(
        env, n_rounds=30000, print_every=3000
    )
    
    # Final evaluation
    best_solution = evaluate_final_solution(bandit, env, n_samples=5000)
    
    # Analysis
    if best_solution:
        b_vector, cost, errors, portfolio_values = best_solution
        
        print("\n" + "="*60)
        print("SUPER-REPLICATION STRATEGY LEARNED")
        print("="*60)
        print(f"Thompson Sampling learned:")
        for j, pool in enumerate(env.pooling):
            scenarios_str = ",".join(str(i+1) for i in pool)
            print(f"  b_{{{scenarios_str}}} = {b_vector[j]:.4f}")
        print(f"  Total cost = {cost:.4f}")
        
        print(f"\nSuper-replication check:")
        total_violation = 0
        for i in range(env.N):
            violation = max(0, env.C_T[i] - portfolio_values[i])
            total_violation += violation
            status = "✓" if violation < 1e-6 else "❌"
            print(f"  Scenario {i+1}: Portfolio {portfolio_values[i]:.2f} {'≥' if violation < 1e-6 else '<'} Payoff {env.C_T[i]:.2f} {status}")
        
        if total_violation < 1e-6:
            print(f"\n✓ SUPER-REPLICATION SUCCESSFUL!")
            print(f"  Minimum cost: {cost:.4f}")
            print(f"  Theoretical price (complete market): {env.C0:.4f}")
            print(f"  Cost of incompleteness: {cost - env.C0:.4f} ({100*(cost - env.C0)/env.C0:.1f}% premium)")
        else:
            print(f"\n❌ SUPER-REPLICATION FAILED")
        print("="*60)


if __name__ == "__main__":
    main_nnomial_incomplete()