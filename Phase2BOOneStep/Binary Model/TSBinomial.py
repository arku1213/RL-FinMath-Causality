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


class BinomialBinaryOptionEnvironment:
    """
    Binomial model with ONLY binary options and bonds for replication
    
    Trading instruments:
    - Bond: B (position in risk-free bond)
    - Binary_1: b1 (number of binary options paying 1 if scenario 1 occurs)
    - Binary_2: b2 (number of binary options paying 1 if scenario 2 occurs)
    
    Complete market: 2 scenarios, 3 instruments (B, b1, b2)
    This is OVERCOMPLETE, so perfect replication is possible
    
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
        
        # Must be binomial
        assert self.N == 2, "This implementation is for binomial (N=2)"
        
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
        
        # Store discount factor
        self.erT = np.exp(self.r * self.T)
        
        # Binary option prices = discounted risk-neutral probabilities
        # Price to buy a binary that pays 1 in scenario i
        self.binary_prices = np.exp(-r * T) * self.probabilities
        
        print(f"\nEnvironment Setup:")
        print(f"  S0 = {self.S0}, K = {self.K}, r = {self.r}, T = {self.T}")
        print(f"  Factor 1 = {self.factors[0]}, Factor 2 = {self.factors[1]}")
        print(f"  S_1 = {self.S_T[0]:.2f}, S_2 = {self.S_T[1]:.2f}")
        print(f"  C_1 = {self.C_T[0]:.2f}, C_2 = {self.C_T[1]:.2f}")
        print(f"  Theoretical option price: {self.C0:.4f}")
        print(f"  Binary prices: p_1 = {self.binary_prices[0]:.4f}, p_2 = {self.binary_prices[1]:.4f}")
    
    def get_context(self):
        """Get context for bandit (static in this case)"""
        return np.array([self.S0, self.K], dtype=np.float32)
    
    def evaluate_replication(self, B, b1, b2):
        """
        Evaluate how well a given (B, b1, b2) replicates the option
        
        For EXACT replication, we want:
        - Portfolio value = Option payoff in ALL scenarios
        - Minimize initial cost
        
        Returns:
        - reward: Negative of squared replication error + cost penalty
        - cost: Initial cost of the portfolio
        - errors: Replication errors in each scenario
        """
        
        # Initial cost
        cost = B + b1 * self.binary_prices[0] + b2 * self.binary_prices[1]
        
        # Portfolio values at maturity
        V_scenario1 = B * self.erT + b1
        V_scenario2 = B * self.erT + b2
        
        portfolio_values = np.array([V_scenario1, V_scenario2])
        
        # Replication errors (should be zero for perfect replication)
        errors = portfolio_values - self.C_T
        
        # Mean squared error
        mse = np.mean(errors ** 2)
        
        # Reward function for EXACT replication:
        # We want zero replication error
        # Penalize error heavily, and slightly penalize cost deviation from theoretical
        
        replication_penalty = mse * 1000  # Heavy penalty for error
        cost_deviation = abs(cost - self.C0)  # Should equal theoretical price
        
        reward = -(replication_penalty + cost_deviation)
        
        return reward, cost, errors, portfolio_values


class ThompsonSamplingBandit:
    """
    Thompson Sampling for continuous actions in binomial case
    Learning to find exact replication with bonds and binary options
    
    NO BIAS: Uniform exploration, no hardcoded preferences
    """
    def __init__(self):
        # Store all observed (action, reward) pairs
        self.observations = []
        self.n_updates = 0
        
    def select_action(self, context):
        """Sample from posterior (or uniform if no data) - NO BIAS"""
        if len(self.observations) < 30:  # Initial exploration
            # Pure uniform exploration at start - NO BIAS
            B = np.random.uniform(-50, 50)
            b1 = np.random.uniform(-10, 30)
            b2 = np.random.uniform(-10, 30)
        else:
            # After collecting data, sample from empirical distribution
            # Weight by softmax of rewards
            Bs = [obs[0] for obs in self.observations]
            b1s = [obs[1] for obs in self.observations]
            b2s = [obs[2] for obs in self.observations]
            rewards = np.array([obs[3] for obs in self.observations])
            
            # Softmax weights (higher reward = higher probability)
            rewards_shifted = rewards - rewards.max()
            exp_rewards = np.exp(rewards_shifted / 0.1)  # Temperature = 0.1
            
            # Handle potential numerical issues
            if np.any(np.isnan(exp_rewards)) or np.any(np.isinf(exp_rewards)):
                weights = np.ones(len(self.observations)) / len(self.observations)
            else:
                weights = exp_rewards / exp_rewards.sum()
            
            # Sample from weighted distribution + noise for exploration
            idx = np.random.choice(len(self.observations), p=weights)
            B = Bs[idx] + np.random.normal(0, 1.0)
            b1 = b1s[idx] + np.random.normal(0, 0.5)
            b2 = b2s[idx] + np.random.normal(0, 0.5)
            
            # Clip to reasonable range
            B = np.clip(B, -50, 50)
            b1 = np.clip(b1, -10, 30)
            b2 = np.clip(b2, -10, 30)
        
        return B, b1, b2
    
    def update(self, context, B, b1, b2, reward):
        """Store observation (non-parametric) - NO BIAS"""
        self.observations.append((B, b1, b2, reward))
        self.n_updates += 1
        
        # Keep only recent observations to prevent memory explosion
        if len(self.observations) > 1000:
            # Remove worst 20%
            self.observations.sort(key=lambda x: x[3])  # Sort by reward
            self.observations = self.observations[200:]


def train_thompson_sampling(env, n_rounds=5000, print_every=500):
    """Train Thompson Sampling for exact replication"""
    print("\n" + "="*60)
    print("THOMPSON SAMPLING FOR BINOMIAL REPLICATION")
    print("="*60)
    print("Objective: Find exact replication using bonds and binaries")
    print("Instruments: Bond (B), Binary_1 (b1), Binary_2 (b2)")
    print("Goal: Portfolio value = Option payoff in ALL scenarios")
    print("="*60)
    print()
    
    bandit = ThompsonSamplingBandit()
    context = env.get_context()
    
    rewards_history = []
    costs_history = []
    errors_history = []
    best_solution = None
    best_error = float('inf')
    
    for round_num in range(n_rounds):
        # Select action
        B, b1, b2 = bandit.select_action(context)
        
        # Evaluate
        reward, cost, errors, portfolio_values = env.evaluate_replication(B, b1, b2)
        
        # Track best solution (lowest replication error)
        total_error = np.sum(np.abs(errors))
        if total_error < best_error:
            best_error = total_error
            best_solution = (B, b1, b2, cost, errors, portfolio_values)
        
        # Update bandit
        bandit.update(context, B, b1, b2, reward)
        
        rewards_history.append(reward)
        costs_history.append(cost)
        errors_history.append(total_error)
        
        # Print progress
        if (round_num + 1) % print_every == 0:
            recent_reward = np.mean(rewards_history[-print_every:])
            recent_cost = np.mean(costs_history[-print_every:])
            recent_error = np.mean(errors_history[-print_every:])
            
            print(f"Round {round_num + 1}/{n_rounds}")
            print(f"  Avg Reward: {recent_reward:.4f}")
            print(f"  Avg Cost: {recent_cost:.4f}")
            print(f"  Avg Total Error: {recent_error:.6f}")
            if best_solution:
                B_best, b1_best, b2_best, cost_best, errs_best, vals_best = best_solution
                print(f"  Best solution so far:")
                print(f"    B={B_best:.4f}, b1={b1_best:.4f}, b2={b2_best:.4f}")
                print(f"    Cost={cost_best:.4f}, Total error: {best_error:.8f}")
    
    return bandit, best_solution


def evaluate_final_solution(bandit, env, n_samples=2000):
    """Find best replicating portfolio from learned policy"""
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    context = env.get_context()
    
    best_error = float('inf')
    best_solution = None
    all_solutions = []
    
    for _ in range(n_samples):
        B, b1, b2 = bandit.select_action(context)
        reward, cost, errors, portfolio_values = env.evaluate_replication(B, b1, b2)
        
        total_error = np.sum(np.abs(errors))
        all_solutions.append((B, b1, b2, cost, errors, portfolio_values, total_error))
        
        if total_error < best_error:
            best_error = total_error
            best_solution = (B, b1, b2, cost, errors, portfolio_values)
    
    if best_solution:
        B, b1, b2, cost, errors, portfolio_values = best_solution
        
        print(f"\n*** BEST REPLICATION STRATEGY ***")
        print(f"  B = {B:.8f}")
        print(f"  b1 = {b1:.8f}")
        print(f"  b2 = {b2:.8f}")
        print(f"  Initial cost: {cost:.8f}")
        
        print(f"\n  Cost breakdown:")
        print(f"    Bond: {B:.8f}")
        print(f"    Binary_1: {b1 * env.binary_prices[0]:.8f} ({b1:.4f} units @ {env.binary_prices[0]:.4f})")
        print(f"    Binary_2: {b2 * env.binary_prices[1]:.8f} ({b2:.4f} units @ {env.binary_prices[1]:.4f})")
        
        print(f"\n  Replication verification:")
        for i, scenario in enumerate(['Scenario 1', 'Scenario 2']):
            status = "✓" if abs(errors[i]) < 1e-6 else "❌"
            print(f"    {scenario}: Portfolio={portfolio_values[i]:.8f}, Payoff={env.C_T[i]:.8f}, Error={errors[i]:.10f} {status}")
        
        print(f"\n  Total absolute error: {np.sum(np.abs(errors)):.10f}")
        print(f"  Cost vs theoretical: {cost:.8f} vs {env.C0:.8f}")
        print(f"  Cost error: {abs(cost - env.C0):.10f}")
        
        # Show distribution of costs
        costs = [sol[3] for sol in all_solutions]
        print(f"\n  Cost statistics from {n_samples} samples:")
        print(f"    Mean: {np.mean(costs):.8f}")
        print(f"    Std: {np.std(costs):.8f}")
        print(f"    Min: {np.min(costs):.8f}")
        print(f"    Max: {np.max(costs):.8f}")
    else:
        print("\nNo solution found!")
    
    return best_solution


def find_analytical_solution(env):
    """Calculate the analytical solution for comparison"""
    print("\n" + "="*60)
    print("ANALYTICAL SOLUTION (for comparison)")
    print("="*60)
    
    # Pure binary solution: B = 0
    B_pure = 0
    b1_pure = env.C_T[0]
    b2_pure = env.C_T[1]
    cost_pure = B_pure + b1_pure * env.binary_prices[0] + b2_pure * env.binary_prices[1]
    
    print("\nPure Binary Strategy:")
    print(f"  B = {B_pure:.8f}")
    print(f"  b1 = {b1_pure:.8f}")
    print(f"  b2 = {b2_pure:.8f}")
    print(f"  Cost = {cost_pure:.8f}")
    
    # Balanced solution: minimize binary positions
    avg_payoff = np.mean(env.C_T)
    B_balanced = avg_payoff / env.erT
    b1_balanced = env.C_T[0] - B_balanced * env.erT
    b2_balanced = env.C_T[1] - B_balanced * env.erT
    cost_balanced = B_balanced + b1_balanced * env.binary_prices[0] + b2_balanced * env.binary_prices[1]
    
    print("\nBalanced Strategy:")
    print(f"  B = {B_balanced:.8f}")
    print(f"  b1 = {b1_balanced:.8f}")
    print(f"  b2 = {b2_balanced:.8f}")
    print(f"  Cost = {cost_balanced:.8f}")
    
    print(f"\nTheoretical option price: {env.C0:.8f}")
    print("All costs should equal theoretical price (no arbitrage)!")
    
    return {
        'pure': (B_pure, b1_pure, b2_pure, cost_pure),
        'balanced': (B_balanced, b1_balanced, b2_balanced, cost_balanced)
    }


def main_binomial():
    """
    Use Thompson Sampling to learn exact replication in the complete binomial market
    """
    
    print("="*60)
    print("BINOMIAL OPTION REPLICATION WITH THOMPSON SAMPLING")
    print("="*60)
    print("Objective: Learn exact replication using RL (Thompson Sampling)")
    print("Instruments: Bond + Binary_1 + Binary_2")
    print("Market: COMPLETE (2 scenarios, 3 instruments)")
    print("NO BIAS: Uniform prior, no hardcoded preferences")
    print("="*60)
    print()
    
    # Create environment
    env = BinomialBinaryOptionEnvironment(
        S0=100, 
        K=100, 
        r=0.05, 
        T=1.0,
        factors=[1.2, 0.8]  # Scenario 1: up 20%, Scenario 2: down 20%
    )
    
    # Show analytical solutions first
    analytical = find_analytical_solution(env)
    
    # Train Thompson Sampling
    bandit, best_training = train_thompson_sampling(
        env, n_rounds=5000, print_every=500
    )
    
    # Final evaluation
    best_solution = evaluate_final_solution(bandit, env, n_samples=2000)
    
    # Compare with analytical
    if best_solution:
        B_ts, b1_ts, b2_ts, cost_ts, _, _ = best_solution
        B_ana, b1_ana, b2_ana, cost_ana = analytical['balanced']
        
        print("\n" + "="*60)
        print("THOMPSON SAMPLING vs ANALYTICAL")
        print("="*60)
        print(f"Thompson Sampling: B={B_ts:.6f}, b1={b1_ts:.6f}, b2={b2_ts:.6f}, Cost={cost_ts:.6f}")
        print(f"Analytical:        B={B_ana:.6f}, b1={b1_ana:.6f}, b2={b2_ana:.6f}, Cost={cost_ana:.6f}")
        print(f"\nErrors:")
        print(f"  B error: {abs(B_ts - B_ana):.8f}")
        print(f"  b1 error: {abs(b1_ts - b1_ana):.8f}")
        print(f"  b2 error: {abs(b2_ts - b2_ana):.8f}")
        print(f"  Cost error: {abs(cost_ts - cost_ana):.8f}")
        print("="*60)


if __name__ == "__main__":
    main_binomial()