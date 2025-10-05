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


class TrinomialIncompleteEnvironment:
    """
    Trinomial INCOMPLETE model with pooled binary options (NO BOND)
    
    Trading instruments:
    - Binary_{1,2}: Pays 1 if scenario 1 OR scenario 2 occurs (pooled binary)
    - Binary_3: Pays 1 if scenario 3 occurs
    
    Incomplete market: 3 scenarios, 2 instruments
    SUPER-REPLICATION: Find cheapest portfolio where Portfolio >= Payoff everywhere
    
    Key constraint: Scenarios 1 and 2 must have the SAME portfolio value
    """
    def __init__(self, S0, K, r, T, factors):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        
        # Convert factors to numpy array
        self.factors = np.array(factors)
        self.N = 3  # Trinomial
        
        assert len(self.factors) == 3, "This implementation is for trinomial (N=3)"
        
        print(f"Initializing trinomial INCOMPLETE market...")
        
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
        
        # POOLED Binary option prices
        # Binary_{1,2}: Pays 1 if scenario 1 OR 2 occurs
        # Price = e^(-rT) × (p_1 + p_2)
        self.binary_12_price = np.exp(-r * T) * (self.probabilities[0] + self.probabilities[1])
        
        # Binary_3: Pays 1 if scenario 3 occurs
        # Price = e^(-rT) × p_3
        self.binary_3_price = np.exp(-r * T) * self.probabilities[2]
        
        print(f"\nEnvironment Setup:")
        print(f"  S0 = {self.S0}, K = {self.K}, r = {self.r}, T = {self.T}")
        print(f"  Factors: {self.factors}")
        print(f"  Stock prices at maturity: {self.S_T}")
        print(f"  Option payoffs: {self.C_T}")
        print(f"  Theoretical option price: {self.C0:.4f}")
        
        print(f"\n  INCOMPLETE MARKET - Available Instruments:")
        print(f"  Binary_{{1,2}}: Pays 1 if scenario 1 OR 2, Price = {self.binary_12_price:.4f}")
        print(f"  Binary_3: Pays 1 if scenario 3, Price = {self.binary_3_price:.4f}")
        
        print(f"\n  SUPER-REPLICATION OBJECTIVE:")
        print(f"  Find CHEAPEST portfolio where Portfolio >= Payoff in ALL scenarios")
        print(f"  Target payoffs: C_1={self.C_T[0]:.2f}, C_2={self.C_T[1]:.2f}, C_3={self.C_T[2]:.2f}")
    
    def get_context(self):
        """Get context for bandit (static in this case)"""
        return np.array([self.S0, self.K], dtype=np.float32)
    
    def evaluate_replication(self, b_12, b_3):
        """
        Evaluate super-replication for given (b_12, b_3)
        
        Portfolio values at maturity:
        - Scenario 1: b_12 (Binary_{1,2} pays 1)
        - Scenario 2: b_12 (Binary_{1,2} pays 1)
        - Scenario 3: b_3 (Binary_3 pays 1)
        
        SUPER-REPLICATION: Portfolio >= Payoff in ALL scenarios
        
        Returns:
            reward: -cost if super-replicates, else heavy penalty
            cost: Initial cost of the portfolio
            errors: Replication errors in each scenario
        """
        
        # Initial cost (no bond!)
        cost = b_12 * self.binary_12_price + b_3 * self.binary_3_price
        
        # Portfolio values at maturity
        V_scenario1 = b_12  # Binary_{1,2} pays 1 in scenario 1
        V_scenario2 = b_12  # Binary_{1,2} pays 1 in scenario 2 (SAME AS SCENARIO 1!)
        V_scenario3 = b_3   # Binary_3 pays 1 in scenario 3
        
        portfolio_values = np.array([V_scenario1, V_scenario2, V_scenario3])
        
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
    Thompson Sampling for trinomial incomplete market
    Learning super-replication strategy
    
    NO BIAS: Uniform exploration, no hardcoded preferences
    """
    def __init__(self):
        # Store all observed (action, reward) pairs
        self.observations = []
        self.n_updates = 0
        
    def select_action(self, context, exploration_factor=1.0):
        """
        Sample from posterior (or uniform if no data) - NO BIAS
        exploration_factor: scales noise (1.0 = full noise, 0.0 = no noise)
        """
        if len(self.observations) < 100:  # Initial exploration
            # Pure uniform exploration at start - NO BIAS
            b_12 = np.random.uniform(-10, 60)
            b_3 = np.random.uniform(-10, 60)
        else:
            # After collecting data, sample from empirical distribution
            b_12s = [obs[0] for obs in self.observations]
            b_3s = [obs[1] for obs in self.observations]
            rewards = np.array([obs[2] for obs in self.observations])
            
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
            noise = exploration_factor * 0.2
            
            b_12 = b_12s[idx] + np.random.normal(0, noise)
            b_3 = b_3s[idx] + np.random.normal(0, noise)
            
            # Clip to reasonable range
            b_12 = np.clip(b_12, -10, 60)
            b_3 = np.clip(b_3, -10, 60)
        
        return b_12, b_3
    
    def update(self, context, b_12, b_3, reward):
        """Store observation (non-parametric) - NO BIAS"""
        self.observations.append((b_12, b_3, reward))
        self.n_updates += 1
        
        # Keep only best observations to focus on good regions
        if len(self.observations) > 2000:
            # Keep top 80% by reward
            self.observations.sort(key=lambda x: x[2], reverse=True)
            self.observations = self.observations[:1600]


def train_thompson_sampling(env, n_rounds=20000, print_every=2000):
    """Train Thompson Sampling for super-replication in incomplete market"""
    print("\n" + "="*60)
    print("THOMPSON SAMPLING FOR SUPER-REPLICATION")
    print("="*60)
    print("Objective: Find CHEAPEST portfolio where Portfolio >= Payoff everywhere")
    print("Instruments: Binary_{1,2}, Binary_3")
    print("Constraint: Scenarios 1 and 2 must have SAME portfolio value")
    print("Goal: Minimize cost subject to super-replication constraint")
    print("="*60)
    print()
    
    bandit = ThompsonSamplingBandit()
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
        b_12, b_3 = bandit.select_action(context, exploration_factor)
        
        # Evaluate
        reward, cost, errors, portfolio_values = env.evaluate_replication(b_12, b_3)
        
        # Track violations
        violations = np.sum(np.maximum(0, env.C_T - portfolio_values))
        
        # Track best VALID solution (super-replicates with lowest cost)
        if violations < 1e-6:  # Valid super-replication
            if cost < best_cost:
                best_cost = cost
                best_solution = (b_12, b_3, cost, errors, portfolio_values)
        
        # Update bandit
        bandit.update(context, b_12, b_3, reward)
        
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
                b12_best, b3_best, cost_best, errs_best, vals_best = best_solution
                print(f"  Best VALID solution so far:")
                print(f"    b_{{1,2}}={b12_best:.4f}, b_3={b3_best:.4f}")
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
        b_12, b_3 = bandit.select_action(context, exploration_factor=0.05)
        reward, cost, errors, portfolio_values = env.evaluate_replication(b_12, b_3)
        
        violations = np.sum(np.maximum(0, env.C_T - portfolio_values))
        all_solutions.append((b_12, b_3, cost, errors, portfolio_values, violations))
        
        # Only consider valid super-replicating solutions
        if violations < 1e-6:
            if cost < best_cost:
                best_cost = cost
                best_solution = (b_12, b_3, cost, errors, portfolio_values)
    
    if best_solution:
        b_12, b_3, cost, errors, portfolio_values = best_solution
        
        print(f"\n*** BEST SUPER-REPLICATING STRATEGY ***")
        print(f"  b_{{1,2}} = {b_12:.8f} (applies to scenarios 1 AND 2)")
        print(f"  b_3 = {b_3:.8f}")
        print(f"  Initial cost: {cost:.8f}")
        
        print(f"\n  Cost breakdown:")
        print(f"    Binary_{{1,2}}: {b_12 * env.binary_12_price:.8f} ({b_12:.4f} units @ {env.binary_12_price:.4f})")
        print(f"    Binary_3: {b_3 * env.binary_3_price:.8f} ({b_3:.4f} units @ {env.binary_3_price:.4f})")
        
        print(f"\n  Super-replication verification:")
        for i, scenario in enumerate(['Scenario 1 (up)', 'Scenario 2 (mid)', 'Scenario 3 (down)']):
            violation = max(0, env.C_T[i] - portfolio_values[i])
            if violation < 1e-6:
                status = "✓ SUPER-REPLICATES"
            else:
                status = f"❌ VIOLATION: short by {violation:.4f}"
            print(f"    {scenario}: Portfolio={portfolio_values[i]:.8f}, Payoff={env.C_T[i]:.8f}, {status}")
        
        total_violation = np.sum(np.maximum(0, env.C_T - portfolio_values))
        print(f"\n  Total violation: {total_violation:.10f}")
        print(f"  Minimum cost found: {cost:.8f}")
        
        print(f"\n  SUPER-REPLICATION ANALYSIS:")
        print(f"  - Scenarios 1 and 2 forced to have same value: {b_12:.4f}")
        print(f"  - Target was C_1={env.C_T[0]:.2f}, C_2={env.C_T[1]:.2f}, C_3={env.C_T[2]:.2f}")
        print(f"  - Constraint: b_{{1,2}} >= max(C_1, C_2) = {max(env.C_T[0], env.C_T[1]):.2f}")
        
        # Show statistics
        valid_solutions = [(s[2], s[5]) for s in all_solutions if s[5] < 1e-6]
        if valid_solutions:
            valid_costs = [c for c, v in valid_solutions]
            print(f"\n  Cost statistics from {len(valid_solutions)} valid solutions:")
            print(f"    Mean: {np.mean(valid_costs):.8f}")
            print(f"    Std: {np.std(valid_costs):.8f}")
            print(f"    Min: {np.min(valid_costs):.8f}")
            print(f"    Max: {np.max(valid_costs):.8f}")
        else:
            print(f"\n  WARNING: No valid super-replicating solutions found in {n_samples} samples!")
    else:
        print("\n❌ NO VALID SUPER-REPLICATING SOLUTION FOUND!")
        print("Thompson Sampling failed to learn a portfolio that super-replicates.")
    
    return best_solution


def main_trinomial_incomplete():
    """
    Use Thompson Sampling for super-replication in incomplete market
    NO BOND - only pooled binaries
    """
    
    print("="*60)
    print("TRINOMIAL INCOMPLETE MARKET - SUPER-REPLICATION")
    print("="*60)
    print("Objective: Find CHEAPEST super-replicating portfolio using RL")
    print("Instruments: Binary_{1,2} + Binary_3 (NO BOND)")
    print("Market: INCOMPLETE (3 scenarios, 2 instruments)")
    print("Constraint: Portfolio >= Payoff in ALL scenarios")
    print("Goal: MINIMIZE COST subject to super-replication")
    print("NO BIAS: Uniform prior, no hardcoded preferences")
    print("="*60)
    print()
    
    # Create environment
    env = TrinomialIncompleteEnvironment(
        S0=100, 
        K=100, 
        r=0.05, 
        T=1.0,
        factors=[1.3, 1.0, 0.7]  # up, middle, down
    )
    
    # Train Thompson Sampling
    bandit, best_training = train_thompson_sampling(
        env, n_rounds=20000, print_every=2000
    )
    
    # Final evaluation
    best_solution = evaluate_final_solution(bandit, env, n_samples=5000)
    
    # Analysis
    if best_solution:
        b_12, b_3, cost, errors, portfolio_values = best_solution
        
        print("\n" + "="*60)
        print("SUPER-REPLICATION STRATEGY LEARNED")
        print("="*60)
        print(f"Thompson Sampling learned:")
        print(f"  b_{{1,2}} = {b_12:.4f}")
        print(f"  b_3 = {b_3:.4f}")
        print(f"  Total cost = {cost:.4f}")
        print(f"\nThis means:")
        print(f"  - Hold {b_12:.2f} units of Binary_{{1,2}} (pays if scenario 1 OR 2)")
        print(f"  - Hold {b_3:.2f} units of Binary_3 (pays if scenario 3)")
        print(f"\nSuper-replication check:")
        for i in range(3):
            violation = max(0, env.C_T[i] - portfolio_values[i])
            status = "✓" if violation < 1e-6 else "❌"
            print(f"  Scenario {i+1}: Portfolio {portfolio_values[i]:.2f} {'≥' if violation < 1e-6 else '<'} Payoff {env.C_T[i]:.2f} {status}")
        
        total_violation = np.sum(np.maximum(0, env.C_T - portfolio_values))
        if total_violation < 1e-6:
            print(f"\n✓ SUPER-REPLICATION SUCCESSFUL - Portfolio >= Payoff everywhere!")
            print(f"  Minimum cost found: {cost:.4f}")
        else:
            print(f"\n❌ SUPER-REPLICATION FAILED - Total violation: {total_violation:.4f}")
        print("="*60)


if __name__ == "__main__":
    main_trinomial_incomplete()