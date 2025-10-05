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
            reward: Reward that enforces ALL scenarios must be perfectly replicated
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
        
        # NEW REWARD: Must replicate ALL scenarios correctly
        # Use squared errors to heavily penalize large deviations
        # AND use max error to ensure no single scenario is badly replicated
        
        max_error = np.max(np.abs(errors))
        mean_squared_error = np.mean(errors ** 2)
        
        # Exponential penalty for max error - forces ALL scenarios to be correct
        # If max_error = 0, reward = 0 (best)
        # If max_error > 0, reward becomes very negative
        reward = -(mean_squared_error + max_error ** 2 * 100)
        
        return reward, cost, errors, portfolio_values


class CMAESOptimizer:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    
    CMA-ES is a state-of-the-art evolutionary algorithm for continuous optimization.
    It adapts the full covariance matrix of the search distribution, making it much
    more robust to local minima than CEM.
    
    Key advantages over CEM:
    1. Adapts correlations between variables (full covariance matrix)
    2. Better global search with step-size control
    3. More robust to local minima
    4. Self-adaptive to the problem landscape
    
    NO BIAS: Starts with identity covariance (no assumed correlations)
    """
    def __init__(self, n_dim, population_size=None, initial_mean=None, initial_sigma=5.0):
        self.n_dim = n_dim
        
        # Population size (lambda) - default formula from Hansen
        if population_size is None:
            self.population_size = 4 + int(3 * np.log(n_dim))
        else:
            self.population_size = population_size
        
        # Number of parents (mu) - top half
        self.n_parents = self.population_size // 2
        
        # Initial mean - start at middle of range
        if initial_mean is None:
            self.mean = np.ones(n_dim) * 10.0
        else:
            self.mean = np.array(initial_mean)
        
        # Step size (sigma)
        self.sigma = initial_sigma
        
        # Covariance matrix (C) - start with identity (NO BIAS)
        self.C = np.eye(n_dim)
        
        # Evolution paths for covariance and step-size adaptation
        self.pc = np.zeros(n_dim)  # Evolution path for C
        self.ps = np.zeros(n_dim)  # Evolution path for sigma
        
        # Weights for recombination (weighted by rank)
        self.weights = np.log(self.n_parents + 0.5) - np.log(np.arange(1, self.n_parents + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        
        # Strategy parameters (from Hansen's CMA-ES)
        self.cc = (4 + self.mu_eff / n_dim) / (n_dim + 4 + 2 * self.mu_eff / n_dim)
        self.cs = (self.mu_eff + 2) / (n_dim + self.mu_eff + 5)
        self.c1 = 2 / ((n_dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((n_dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (n_dim + 1)) - 1) + self.cs
        
        # Expected length of random vector
        self.chi_n = n_dim ** 0.5 * (1 - 1/(4*n_dim) + 1/(21*n_dim**2))
        
        # Track best solution
        self.best_solution = None
        self.best_fitness = -np.inf
        self.generation = 0
        
    def sample_population(self):
        """Sample population from multivariate normal distribution"""
        # Eigendecomposition for sampling
        D, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D, 0))  # Ensure non-negative
        
        population = []
        for _ in range(self.population_size):
            # Sample from N(0, I)
            z = np.random.randn(self.n_dim)
            # Transform to N(mean, sigma^2 * C)
            y = B @ (D * z)
            x = self.mean + self.sigma * y
            # Clip to bounds (wider range to include b1=50, b2=40)
            x = np.clip(x, -10, 60)
            population.append(x)
        
        return np.array(population)
    
    def update(self, population, fitness):
        """
        Update CMA-ES parameters based on population fitness
        
        Args:
            population: Array of shape (population_size, n_dim)
            fitness: Array of shape (population_size,) - higher is better
        """
        # Sort by fitness (descending)
        indices = np.argsort(fitness)[::-1]
        population_sorted = population[indices]
        fitness_sorted = fitness[indices]
        
        # Track best
        if fitness_sorted[0] > self.best_fitness:
            self.best_fitness = fitness_sorted[0]
            self.best_solution = population_sorted[0].copy()
        
        # Select parents (top mu)
        parents = population_sorted[:self.n_parents]
        
        # Recombination: new mean
        mean_old = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * parents, axis=0)
        
        # Cumulation: update evolution paths
        mean_diff = (self.mean - mean_old) / self.sigma
        
        # Compute C^(-1/2) * mean_diff for evolution paths
        D, B = np.linalg.eigh(self.C)
        D = np.maximum(D, 1e-10)
        C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
        
        # Update evolution path for sigma (ps)
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (C_inv_sqrt @ mean_diff)
        
        # Compute heaviside function for pc update
        norm_ps = np.linalg.norm(self.ps)
        hsig = float(norm_ps / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / self.chi_n 
                     < 1.4 + 2 / (self.n_dim + 1))
        
        # Update evolution path for C (pc)
        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * mean_diff
        
        # Adapt covariance matrix
        # Rank-one update
        rank_one = self.c1 * (np.outer(self.pc, self.pc) - self.C)
        
        # Rank-mu update
        y_weighted = np.zeros((self.n_dim, self.n_dim))
        for i in range(self.n_parents):
            y = (parents[i] - mean_old) / self.sigma
            y_weighted += self.weights[i] * np.outer(y, y)
        rank_mu = self.cmu * (y_weighted - self.C)
        
        self.C = self.C + rank_one + rank_mu
        
        # Adapt step size using cumulative path length
        self.sigma *= np.exp((self.cs / self.damps) * (norm_ps / self.chi_n - 1))
        
        self.generation += 1
    
    def get_action(self):
        """Sample single action from current distribution"""
        D, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(D, 0))
        z = np.random.randn(self.n_dim)
        y = B @ (D * z)
        x = self.mean + self.sigma * y
        return np.clip(x, -10, 60)


def train_cmaes(env, n_generations=200, population_size=None, print_every=20, initial_mean=None, initial_sigma=5.0):
    """
    Train using CMA-ES for high-dimensional optimization
    
    CMA-ES is more robust than CEM because it:
    1. Adapts the full covariance matrix (learns variable correlations)
    2. Has sophisticated step-size control
    3. Better escapes local minima
    
    Args:
        n_generations: Number of CMA-ES generations
        population_size: Population size (None = auto-select based on dimension)
        print_every: Print progress every N generations
        initial_mean: Initial mean for search distribution
        initial_sigma: Initial step size
    """
    print("\n" + "="*60)
    print(f"CMA-ES FOR {env.N}-NOMIAL REPLICATION")
    print("="*60)
    print("Objective: Find exact replication using ONLY binaries (no bond)")
    print(f"Instruments: Binary_1, Binary_2, ..., Binary_{env.N}")
    print("Goal: Portfolio value = Option payoff in ALL scenarios")
    print("Unique solution: b_i = C_i for all i")
    if population_size is None:
        auto_pop_size = 4 + int(3 * np.log(env.N))
        print(f"CMA-ES Parameters: pop_size={auto_pop_size} (auto), generations={n_generations}")
    else:
        print(f"CMA-ES Parameters: pop_size={population_size}, generations={n_generations}")
    if initial_mean is not None:
        print(f"Initial mean: {initial_mean[:5] if len(initial_mean) > 5 else initial_mean}")
    print(f"Initial sigma: {initial_sigma}")
    print("="*60)
    print()
    
    cmaes = CMAESOptimizer(
        n_dim=env.N,
        population_size=population_size,
        initial_mean=initial_mean,
        initial_sigma=initial_sigma
    )
    
    best_overall_solution = None
    best_overall_error = float('inf')
    best_overall_reward = -np.inf
    
    for generation in range(n_generations):
        # Sample population
        population = cmaes.sample_population()
        
        # Evaluate all samples
        fitness = []  # CMA-ES maximizes fitness
        errors_list = []
        costs = []
        
        for b_vector in population:
            reward, cost, errors, portfolio_values = env.evaluate_replication(b_vector)
            fitness.append(reward)  # CMA-ES maximizes this
            errors_list.append(np.sum(np.abs(errors)))
            costs.append(cost)
        
        fitness = np.array(fitness)
        errors_list = np.array(errors_list)
        
        # Track best solution found
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_overall_reward:
            best_overall_reward = fitness[best_idx]
            best_overall_error = errors_list[best_idx]
            best_overall_solution = (population[best_idx].copy(), costs[best_idx])
        
        # Update CMA-ES distribution
        cmaes.update(population, fitness)
        
        # Print progress
        if (generation + 1) % print_every == 0:
            avg_fitness = np.mean(fitness)
            avg_error = np.mean(errors_list)
            best_error_this_gen = errors_list[best_idx]
            
            print(f"Generation {generation + 1}/{n_generations}")
            print(f"  Population avg fitness: {avg_fitness:.4f}")
            print(f"  Population avg error: {avg_error:.4f}")
            print(f"  Best this generation: error={best_error_this_gen:.6f}")
            print(f"  Best overall: error={best_overall_error:.6f}")
            print(f"  Step size (sigma): {cmaes.sigma:.6f}")
            print(f"  Current mean: {cmaes.mean[:5]}..." if env.N > 5 else f"  Current mean: {cmaes.mean}")
    
    return cmaes, best_overall_solution


def evaluate_cmaes_solution(cmaes, env, n_samples=1000):
    """Evaluate final CMA-ES solution"""
    print("\n" + "="*60)
    print("FINAL EVALUATION (CMA-ES)")
    print("="*60)
    
    best_error = float('inf')
    best_solution = None
    all_solutions = []
    
    # Sample from final converged distribution
    for _ in range(n_samples):
        b_vector = cmaes.get_action()
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
        
        # Show distribution of costs
        costs = [sol[1] for sol in all_solutions]
        errors_all = [sol[4] for sol in all_solutions]
        
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
    Use CMA-ES for N-nomial replication
    CMA-ES is superior to CEM for high-dimensional problems
    
    User can specify any N and factors
    """
    
    print("="*60)
    print("N-NOMIAL OPTION REPLICATION WITH CMA-ES")
    print("="*60)
    print("Objective: Learn exact replication using RL (CMA-ES)")
    print("Instruments: Binary_1 + Binary_2 + ... + Binary_N (NO BOND)")
    print("Market: EXACTLY COMPLETE (N scenarios, N instruments)")
    print("UNIQUE SOLUTION: b_i = C_i for all i")
    print("NO BIAS: Starts with identity covariance, no prior knowledge")
    print("="*60)
    print()
    
    # Example: 10-nomial model (CMA-ES handles this excellently)
    # User can change these parameters
    N = 10
    factors = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
    
    print(f"Running {N}-nomial example with CMA-ES...")
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
    
    # Train using CMA-ES (superior to CEM for high dimensions)
    # UNBIASED INITIALIZATION: Start at zero (no position = neutral)
    cmaes, best_training = train_cmaes(
        env, 
        n_generations=400,      # More generations for unbiased search
        population_size=None,    # Auto-select based on dimension (4 + 3*ln(N))
        print_every=20,
        initial_mean=np.zeros(N),  # Unbiased: start with no positions
        initial_sigma=20.0          # Wide exploration to find solution
    )
    
    # Final evaluation
    best_solution = evaluate_cmaes_solution(cmaes, env, n_samples=1000)
    
    # Compare with analytical
    if best_solution:
        b_cmaes, cost_cmaes, _, _ = best_solution
        
        print("\n" + "="*60)
        print("CMA-ES vs ANALYTICAL")
        print("="*60)
        print("CMA-ES:")
        for i in range(N):
            print(f"  b{i+1} = {b_cmaes[i]:.6f}")
        print(f"  Cost = {cost_cmaes:.6f}")
        
        print("\nAnalytical:")
        for i in range(N):
            print(f"  b{i+1} = {b_ana[i]:.6f}")
        print(f"  Cost = {cost_ana:.6f}")
        
        print("\nErrors:")
        for i in range(N):
            print(f"  b{i+1} error: {abs(b_cmaes[i] - b_ana[i]):.8f}")
        print(f"  Cost error: {abs(cost_cmaes - cost_ana):.8f}")
        print("="*60)


if __name__ == "__main__":
    main_nnomial()