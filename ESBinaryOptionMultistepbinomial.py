"""
Evolution Strategies (ES) for Multi-Step Binomial Hedging with Binary Options
Uses CMA-ES (Covariance Matrix Adaptation) for constrained optimization
"""

import numpy as np
from itertools import product
from typing import Tuple, Dict, List

np.random.seed(42)


class BinomialHedgingEnv:
    """Multi-step binomial environment for static hedging evaluation."""
    
    def __init__(self, S0=100.0, K=100.0, r=0.05, T=2, u=1.2, d=0.8):
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
        
        # Binary option prices
        self.binary_up_price = np.exp(-r * self.dt) * self.p
        self.binary_down_price = np.exp(-r * self.dt) * self.q
        
        # Generate all possible paths
        self.paths = list(product([0, 1], repeat=T))
        self.n_paths = len(self.paths)
        
        # Terminal prices and payoffs
        self.S_T = np.zeros(self.n_paths)
        self.C_T = np.zeros(self.n_paths)
        self.path_probs = np.zeros(self.n_paths)
        
        for i, path in enumerate(self.paths):
            n_ups = sum(path)
            n_downs = T - n_ups
            
            self.S_T[i] = S0 * (u ** n_ups) * (d ** n_downs)
            self.C_T[i] = max(self.S_T[i] - K, 0)
            self.path_probs[i] = (self.p ** n_ups) * (self.q ** n_downs)
        
        # Theoretical option price
        self.C0_theoretical = np.exp(-r * T) * np.sum(self.path_probs * self.C_T)
        
        print(f"\nBinomial Hedging Environment (T={T})")
        print(f"S0={S0}, K={K}, r={r}")
        print(f"Risk-neutral: p={self.p:.6f}, q={self.q:.6f}")
        print(f"Binary prices: up={self.binary_up_price:.6f}, down={self.binary_down_price:.6f}")
        print(f"Theoretical option price: {self.C0_theoretical:.6f}")
        print(f"Paths: {self.n_paths}")
    
    def evaluate(self, params: np.ndarray) -> Tuple[float, Dict]:
        """
        Evaluate hedge parameters [delta, B, b_up, b_down].
        Returns: (objective, info_dict)
        """
        delta, B, b_up, b_down = params
        
        # Hedge cost
        hedge_cost = (delta * self.S0 + B + 
                     b_up * self.binary_up_price * self.T +
                     b_down * self.binary_down_price * self.T)
        
        # Portfolio values at maturity
        portfolio_values = np.zeros(self.n_paths)
        for i, path in enumerate(self.paths):
            n_ups = sum(path)
            n_downs = self.T - n_ups
            
            portfolio_values[i] = (delta * self.S_T[i] + 
                                  B * np.exp(self.r * self.T) + 
                                  b_up * n_ups + 
                                  b_down * n_downs)
        
        # Super-replication gaps
        gaps = portfolio_values - self.C_T
        min_gap = np.min(gaps)
        
        # Check feasibility
        is_feasible = (min_gap >= -1e-6) and (hedge_cost > 1e-6)
        
        # Objective function (minimize)
        if not is_feasible:
            # Penalty for constraint violation
            if hedge_cost <= 1e-6:
                penalty = 1000.0 + abs(hedge_cost) * 100
            else:
                penalty = 1000.0 + abs(min_gap) * 100
            objective = hedge_cost + penalty
        else:
            # Minimize: cost + excess coverage
            avg_gap = np.mean(gaps)
            max_gap = np.max(gaps)
            std_gap = np.std(gaps)
            
            # Weighted sum favoring tight hedges
            objective = hedge_cost + 2.0 * avg_gap + 1.0 * max_gap + 0.5 * std_gap
        
        info = {
            'hedge_cost': hedge_cost,
            'min_gap': min_gap,
            'mean_gap': np.mean(gaps),
            'max_gap': np.max(gaps),
            'std_gap': np.std(gaps),
            'is_feasible': is_feasible,
            'gaps': gaps
        }
        
        return objective, info


class SimpleES:
    """
    Simple Evolution Strategy with adaptive parameters.
    Uses (μ, λ) selection with covariance matrix adaptation.
    """
    
    def __init__(self, 
                 dim: int = 4,
                 popsize: int = 50,
                 mu: int = 25,
                 sigma: float = 0.5,
                 bounds: List[Tuple[float, float]] = None):
        """
        Args:
            dim: Dimension of parameter space
            popsize: Population size (λ)
            mu: Number of parents (μ)
            sigma: Initial step size
            bounds: List of (min, max) for each dimension
        """
        self.dim = dim
        self.popsize = popsize
        self.mu = mu
        self.sigma = sigma
        
        # Bounds for each parameter
        if bounds is None:
            # Default: [delta, B, b_up, b_down]
            self.bounds = np.array([
                [0.0, 2.0],      # delta
                [-100.0, 100.0],  # B
                [-50.0, 50.0],    # b_up
                [-50.0, 50.0]     # b_down
            ])
        else:
            self.bounds = np.array(bounds)
        
        # Initialize mean in center of bounds
        self.mean = np.mean(self.bounds, axis=1)
        
        # Covariance matrix (diagonal initially)
        self.C = np.eye(dim)
        
        # Evolution path
        self.pc = np.zeros(dim)
        
        # Learning rates
        self.cc = 4.0 / (dim + 4.0)  # For evolution path
        self.c1 = 2.0 / (dim**2)      # For covariance update
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mu - 1) / (dim + 1)) - 1)
        
        # Weights for recombination
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
    
    def clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip parameters to valid bounds."""
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
    
    def ask(self) -> np.ndarray:
        """Generate population of candidate solutions."""
        population = []
        
        for _ in range(self.popsize):
            # Sample from multivariate Gaussian
            z = np.random.randn(self.dim)
            y = self.mean + self.sigma * (self.C @ z)
            
            # Clip to bounds
            y = self.clip_to_bounds(y)
            population.append(y)
        
        return np.array(population)
    
    def tell(self, population: np.ndarray, fitnesses: np.ndarray):
        """Update distribution based on fitness evaluations."""
        # Sort by fitness (lower is better)
        sorted_indices = np.argsort(fitnesses)
        population = population[sorted_indices]
        fitnesses = fitnesses[sorted_indices]
        
        # Track best
        if fitnesses[0] < self.best_fitness:
            self.best_fitness = fitnesses[0]
            self.best_solution = population[0].copy()
        
        # Select top μ individuals
        elite = population[:self.mu]
        
        # Weighted recombination
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * elite, axis=0)
        self.mean = self.clip_to_bounds(self.mean)
        
        # Update evolution path
        mean_diff = (self.mean - old_mean) / self.sigma
        self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * mean_diff
        
        # Update covariance matrix
        rank_one = self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]
        
        # Rank-μ update
        centered_elite = (elite - old_mean) / self.sigma
        rank_mu = np.sum(self.weights[:, np.newaxis, np.newaxis] * 
                        (centered_elite[:, :, np.newaxis] @ centered_elite[:, np.newaxis, :]), 
                        axis=0)
        
        self.C = (1 - self.c1) * self.C + self.c1 * rank_one + self.c1 * rank_mu
        
        # Ensure positive definite
        self.C = (self.C + self.C.T) / 2
        eigenvalues = np.linalg.eigvalsh(self.C)
        if np.min(eigenvalues) < 1e-10:
            self.C += np.eye(self.dim) * 1e-8
        
        # Adapt step size
        self.sigma *= np.exp((np.linalg.norm(self.pc) / np.sqrt(self.dim) - 1) * self.cc / self.damps)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
        
        # Store history
        self.history.append({
            'best_fitness': self.best_fitness,
            'mean_fitness': np.mean(fitnesses),
            'sigma': self.sigma
        })


def train_es(env: BinomialHedgingEnv, 
             generations: int = 500,
             popsize: int = 100,
             log_freq: int = 50) -> Tuple[np.ndarray, Dict]:
    """Train ES on binomial hedging problem."""
    
    print("\n" + "="*80)
    print("EVOLUTION STRATEGIES: Binomial Hedging with Binary Options")
    print("="*80)
    print(f"Population size: {popsize}")
    print(f"Generations: {generations}")
    print("="*80 + "\n")
    
    # Initialize ES
    es = SimpleES(
        dim=4,
        popsize=popsize,
        mu=popsize // 2,
        sigma=0.3,
        bounds=[
            [0.0, 2.0],      # delta
            [-100.0, 100.0],  # B
            [-50.0, 50.0],    # b_up
            [-50.0, 50.0]     # b_down
        ]
    )
    
    best_feasible = None
    best_feasible_cost = float('inf')
    
    for gen in range(generations):
        # Generate population
        population = es.ask()
        
        # Evaluate
        fitnesses = np.zeros(popsize)
        infos = []
        
        for i, params in enumerate(population):
            fitness, info = env.evaluate(params)
            fitnesses[i] = fitness
            infos.append(info)
            
            # Track best feasible solution
            if info['is_feasible'] and info['hedge_cost'] < best_feasible_cost:
                best_feasible_cost = info['hedge_cost']
                best_feasible = (params.copy(), info)
        
        # Update ES
        es.tell(population, fitnesses)
        
        # Logging
        if (gen + 1) % log_freq == 0 or gen == 0:
            feasible_count = sum(1 for info in infos if info['is_feasible'])
            feasible_rate = feasible_count / popsize * 100
            
            print(f"Generation {gen + 1:4d}")
            print(f"  Best fitness: {es.best_fitness:.4f}")
            print(f"  Mean fitness: {np.mean(fitnesses):.4f}")
            print(f"  Sigma: {es.sigma:.6f}")
            print(f"  Feasible rate: {feasible_rate:.1f}%")
            
            if best_feasible:
                params, info = best_feasible
                premium = (info['hedge_cost'] - env.C0_theoretical) / env.C0_theoretical * 100
                print(f"  Best feasible hedge:")
                print(f"    δ={params[0]:.4f}, B={params[1]:.4f}, "
                      f"b_up={params[2]:.4f}, b_down={params[3]:.4f}")
                print(f"    Cost={info['hedge_cost']:.4f} (premium={premium:.2f}%)")
                print(f"    Gaps: min={info['min_gap']:.4f}, mean={info['mean_gap']:.4f}, max={info['max_gap']:.4f}")
            print()
    
    return es, best_feasible


def evaluate_solution(env: BinomialHedgingEnv, params: np.ndarray, info: Dict):
    """Detailed evaluation of final solution."""
    
    print("\n" + "="*80)
    print("FINAL SOLUTION")
    print("="*80 + "\n")
    
    delta, B, b_up, b_down = params
    
    print("*** BEST HEDGE ***")
    print(f"  δ (stock)     = {delta:.6f}")
    print(f"  B (bond)      = {B:.6f}")
    print(f"  b_up (binary) = {b_up:.6f}")
    print(f"  b_down (bin.) = {b_down:.6f}")
    
    print(f"\n  Hedge cost: {info['hedge_cost']:.6f}")
    
    # Cost breakdown
    stock_cost = delta * env.S0
    bond_cost = B
    binary_cost = (b_up * env.binary_up_price * env.T + 
                  b_down * env.binary_down_price * env.T)
    
    print(f"  Cost breakdown:")
    print(f"    Stock:    {stock_cost:.6f}")
    print(f"    Bond:     {bond_cost:.6f}")
    print(f"    Binaries: {binary_cost:.6f}")
    
    print(f"\n  Theoretical option price: {env.C0_theoretical:.6f}")
    premium = info['hedge_cost'] - env.C0_theoretical
    premium_pct = premium / env.C0_theoretical * 100
    print(f"  Super-replication premium: {premium:.6f} ({premium_pct:.2f}%)")
    
    print(f"\n  Gap statistics:")
    print(f"    Min:  {info['min_gap']:.6f}")
    print(f"    Mean: {info['mean_gap']:.6f}")
    print(f"    Max:  {info['max_gap']:.6f}")
    print(f"    Std:  {info['std_gap']:.6f}")
    
    # Show all paths
    print(f"\n  Super-replication verification (all {env.n_paths} paths):")
    for i, path in enumerate(env.paths):
        path_str = ''.join(['U' if x else 'D' for x in path])
        n_ups = sum(path)
        n_downs = env.T - n_ups
        
        portfolio_val = (delta * env.S_T[i] + 
                       B * np.exp(env.r * env.T) + 
                       b_up * n_ups + 
                       b_down * n_downs)
        
        status = "✓" if info['gaps'][i] >= -1e-6 else "✗"
        print(f"    Path {path_str}: S_T={env.S_T[i]:6.2f}, "
              f"Portfolio={portfolio_val:7.4f} ≥ Payoff={env.C_T[i]:7.4f} {status} "
              f"(gap={info['gaps'][i]:7.4f})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Configuration - MODIFY THESE PARAMETERS
    T = 3  # Number of time steps (change this to 2, 3, 4, 5, 6, etc.)
    S0 = 100.0
    K = 100.0
    r = 0.05
    u = 1.2
    d = 0.8
    
    # Scale training parameters based on T
    # More paths → need more generations and larger population
    n_paths = 2 ** T
    generations = min(1000 + T * 200, 3000)  # Scale with complexity
    popsize = min(100 + T * 10, 200)
    log_freq = max(generations // 10, 50)
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION")
    print(f"{'='*80}")
    print(f"Time steps: T = {T}")
    print(f"Initial price: S0 = {S0}")
    print(f"Strike: K = {K}")
    print(f"Risk-free rate: r = {r}")
    print(f"Up factor: u = {u}")
    print(f"Down factor: d = {d}")
    print(f"\nTraining parameters:")
    print(f"  Generations: {generations}")
    print(f"  Population size: {popsize}")
    print(f"  Expected paths: {n_paths}")
    print(f"{'='*80}")
    
    # Create environment
    env = BinomialHedgingEnv(S0=S0, K=K, r=r, T=T, u=u, d=d)
    
    # Train ES
    es, best_solution = train_es(
        env,
        generations=generations,
        popsize=popsize,
        log_freq=log_freq
    )
    
    # Evaluate
    if best_solution:
        params, info = best_solution
        evaluate_solution(env, params, info)
    else:
        print("\nNo feasible solution found!")
        print("Try increasing generations or population size.")