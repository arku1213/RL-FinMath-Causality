import numpy as np
from math import comb

class BinomialEnvironment:
    """
    Binomial tree with TERMINAL-PAYING binaries.
    Binaries pay $1 at terminal time T based on which terminal node is reached.
    This eliminates path-dependence and creates a proper complete market.
    """
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T_steps=2, dt=1.0, option_type='call'):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_steps = T_steps
        self.dt = dt
        self.option_type = option_type
        
        # CRR model
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Build tree
        self.build_tree()
        self.calculate_terminal_payoffs()
        self.calculate_binary_prices()
        
    def build_tree(self):
        """Build recombining binomial tree"""
        self.nodes = {}
        
        for t in range(self.T_steps + 1):
            for n_ups in range(t + 1):
                price = self.S0 * (self.u ** n_ups) * (self.d ** (t - n_ups))
                self.nodes[(t, n_ups)] = price
        
        # Terminal-paying binaries: one per terminal node
        self.n_binaries = self.T_steps + 1
        
        print(f"\n{'='*60}")
        print(f"Binomial Tree Structure (T={self.T_steps})")
        print(f"{'='*60}")
        print(f"Terminal nodes: {self.n_binaries}")
        print(f"Binaries (terminal-paying): {self.n_binaries}")
        print(f"Complete market: {self.n_binaries} scenarios = {self.n_binaries} instruments ✓")
        
    def calculate_terminal_payoffs(self):
        """Calculate option payoff at terminal nodes"""
        self.terminal_payoffs = []
        
        print(f"\nTerminal Payoffs ({self.option_type}):")
        for n_ups in range(self.T_steps + 1):
            S_T = self.nodes[(self.T_steps, n_ups)]
            if self.option_type == 'call':
                payoff = max(S_T - self.K, 0)
            else:
                payoff = max(self.K - S_T, 0)
            self.terminal_payoffs.append(payoff)
            print(f"  Terminal {n_ups} (n_ups={n_ups}, S={S_T:.2f}): ${payoff:.2f}")
    
    def calculate_binary_prices(self):
        """Calculate risk-neutral price for terminal-paying binaries"""
        self.binary_prices = []
        
        print(f"\nBinary Prices (Terminal-Paying, Risk-Neutral):")
        for n_ups in range(self.T_steps + 1):
            # Probability of reaching terminal state with n_ups up-moves
            n_downs = self.T_steps - n_ups
            prob = comb(self.T_steps, n_ups) * (self.p ** n_ups) * ((1 - self.p) ** n_downs)
            
            # Discounted price (paid at t=0, pays at terminal time T)
            price = np.exp(-self.r * self.T_steps * self.dt) * prob
            self.binary_prices.append(price)
            
            S_T = self.nodes[(self.T_steps, n_ups)]
            print(f"  Binary_{n_ups+1} (terminal n_ups={n_ups}, S={S_T:.2f}): ${price:.6f}")
    
    def evaluate_strategy(self, binary_positions):
        """
        Evaluate terminal-paying binary strategy.
        Each binary pays $1 if we reach its corresponding terminal node.
        No path-dependence!
        """
        # Total initial cost
        cost = sum(binary_positions[i] * self.binary_prices[i] for i in range(self.n_binaries))
        
        # Evaluate replication at each terminal node
        errors = []
        portfolio_values = []
        
        for terminal_n_ups in range(self.T_steps + 1):
            # Portfolio value at this terminal = position in this binary
            portfolio_value = binary_positions[terminal_n_ups]
            
            target_payoff = self.terminal_payoffs[terminal_n_ups]
            error = portfolio_value - target_payoff
            
            portfolio_values.append(portfolio_value)
            errors.append(error)
        
        errors = np.array(errors)
        total_abs_error = np.sum(np.abs(errors))
        max_error = np.max(np.abs(errors))
        mse_error = np.mean(errors ** 2)
        
        return cost, total_abs_error, max_error, mse_error, errors, portfolio_values


class CEMOptimizer:
    """Cross-Entropy Method optimizer for finding optimal binary positions"""
    def __init__(self, n_binaries, population_size=100, elite_frac=0.1):
        self.n_binaries = n_binaries
        self.population_size = population_size
        self.n_elite = max(1, int(population_size * elite_frac))
        
        # Initialize distribution (no bias - start from zero)
        self.mean = np.zeros(n_binaries)
        self.std = np.ones(n_binaries) * 30.0  # Wide initial exploration
        
    def sample_population(self):
        """Sample population from current distribution"""
        population = []
        for _ in range(self.population_size):
            sample = np.random.normal(self.mean, self.std)
            population.append(sample)
        return np.array(population)
    
    def update_distribution(self, elite_samples):
        """Update distribution based on elite samples"""
        self.mean = np.mean(elite_samples, axis=0)
        self.std = np.std(elite_samples, axis=0) + 1e-6  # Avoid collapse to zero
        
        # Minimum std to maintain some exploration
        self.std = np.maximum(self.std, 0.01)
    
    def optimize(self, env, n_iterations=100, verbose=True):
        """Run CEM optimization"""
        
        best_cost = float('inf')
        best_positions = None
        best_error = float('inf')
        
        history = {
            'costs': [],
            'errors': [],
            'mean_costs': [],
            'mean_errors': []
        }
        
        for iteration in range(n_iterations):
            # Sample population
            population = self.sample_population()
            
            # Evaluate all samples
            rewards = []
            costs = []
            errors = []
            
            for sample in population:
                cost, total_abs_error, max_error, mse_error, err_vec, portfolio_vals = env.evaluate_strategy(sample)
                
                # Reward: minimize cost + heavily penalize replication errors
                penalty = 10000 * mse_error + 5000 * max_error
                reward = -cost - penalty
                
                rewards.append(reward)
                costs.append(cost)
                errors.append(total_abs_error)
                
                # Track best solution
                if total_abs_error < best_error or (total_abs_error < 0.1 and cost < best_cost):
                    best_error = total_abs_error
                    best_cost = cost
                    best_positions = sample.copy()
            
            # Select elite samples
            elite_indices = np.argsort(rewards)[-self.n_elite:]
            elite_samples = population[elite_indices]
            
            # Update distribution
            self.update_distribution(elite_samples)
            
            # Track history
            history['costs'].append(best_cost)
            history['errors'].append(best_error)
            history['mean_costs'].append(np.mean(costs))
            history['mean_errors'].append(np.mean(errors))
            
            if verbose and iteration % 50 == 0:
                print(f"Iter {iteration:3d} | Best Cost: ${best_cost:8.4f} | Best Error: {best_error:8.6f} | "
                      f"Std: {np.mean(self.std):.4f}")
        
        return best_positions, best_cost, best_error, history


def train_cem_multistart(T_steps=2, n_restarts=10, n_iterations=300, population_size=200, option_type='call'):
    """Train CEM with multiple random restarts to escape local minima"""
    
    env = BinomialEnvironment(S0=100, K=100, r=0.05, sigma=0.2, 
                              T_steps=T_steps, option_type=option_type)
    
    print(f"\n{'='*60}")
    print(f"Training CEM with Multi-Start ({n_restarts} restarts)")
    print(f"Per restart: Population={population_size}, Iterations={n_iterations}")
    print(f"Total evaluations: {n_restarts * population_size * n_iterations:,}")
    print(f"{'='*60}\n")
    
    all_results = []
    best_overall_cost = float('inf')
    best_overall_positions = None
    best_overall_error = float('inf')
    
    for restart_idx in range(n_restarts):
        print(f"\n{'='*60}")
        print(f"RESTART {restart_idx + 1}/{n_restarts}")
        print(f"{'='*60}")
        
        # Create new CEM with different random seed
        np.random.seed(restart_idx * 12345)  # Different seed each restart
        
        cem = CEMOptimizer(n_binaries=env.n_binaries, 
                          population_size=population_size, 
                          elite_frac=0.1)
        
        best_positions, best_cost, best_error, history = cem.optimize(
            env, n_iterations=n_iterations, verbose=False
        )
        
        # Evaluate final solution
        cost, total_abs_error, max_error, mse_error, errors, portfolio_values = env.evaluate_strategy(best_positions)
        
        all_results.append({
            'restart': restart_idx + 1,
            'positions': best_positions,
            'cost': cost,
            'total_error': total_abs_error,
            'max_error': max_error,
            'mse_error': mse_error,
            'errors': errors,
            'portfolio_values': portfolio_values
        })
        
        # Track best across all restarts
        if total_abs_error < best_overall_error or (total_abs_error < 1.0 and cost < best_overall_cost):
            best_overall_error = total_abs_error
            best_overall_cost = cost
            best_overall_positions = best_positions.copy()
        
        print(f"Restart {restart_idx + 1}: Cost=${cost:.4f}, Error={total_abs_error:.6f}, MaxErr={max_error:.6f}")
    
    # Summary of all restarts
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL RESTARTS")
    print(f"{'='*60}")
    
    errors_list = [r['total_error'] for r in all_results]
    costs_list = [r['cost'] for r in all_results]
    
    print(f"\nTotal Error Statistics:")
    print(f"  Best: {min(errors_list):.6f}")
    print(f"  Worst: {max(errors_list):.6f}")
    print(f"  Mean: {np.mean(errors_list):.6f}")
    print(f"  Std: {np.std(errors_list):.6f}")
    
    print(f"\nCost Statistics:")
    print(f"  Best: ${min(costs_list):.4f}")
    print(f"  Worst: ${max(costs_list):.4f}")
    print(f"  Mean: ${np.mean(costs_list):.4f}")
    print(f"  Std: ${np.std(costs_list):.4f}")
    
    # Find and display best solution
    best_idx = np.argmin(errors_list)
    best_result = all_results[best_idx]
    
    print(f"\n{'='*60}")
    print(f"BEST SOLUTION (from Restart {best_result['restart']})")
    print(f"{'='*60}")
    
    print(f"\n*** BEST REPLICATION STRATEGY ***")
    for i in range(env.n_binaries):
        print(f"  b{i+1} = {best_result['positions'][i]:.8f}")
    
    print(f"  Initial cost: {best_result['cost']:.8f}")
    
    print(f"  Cost breakdown:")
    for i in range(env.n_binaries):
        contribution = best_result['positions'][i] * env.binary_prices[i]
        S_T = env.nodes[(env.T_steps, i)]
        print(f"    Binary_{i+1} (S_T={S_T:.2f}): {contribution:.8f} ({best_result['positions'][i]:.4f} units @ {env.binary_prices[i]:.6f})")
    
    print(f"  Replication verification:")
    for terminal_idx in range(len(best_result['errors'])):
        portfolio = best_result['portfolio_values'][terminal_idx]
        payoff = env.terminal_payoffs[terminal_idx]
        error = best_result['errors'][terminal_idx]
        status = "✓" if abs(error) < 0.01 else "❌"
        S_T = env.nodes[(env.T_steps, terminal_idx)]
        print(f"    Terminal {terminal_idx} (S_T={S_T:.2f}): Portfolio={portfolio:.8f}, Payoff={payoff:.8f}, Error={error:+.10f} {status}")
    
    print(f"  Total absolute error: {best_result['total_error']:.10f}")
    print(f"  Max absolute error: {best_result['max_error']:.10f}")
    print(f"  MSE: {best_result['mse_error']:.10f}")
    
    # Calculate theoretical price for comparison
    theoretical_price = sum(env.terminal_payoffs[i] * env.binary_prices[i] for i in range(len(env.terminal_payoffs)))
    print(f"  Cost vs theoretical: {best_result['cost']:.8f} vs {theoretical_price:.8f}")
    print(f"  Cost error: {abs(best_result['cost'] - theoretical_price):.10f}")
    print(f"  Cost error %: {100 * abs(best_result['cost'] - theoretical_price) / theoretical_price:.6f}%")
    
    return env, best_result['positions'], all_results


if __name__ == "__main__":
    print("="*60)
    print("BINOMIAL HEDGING WITH TERMINAL-PAYING BINARIES (CEM)")
    print("Complete market: N terminal nodes = N binaries")
    print("No path-dependence!")
    print("="*60)
    
    # T=2 with single run (fast, already works well)
    # env, best_pos, history = train_cem(T_steps=2, n_iterations=200, population_size=100, option_type='call')
    
    # T=5 with multi-start (10 restarts to escape local minima)
    env, best_pos, all_results = train_cem_multistart(
        T_steps=5, 
        n_restarts=10, 
        n_iterations=300, 
        population_size=200, 
        option_type='call'
    )
    
    print("\n" + "="*60)
    print("Multi-Start CEM Strategy:")
    print("  Multiple random restarts help escape local minima")
    print("  Each restart explores from different initial point")
    print("  Take best solution across all restarts")
    print("\nRecommended parameters:")
    print("  T=2: single run, population=100, iterations=200")
    print("  T=5: 10 restarts, population=200, iterations=300")
    print("  T=10: 20 restarts, population=300, iterations=500")
    print("="*60)