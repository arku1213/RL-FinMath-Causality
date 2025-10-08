import numpy as np
from math import comb

class BinomialEnvironment:
    """
    Binomial tree for forward dynamic programming.
    We'll solve each node's 1-step problem independently using CEM.
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
        
        print(f"\n{'='*60}")
        print(f"Forward DP Environment (T={T_steps})")
        print(f"{'='*60}")
        print(f"Nodes to solve: {sum(t+1 for t in range(T_steps))}")
        print(f"Each node: 1-step binomial (2 binaries)")
        print(f"Terminal nodes: {T_steps + 1}")
        
    def build_tree(self):
        """Build all nodes"""
        self.nodes = {}
        for t in range(self.T_steps + 1):
            for n_ups in range(t + 1):
                price = self.S0 * (self.u ** n_ups) * (self.d ** (t - n_ups))
                self.nodes[(t, n_ups)] = price
    
    def calculate_terminal_payoffs(self):
        """Calculate terminal payoffs"""
        self.terminal_payoffs = {}
        for n_ups in range(self.T_steps + 1):
            S_T = self.nodes[(self.T_steps, n_ups)]
            if self.option_type == 'call':
                payoff = max(S_T - self.K, 0)
            else:
                payoff = max(self.K - S_T, 0)
            self.terminal_payoffs[n_ups] = payoff
        
        print(f"\nTerminal Payoffs:")
        for n_ups, payoff in self.terminal_payoffs.items():
            S_T = self.nodes[(self.T_steps, n_ups)]
            print(f"  Terminal {n_ups} (S={S_T:.2f}): ${payoff:.2f}")


class CEM1Step:
    """CEM optimizer for solving 1-step binomial subproblems"""
    def __init__(self, population_size=100, elite_frac=0.1, n_iterations=100):
        self.population_size = population_size
        self.n_elite = max(1, int(population_size * elite_frac))
        self.n_iterations = n_iterations
    
    def solve(self, target_up, target_down, price_up, price_down, verbose=False):
        """
        Solve 1-step binomial:
        Find b_up and b_down such that:
        - b_up replicates target_up
        - b_down replicates target_down
        Minimize: cost = b_up * price_up + b_down * price_down
        """
        # Initialize distribution
        mean = np.array([target_up, target_down])  # Smart initialization near targets
        std = np.array([abs(target_up) * 0.5 + 10, abs(target_down) * 0.5 + 10])
        
        best_solution = None
        best_cost = float('inf')
        best_error = float('inf')
        
        for iteration in range(self.n_iterations):
            # Sample population
            population = []
            for _ in range(self.population_size):
                sample = np.random.normal(mean, std)
                population.append(sample)
            population = np.array(population)
            
            # Evaluate
            costs = []
            errors = []
            rewards = []
            
            for sample in population:
                b_up, b_down = sample
                
                # Cost
                cost = b_up * price_up + b_down * price_down
                
                # Replication errors
                error_up = abs(b_up - target_up)
                error_down = abs(b_down - target_down)
                total_error = error_up + error_down
                
                # Reward: minimize cost + heavily penalize errors
                penalty = 10000 * (error_up**2 + error_down**2) + 5000 * max(error_up, error_down)
                reward = -cost - penalty
                
                costs.append(cost)
                errors.append(total_error)
                rewards.append(reward)
                
                # Track best
                if total_error < best_error or (total_error < 0.1 and cost < best_cost):
                    best_error = total_error
                    best_cost = cost
                    best_solution = sample.copy()
            
            # Select elites
            elite_indices = np.argsort(rewards)[-self.n_elite:]
            elite_samples = population[elite_indices]
            
            # Update distribution
            mean = np.mean(elite_samples, axis=0)
            std = np.std(elite_samples, axis=0) + 1e-6
            std = np.maximum(std, 0.01)  # Minimum std
            
            if verbose and iteration % 20 == 0:
                print(f"  Iter {iteration:3d} | Cost: ${best_cost:8.4f} | Error: {best_error:8.6f}")
        
        return best_solution, best_cost, best_error


def solve_forward_dp(env, verbose=True):
    """
    TRUE Forward Dynamic Programming with CEM at each node.
    
    Process:
    1. Start at T=0
    2. At each node, solve 1-step problem: "replicate the TERMINAL payoffs reachable from here"
    3. Move forward in time
    4. No backward induction - purely forward!
    """
    
    print(f"\n{'='*60}")
    print("FORWARD DP: Solving Each Node with CEM (FORWARD ONLY)")
    print(f"{'='*60}\n")
    
    cem = CEM1Step(population_size=100, elite_frac=0.1, n_iterations=100)
    
    # Store solutions: {(t, n_ups): {'b_up': x, 'b_down': y, 'cost': z}}
    solutions = {}
    
    # Work FORWARD from T=0 to T-1
    for t in range(env.T_steps):
        print(f"{'='*60}")
        print(f"Solving Time t={t} (all nodes at this time)")
        print(f"{'='*60}")
        
        for n_ups in range(t + 1):
            S_t = env.nodes[(t, n_ups)]
            
            # At each node, we solve: "What binaries do I buy to replicate 
            # the TERMINAL PAYOFFS reachable from my next-step nodes?"
            
            # From (t, n_ups), we can reach:
            # - (t+1, n_ups) via down move
            # - (t+1, n_ups+1) via up move
            
            # For each of these nodes, what are the TERMINAL payoffs reachable?
            # We need to calculate the "continuation value" at each next node
            
            # Down node (t+1, n_ups):
            target_down = calculate_continuation_value(env, t+1, n_ups)
            
            # Up node (t+1, n_ups+1):
            target_up = calculate_continuation_value(env, t+1, n_ups+1)
            
            # Binary prices (risk-neutral, 1-step)
            price_up = env.p * np.exp(-env.r * env.dt)
            price_down = (1 - env.p) * np.exp(-env.r * env.dt)
            
            # Solve 1-step problem with CEM
            if verbose:
                print(f"\nNode ({t}, {n_ups}) @ S={S_t:.2f}")
                print(f"  Targets: Up=${target_up:.4f}, Down=${target_down:.4f}")
                print(f"  Binary prices: Up=${price_up:.6f}, Down=${price_down:.6f}")
            
            solution, cost, error = cem.solve(target_up, target_down, 
                                             price_up, price_down, 
                                             verbose=False)
            
            b_up, b_down = solution
            
            solutions[(t, n_ups)] = {
                'b_up': b_up,
                'b_down': b_down,
                'cost': cost,
                'error': error,
                'target_up': target_up,
                'target_down': target_down
            }
            
            if verbose:
                print(f"  Solution: b_up={b_up:.4f}, b_down={b_down:.4f}")
                print(f"  Cost: ${cost:.4f}, Error: {error:.6f}")
    
    # Evaluate full strategy
    print(f"\n{'='*60}")
    print("FULL STRATEGY EVALUATION")
    print(f"{'='*60}")
    
    total_cost, errors, path_details = evaluate_full_strategy(env, solutions)
    
    print(f"\nTotal Initial Cost: ${total_cost:.8f}")
    print(f"\nReplication Verification (All Paths):")
    
    for i, (path, portfolio_val, target, error) in enumerate(path_details):
        path_str = '→'.join([f"({t},{n})" for t, n in path])
        status = "✓" if abs(error) < 0.01 else "❌"
        print(f"  Path {i+1}: {path_str}")
        print(f"    Portfolio=${portfolio_val:.6f}, Target=${target:.6f}, Error={error:+.8f} {status}")
    
    total_abs_error = sum(abs(e) for _, _, _, e in path_details)
    max_error = max(abs(e) for _, _, _, e in path_details)
    
    print(f"\nTotal Absolute Error: {total_abs_error:.10f}")
    print(f"Max Absolute Error: {max_error:.10f}")
    
    # Theoretical price
    theoretical = sum(env.terminal_payoffs[i] * 
                     comb(env.T_steps, i) * (env.p ** i) * ((1-env.p) ** (env.T_steps - i)) *
                     np.exp(-env.r * env.T_steps * env.dt)
                     for i in range(env.T_steps + 1))
    
    print(f"\nCost vs Theoretical: ${total_cost:.8f} vs ${theoretical:.8f}")
    print(f"Cost Error: ${abs(total_cost - theoretical):.8f} ({100*abs(total_cost - theoretical)/theoretical:.4f}%)")
    
    return solutions, total_cost, total_abs_error


def calculate_continuation_value(env, t, n_ups):
    """
    Calculate the expected value from node (t, n_ups) to terminal.
    This is what we need to replicate at this node.
    
    For pure forward, we just need: "What's the expected terminal payoff 
    reachable from here?"
    """
    if t == env.T_steps:
        # Already at terminal
        return env.terminal_payoffs[n_ups]
    
    # Calculate expected payoff from all terminal nodes reachable from (t, n_ups)
    # Using risk-neutral probabilities
    
    remaining_steps = env.T_steps - t
    value = 0
    
    for additional_ups in range(remaining_steps + 1):
        terminal_n_ups = n_ups + additional_ups
        terminal_payoff = env.terminal_payoffs[terminal_n_ups]
        
        # Probability of getting 'additional_ups' ups in remaining_steps
        prob = comb(remaining_steps, additional_ups) * \
               (env.p ** additional_ups) * \
               ((1 - env.p) ** (remaining_steps - additional_ups))
        
        value += prob * terminal_payoff
    
    # Discount back to time t
    value *= np.exp(-env.r * remaining_steps * env.dt)
    
    return value


def evaluate_full_strategy(env, solutions):
    """
    Evaluate the full strategy by simulating all possible paths.
    """
    def generate_all_paths(t, n_ups, path):
        if t == env.T_steps:
            return [path]
        
        paths = []
        # Down
        paths.extend(generate_all_paths(t+1, n_ups, path + [(t+1, n_ups)]))
        # Up
        paths.extend(generate_all_paths(t+1, n_ups+1, path + [(t+1, n_ups+1)]))
        return paths
    
    all_paths = generate_all_paths(0, 0, [(0, 0)])
    
    path_details = []
    
    for path in all_paths:
        portfolio_value = 0
        cost_incurred = 0
        
        # Follow this path and accumulate portfolio
        for i in range(len(path) - 1):
            t, n_ups = path[i]
            next_t, next_n_ups = path[i + 1]
            
            if (t, n_ups) not in solutions:
                continue
            
            sol = solutions[(t, n_ups)]
            
            # Determine which binary pays
            if next_n_ups > n_ups:
                # Went up
                portfolio_value += sol['b_up']
            else:
                # Went down
                portfolio_value += sol['b_down']
            
            # Cost incurred at this step (only at t=0 for total cost)
            if t == 0:
                cost_incurred = sol['cost']
        
        # Terminal payoff
        terminal_n_ups = path[-1][1]
        target_payoff = env.terminal_payoffs[terminal_n_ups]
        error = portfolio_value - target_payoff
        
        path_details.append((path, portfolio_value, target_payoff, error))
    
    # Total cost is just the cost at root
    total_cost = solutions[(0, 0)]['cost']
    
    errors = [e for _, _, _, e in path_details]
    
    return total_cost, errors, path_details


if __name__ == "__main__":
    print("="*60)
    print("FORWARD DP WITH CEM AT EACH NODE")
    print("Pure RL approach (CEM) solving local 1-step problems")
    print("No bias, no Deep RL, just CEM chaining")
    print("="*60)
    
    # T=2 first
    print("\n" + "="*60)
    print("T=2 BINOMIAL")
    print("="*60)
    env_t2 = BinomialEnvironment(T_steps=2, option_type='call')
    solutions_t2, cost_t2, error_t2 = solve_forward_dp(env_t2, verbose=True)
    
    # Then T=5
    print("\n\n" + "="*60)
    print("T=5 BINOMIAL")
    print("="*60)
    env_t5 = BinomialEnvironment(T_steps=5, option_type='call')
    solutions_t5, cost_t5, error_t5 = solve_forward_dp(env_t5, verbose=False)  # Less verbose for T=5