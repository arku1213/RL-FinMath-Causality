import numpy as np
from typing import Dict, List, Tuple

class CMAESBackwardInduction:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) with Backward Induction
    for hedging options using binary options in a binomial tree model.
    
    Evolutionary algorithm: Population-based, gradient-free optimization.
    RL-adjacent: Learns from experience, explores/exploits, adapts to fitness.
    """
    
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T_steps=2, dt=1.0,
                 generations_per_node=None, population_size=None, 
                 sigma_init=2.0):
        """
        Parameters:
        -----------
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T_steps: Number of time steps
        dt: Time increment
        generations_per_node: Number of CMA-ES generations per node (auto-scales if None)
        population_size: Population size (auto-scales if None)
        sigma_init: Initial step size for CMA-ES
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_steps = T_steps
        self.dt = dt
        self.sigma_init = sigma_init
        
        # Binomial tree parameters
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Calculate max payoff for auto-scaling
        max_stock_price = S0 * (self.u ** T_steps)
        max_payoff = max(max_stock_price - K, 0)
        
        # ADAPTIVE CLIPPING: Scale constraint range based on max payoff
        self.clip_range = max(150, max_payoff * 1.2)  # At least 150, or 120% of max payoff
        
        # AUTO-SCALE generations and population
        if generations_per_node is None:
            # CMA-ES is sample efficient but needs some iterations
            self.generations_per_node = int(100 * (1 + 0.5 * T_steps))
        else:
            self.generations_per_node = generations_per_node
        
        # Population size scales with dimension
        # Rule of thumb: 4 + floor(3 * ln(n)) where n = dimension
        self.population_size = population_size  # Will be set per node based on dimension
        
        print(f"\n{'='*60}")
        print(f"CMA-ES AUTO-SCALING FOR T={T_steps}")
        print(f"{'='*60}")
        print(f"Max Stock Price: ${max_stock_price:.2f}")
        print(f"Max Payoff: ${max_payoff:.2f}")
        print(f"Adaptive Clip Range: [{-self.clip_range:.0f}, {self.clip_range:.0f}]")
        print(f"Generations per Node: {self.generations_per_node}")
        print(f"Initial Step Size (σ): {sigma_init}")
        print(f"{'='*60}\n")
        
        # Build tree
        self.tree = self._build_tree()
        
        # Store learned hedges
        self.optimal_hedges = {}
        self.learning_history = []
        
    def _build_tree(self) -> Dict:
        """Build binomial tree structure"""
        tree = {}
        node_id = 0
        
        for t in range(self.T_steps + 1):
            for i in range(t + 1):
                S = self.S0 * (self.u ** (t - i)) * (self.d ** i)
                
                tree[node_id] = {
                    'time': t,
                    'price': S,
                    'up_moves': t - i,
                    'down_moves': i,
                    'terminal': (t == self.T_steps)
                }
                
                if t < self.T_steps:
                    tree[node_id]['child_up'] = node_id + t + 1
                    tree[node_id]['child_down'] = node_id + t + 2
                
                node_id += 1
                
        return tree
    
    def _get_num_binaries(self, node_id):
        """Get number of binary options at this node"""
        t = self.tree[node_id]['time']
        steps_remaining = self.T_steps - t
        return steps_remaining + 1 if steps_remaining > 0 else 1
    
    def _payoff_function(self, S_T):
        """Option payoff (European Call)"""
        return max(S_T - self.K, 0)
    
    def _evaluate_hedge(self, node_id, binary_positions):
        """Evaluate hedge (same as other methods)"""
        node = self.tree[node_id]
        t = node['time']
        
        if node['terminal']:
            S = node['price']
            target_payoff = self._payoff_function(S)
            replicated_payoff = binary_positions[0] if len(binary_positions) > 0 else 0
            violation = (replicated_payoff - target_payoff) ** 2
            
            discount = np.exp(-self.r * (self.T_steps - t) * self.dt)
            cost = binary_positions[0] * discount if len(binary_positions) > 0 else 0
            
            return cost, violation
        
        child_up_id = node['child_up']
        child_down_id = node['child_down']
        
        if child_up_id in self.optimal_hedges:
            value_up = np.sum(self.optimal_hedges[child_up_id])
        else:
            S_up = self.tree[child_up_id]['price']
            value_up = self._payoff_function(S_up) if self.tree[child_up_id]['terminal'] else 0
        
        if child_down_id in self.optimal_hedges:
            value_down = np.sum(self.optimal_hedges[child_down_id])
        else:
            S_down = self.tree[child_down_id]['price']
            value_down = self._payoff_function(S_down) if self.tree[child_down_id]['terminal'] else 0
        
        target_values = np.array([value_up, value_down])
        
        num_binaries = len(binary_positions)
        if num_binaries == 2:
            replicated_values = binary_positions[:2]
        else:
            mid = (num_binaries + 1) // 2
            value_up_rep = np.sum(binary_positions[:mid])
            value_down_rep = np.sum(binary_positions[mid:])
            replicated_values = np.array([value_up_rep, value_down_rep])
        
        violation = np.sum((replicated_values - target_values) ** 2)
        
        discount = np.exp(-self.r * self.dt)
        binary_price_up = discount * self.p
        binary_price_down = discount * (1 - self.p)
        
        if num_binaries == 2:
            cost = binary_positions[0] * binary_price_up + binary_positions[1] * binary_price_down
        else:
            steps_remaining = self.T_steps - t
            binary_prices = []
            for i in range(num_binaries):
                prob = (self.p ** (steps_remaining - i)) * ((1 - self.p) ** i)
                price = discount * prob
                binary_prices.append(price)
            cost = np.sum(binary_positions * np.array(binary_prices))
        
        return cost, violation
    
    def _fitness(self, node_id, action):
        """
        Fitness function (higher is better).
        Negative of cost + violation penalty + position size penalty.
        
        ADAPTIVE: Penalties scale with clip_range to remain effective.
        """
        cost, violation = self._evaluate_hedge(node_id, action)
        
        # ADAPTIVE: Position penalty scales with clip range
        # If clip_range is larger, penalty needs to be adjusted
        position_penalty_rate = 1.0 * (100 / self.clip_range)  # Normalize to 100 baseline
        position_penalty = position_penalty_rate * np.sum(np.abs(action))
        
        # ADAPTIVE: Large position penalty threshold based on clip_range
        threshold = self.clip_range * 0.8  # 80% of clip range
        max_position = np.max(np.abs(action))
        if max_position > threshold:
            large_position_penalty = 10000 * (max_position - threshold) ** 2
        else:
            large_position_penalty = 0
        
        if violation < 0.01:
            fitness = 100000 - cost - position_penalty - large_position_penalty
        else:
            fitness = -100000 * violation - position_penalty - large_position_penalty
        
        return fitness
    
    def _train_node_with_cmaes(self, node_id):
        """
        Train a node using CMA-ES.
        
        CMA-ES algorithm:
        1. Sample population from multivariate Gaussian
        2. Evaluate fitness of each individual
        3. Select top individuals
        4. Update mean and covariance matrix
        5. Repeat
        """
        print(f"\n{'='*60}")
        print(f"CMA-ES Training at Node {node_id}")
        print(f"Time: {self.tree[node_id]['time']}, Price: ${self.tree[node_id]['price']:.2f}")
        print(f"{'='*60}")
        
        n = self._get_num_binaries(node_id)  # Dimension
        
        # CMA-ES parameters (standard settings)
        lambda_ = self.population_size if self.population_size else int(4 + np.floor(3 * np.log(n)))
        mu = lambda_ // 2  # Number of parents
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Step size control parameters
        sigma = self.sigma_init
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        
        # Covariance matrix adaptation parameters
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initialize
        mean = np.zeros(n)  # Start from zero (no bias)
        C = np.eye(n)  # Covariance matrix
        pc = np.zeros(n)  # Evolution path for C
        ps = np.zeros(n)  # Evolution path for sigma
        
        eigeneval = 0
        B = np.eye(n)
        D = np.ones(n)
        
        best_fitness = -np.inf
        best_solution = np.zeros(n)
        generation_fitnesses = []
        
        total_evaluations = 0
        
        for generation in range(self.generations_per_node):
            # Generate population
            population = []
            fitnesses = []
            
            for _ in range(lambda_):
                # Sample from N(mean, sigma^2 * C)
                z = np.random.randn(n)
                y = B @ (D * z)
                x = mean + sigma * y
                
                # ADAPTIVE: Clip to adaptive range based on problem scale
                x = np.clip(x, -self.clip_range, self.clip_range)
                
                population.append(x)
                
                # Evaluate
                fitness = self._fitness(node_id, x)
                fitnesses.append(fitness)
                total_evaluations += 1
                
                # Track best
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = x.copy()
            
            population = np.array(population)
            fitnesses = np.array(fitnesses)
            generation_fitnesses.append(np.max(fitnesses))
            
            # Selection and recombination
            idx = np.argsort(fitnesses)[::-1][:mu]
            selected = population[idx]
            
            old_mean = mean.copy()
            mean = weights @ selected
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ np.linalg.inv(np.diag(D)) @ B.T @ (mean - old_mean)) / sigma
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / 
                   np.sqrt(n) < 1.4 + 2 / (n + 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma
            
            # Update covariance matrix
            artmp = (selected - old_mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(artmp[i], artmp[i])
            
            # Update step size
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update B and D (eigendecomposition every few generations)
            if generation - eigeneval > lambda_ / (c1 + cmu) / n / 10:
                eigeneval = generation
                C = np.triu(C) + np.triu(C, 1).T  # Enforce symmetry
                D, B = np.linalg.eigh(C)
                D = np.sqrt(D)
            
            # Progress
            if (generation + 1) % 20 == 0 or generation == self.generations_per_node - 1:
                cost, viol = self._evaluate_hedge(node_id, best_solution)
                print(f"Generation {generation+1}/{self.generations_per_node}")
                print(f"  Best Fitness: {best_fitness:.2f}")
                print(f"  Best Cost: ${cost:.4f}, Violation: {viol:.6f}")
                print(f"  Step Size (σ): {sigma:.3f}")
        
        # Final
        cost_final, viol_final = self._evaluate_hedge(node_id, best_solution)
        print(f"\n✓ FINAL RESULTS:")
        print(f"  Best Hedge: {best_solution}")
        print(f"  Cost: ${cost_final:.4f}")
        print(f"  Violation: {viol_final:.6f}")
        print(f"  Total Evaluations: {total_evaluations}")
        
        self.learning_history.append({
            'node_id': node_id,
            'time': self.tree[node_id]['time'],
            'best_hedge': best_solution,
            'cost': cost_final,
            'violation': viol_final,
            'fitness_history': generation_fitnesses,
            'total_evaluations': total_evaluations
        })
        
        return best_solution
    
    def backward_induction_with_cmaes(self):
        """Main algorithm: Backward Induction with CMA-ES"""
        print(f"\n{'#'*60}")
        print(f"BACKWARD INDUCTION WITH CMA-ES")
        print(f"T_steps = {self.T_steps}, Generations per node = {self.generations_per_node}")
        print(f"{'#'*60}")
        
        for t in range(self.T_steps, -1, -1):
            print(f"\n{'*'*60}")
            print(f"PROCESSING TIME STEP t = {t}")
            print(f"{'*'*60}")
            
            nodes_at_t = [node_id for node_id, info in self.tree.items() 
                         if info['time'] == t]
            
            for node_id in nodes_at_t:
                best_hedge = self._train_node_with_cmaes(node_id)
                self.optimal_hedges[node_id] = best_hedge
        
        print(f"\n{'#'*60}")
        print(f"BACKWARD INDUCTION COMPLETE!")
        print(f"{'#'*60}")
        
        return self.optimal_hedges
    
    def summarize_results(self):
        """Print summary"""
        print(f"\n{'='*60}")
        print(f"LEARNED HEDGING STRATEGY SUMMARY")
        print(f"{'='*60}")
        
        total_violation = 0
        initial_cost = 0
        total_evaluations = sum(h['total_evaluations'] for h in self.learning_history)
        
        for t in range(self.T_steps + 1):
            nodes_at_t = [node_id for node_id, info in self.tree.items() 
                         if info['time'] == t]
            
            print(f"\nTime t = {t}:")
            for node_id in nodes_at_t:
                hedge = self.optimal_hedges[node_id]
                cost, viol = self._evaluate_hedge(node_id, hedge)
                
                print(f"  Node {node_id} (S=${self.tree[node_id]['price']:.2f}):")
                print(f"    Hedge: {hedge}")
                print(f"    Cost: ${cost:.4f}")
                print(f"    Violation: {viol:.6f}")
                
                if t == 0:
                    initial_cost = cost
                total_violation += viol
        
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Initial Hedge Cost (t=0): ${initial_cost:.4f}")
        print(f"  Total Violation: {total_violation:.6f}")
        print(f"  Total Evaluations: {total_evaluations}")
        print(f"{'='*60}")


# Run
if __name__ == "__main__":
    print("Initializing CMA-ES + Backward Induction...")
    
    cmaes_bi = CMAESBackwardInduction(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T_steps=5,  # Just change this!
        dt=1.0,
        sigma_init=2.0  # Initial step size for CMA-ES
        # generations_per_node and population_size auto-scale
    )
    
    print("\nStarting Backward Induction with CMA-ES...\n")
    optimal_hedges = cmaes_bi.backward_induction_with_cmaes()
    
    cmaes_bi.summarize_results()
    
    print("\n✓ Algorithm complete!")
    print("\nCMA-ES: Evolutionary strategy with covariance adaptation!")