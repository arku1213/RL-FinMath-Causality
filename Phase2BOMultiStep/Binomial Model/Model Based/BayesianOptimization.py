import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm
from scipy.spatial.distance import cdist

class BayesianOptimizationBackwardInduction:
    """
    Bayesian Optimization (Gaussian Processes) with Backward Induction
    for hedging options using binary options in a binomial tree model.
    
    NOT RL: This is black-box optimization using GP + acquisition functions.
    Hybrid: BO + Dynamic Programming structure.
    """
    
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T_steps=2, dt=1.0,
                 iterations_per_node=None, acquisition='UCB', xi=0.01,
                 length_scale=5.0, noise=0.1):
        """
        Parameters:
        -----------
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T_steps: Number of time steps
        dt: Time increment
        iterations_per_node: Number of BO iterations per node (auto-scales if None)
        acquisition: Acquisition function ('UCB', 'EI', 'PI')
        xi: Exploration parameter for EI/PI
        length_scale: GP kernel length scale (controls smoothness)
        noise: Observation noise for GP
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_steps = T_steps
        self.dt = dt
        self.acquisition_type = acquisition
        self.xi = xi
        self.length_scale = length_scale
        self.noise = noise
        
        # Binomial tree parameters
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Calculate max payoff for auto-scaling
        max_stock_price = S0 * (self.u ** T_steps)
        max_payoff = max(max_stock_price - K, 0)
        
        # AUTO-SCALE iterations (BO is very sample efficient!)
        if iterations_per_node is None:
            # BO needs FAR fewer iterations than RL methods
            self.iterations_per_node = int(200 * (1 + 0.3 * T_steps))
        else:
            self.iterations_per_node = iterations_per_node
        
        print(f"\n{'='*60}")
        print(f"BAYESIAN OPTIMIZATION AUTO-SCALING FOR T={T_steps}")
        print(f"{'='*60}")
        print(f"Max Stock Price: ${max_stock_price:.2f}")
        print(f"Max Payoff: ${max_payoff:.2f}")
        print(f"Iterations per Node: {self.iterations_per_node}")
        print(f"Acquisition Function: {acquisition}")
        print(f"GP Length Scale: {length_scale}")
        print(f"{'='*60}\n")
        
        # Build tree
        self.tree = self._build_tree()
        
        # Gaussian Process data per node
        self.gp_data = {}
        self._initialize_gp()
        
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
    
    def _initialize_gp(self):
        """Initialize GP data storage for each node"""
        for node_id in self.tree.keys():
            self.gp_data[node_id] = {
                'X': [],  # Observed actions (hedge positions)
                'y': []   # Observed rewards
            }
    
    def _get_num_binaries(self, node_id):
        """Get number of binary options at this node"""
        t = self.tree[node_id]['time']
        steps_remaining = self.T_steps - t
        return steps_remaining + 1 if steps_remaining > 0 else 1
    
    def _payoff_function(self, S_T):
        """Option payoff (European Call)"""
        return max(S_T - self.K, 0)
    
    def _rbf_kernel(self, X1, X2):
        """
        Radial Basis Function (RBF) kernel for GP.
        K(x1, x2) = exp(-||x1 - x2||² / (2 * length_scale²))
        """
        dists = cdist(X1, X2, metric='euclidean')
        K = np.exp(-0.5 * (dists / self.length_scale) ** 2)
        return K
    
    def _gp_predict(self, node_id, X_new):
        """
        Gaussian Process prediction.
        Returns mean and std at new points X_new.
        """
        gp = self.gp_data[node_id]
        
        if len(gp['X']) == 0:
            # No data yet: prior is zero mean, unit variance
            n = X_new.shape[0]
            return np.zeros(n), np.ones(n)
        
        X_train = np.array(gp['X'])
        y_train = np.array(gp['y'])
        
        # Kernel matrices
        K = self._rbf_kernel(X_train, X_train)
        K += self.noise * np.eye(len(X_train))  # Add noise
        
        K_star = self._rbf_kernel(X_train, X_new)
        K_star_star = self._rbf_kernel(X_new, X_new)
        
        # GP posterior
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            
            # Mean
            mu = K_star.T @ alpha
            
            # Variance
            v = np.linalg.solve(L, K_star)
            var = np.diag(K_star_star) - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-8)  # Numerical stability
            
            return mu, np.sqrt(var)
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            n = X_new.shape[0]
            return np.zeros(n), np.ones(n)
    
    def _acquisition_ucb(self, mu, std, kappa=2.0):
        """
        Upper Confidence Bound (UCB) acquisition function.
        UCB = μ + κ × σ
        """
        return mu + kappa * std
    
    def _acquisition_ei(self, mu, std, y_best):
        """
        Expected Improvement (EI) acquisition function.
        """
        improvement = mu - y_best - self.xi
        Z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        return ei
    
    def _acquisition_pi(self, mu, std, y_best):
        """
        Probability of Improvement (PI) acquisition function.
        """
        improvement = mu - y_best - self.xi
        Z = improvement / (std + 1e-8)
        pi = norm.cdf(Z)
        return pi
    
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
    
    def _get_reward(self, node_id, action):
        """Compute reward"""
        cost, violation = self._evaluate_hedge(node_id, action)
        
        if violation < 0.01:
            reward = 100000 - cost
        else:
            reward = -100000 * violation
        
        return reward, cost, violation
    
    def _train_node_with_bo(self, node_id):
        """Train a node using Bayesian Optimization"""
        print(f"\n{'='*60}")
        print(f"Bayesian Optimization at Node {node_id}")
        print(f"Time: {self.tree[node_id]['time']}, Price: ${self.tree[node_id]['price']:.2f}")
        print(f"{'='*60}")
        
        num_binaries = self._get_num_binaries(node_id)
        best_action = np.zeros(num_binaries)
        best_reward = -np.inf
        iteration_rewards = []
        
        gp = self.gp_data[node_id]
        
        # Initial random samples (more exploration for higher dimensions)
        num_binaries = self._get_num_binaries(node_id)
        n_initial = min(50 + 10 * num_binaries, self.iterations_per_node // 5)
        for _ in range(n_initial):
            action = np.random.randn(num_binaries) * 20.0
            reward, cost, violation = self._get_reward(node_id, action)
            
            gp['X'].append(action)
            gp['y'].append(reward)
            
            if reward > best_reward:
                best_reward = reward
                best_action = action.copy()
            
            iteration_rewards.append(reward)
        
        # Bayesian Optimization loop
        for iteration in range(n_initial, self.iterations_per_node):
            # Generate more candidate points for better coverage
            n_candidates = 200 + 50 * num_binaries
            X_candidates = np.random.randn(n_candidates, num_binaries) * 30.0
            
            # GP prediction
            mu, std = self._gp_predict(node_id, X_candidates)
            
            # Acquisition function
            if self.acquisition_type == 'UCB':
                acq_values = self._acquisition_ucb(mu, std)
            elif self.acquisition_type == 'EI':
                y_best = max(gp['y']) if gp['y'] else 0
                acq_values = self._acquisition_ei(mu, std, y_best)
            elif self.acquisition_type == 'PI':
                y_best = max(gp['y']) if gp['y'] else 0
                acq_values = self._acquisition_pi(mu, std, y_best)
            
            # Select point with highest acquisition value
            best_idx = np.argmax(acq_values)
            next_action = X_candidates[best_idx]
            
            # Evaluate
            reward, cost, violation = self._get_reward(node_id, next_action)
            
            # Update GP
            gp['X'].append(next_action)
            gp['y'].append(reward)
            
            # Track best
            if reward > best_reward:
                best_reward = reward
                best_action = next_action.copy()
            
            iteration_rewards.append(reward)
            
            # Progress (report more frequently)
            if (iteration + 1) % 200 == 0:
                recent = iteration_rewards[-200:]
                cost_best, viol_best = self._evaluate_hedge(node_id, best_action)
                print(f"Iteration {iteration+1}/{self.iterations_per_node}")
                print(f"  Avg Reward (last 200): {np.mean(recent):.2f}")
                print(f"  Best Reward: {best_reward:.2f}")
                print(f"  Best Cost: ${cost_best:.4f}, Violation: {viol_best:.6f}")
        
        # Final
        cost_final, viol_final = self._evaluate_hedge(node_id, best_action)
        print(f"\n✓ FINAL RESULTS:")
        print(f"  Best Hedge: {best_action}")
        print(f"  Cost: ${cost_final:.4f}")
        print(f"  Violation: {viol_final:.6f}")
        print(f"  Total Evaluations: {len(gp['X'])}")
        
        self.learning_history.append({
            'node_id': node_id,
            'time': self.tree[node_id]['time'],
            'best_hedge': best_action,
            'cost': cost_final,
            'violation': viol_final,
            'reward_history': iteration_rewards
        })
        
        return best_action
    
    def backward_induction_with_bo(self):
        """Main algorithm: Backward Induction with Bayesian Optimization"""
        print(f"\n{'#'*60}")
        print(f"BACKWARD INDUCTION WITH BAYESIAN OPTIMIZATION")
        print(f"T_steps = {self.T_steps}, Iterations per node = {self.iterations_per_node}")
        print(f"{'#'*60}")
        
        for t in range(self.T_steps, -1, -1):
            print(f"\n{'*'*60}")
            print(f"PROCESSING TIME STEP t = {t}")
            print(f"{'*'*60}")
            
            nodes_at_t = [node_id for node_id, info in self.tree.items() 
                         if info['time'] == t]
            
            for node_id in nodes_at_t:
                best_hedge = self._train_node_with_bo(node_id)
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
        print(f"{'='*60}")


# Run
if __name__ == "__main__":
    print("Initializing Bayesian Optimization + Backward Induction...")
    
    bo_bi = BayesianOptimizationBackwardInduction(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T_steps=3,  # Just change this!
        dt=1.0,
        acquisition='UCB'  # Can try 'EI' or 'PI'
        # iterations_per_node auto-scales (very sample efficient!)
    )
    
    print("\nStarting Backward Induction with Bayesian Optimization...\n")
    optimal_hedges = bo_bi.backward_induction_with_bo()
    
    bo_bi.summarize_results()
    
    print("\n✓ Algorithm complete!")
    print("\nBayesian Optimization: Black-box optimization with GP!")