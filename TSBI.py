import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class ThompsonSamplingBackwardInduction:
    """
    Hybrid RL approach combining Thompson Sampling with Backward Induction
    for hedging options using binary options in a binomial tree model.
    
    Model-free: Learns optimal hedges through Bayesian exploration.
    """
    
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T_steps=2, dt=1.0, 
                 episodes_per_node=5000, prior_mean=0.0, prior_std=15.0):
        """
        Parameters:
        -----------
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T_steps: Number of time steps
        dt: Time increment
        episodes_per_node: Number of Thompson Sampling episodes per node
        prior_mean: Prior mean for binary positions
        prior_std: Prior std for binary positions (wide prior = more exploration)
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_steps = T_steps
        self.dt = dt
        self.episodes_per_node = episodes_per_node
        
        # Binomial tree parameters
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Build tree structure
        self.tree = self._build_tree()
        self.num_nodes = sum(t + 1 for t in range(T_steps + 1))
        
        # Initialize Bayesian posteriors for each node (Gaussian)
        # Each node has its own belief over binary positions
        self.posteriors = {}  # {node_id: {'mean': array, 'cov': matrix}}
        self._initialize_posteriors(prior_mean, prior_std)
        
        # Store learned optimal hedges
        self.optimal_hedges = {}
        
        # Track learning progress
        self.learning_history = []
        
    def _build_tree(self) -> Dict:
        """Build binomial tree structure with node IDs and stock prices"""
        tree = {}
        node_id = 0
        
        for t in range(self.T_steps + 1):
            for i in range(t + 1):
                # Stock price at this node
                S = self.S0 * (self.u ** (t - i)) * (self.d ** i)
                
                # Node identification
                tree[node_id] = {
                    'time': t,
                    'price': S,
                    'up_moves': t - i,
                    'down_moves': i,
                    'terminal': (t == self.T_steps)
                }
                
                # Add children (if not terminal)
                if t < self.T_steps:
                    tree[node_id]['child_up'] = node_id + t + 1
                    tree[node_id]['child_down'] = node_id + t + 2
                
                node_id += 1
                
        return tree
    
    def _initialize_posteriors(self, prior_mean, prior_std):
        """Initialize Gaussian posteriors for each node"""
        for node_id in self.tree.keys():
            num_binaries = self._get_num_binaries(node_id)
            
            self.posteriors[node_id] = {
                'mean': np.ones(num_binaries) * prior_mean,
                'cov': np.eye(num_binaries) * (prior_std ** 2),
                'observations': []
            }
    
    def _get_num_binaries(self, node_id):
        """Get number of binary options available at this node"""
        t = self.tree[node_id]['time']
        steps_remaining = self.T_steps - t
        # Number of terminal states reachable from this node
        return steps_remaining + 1 if steps_remaining > 0 else 1
    
    def _payoff_function(self, S_T):
        """Option payoff at maturity (European Call by default)"""
        return max(S_T - self.K, 0)
    
    def _get_terminal_states_from_node(self, node_id):
        """Get all terminal stock prices reachable from a given node"""
        t = self.tree[node_id]['time']
        S = self.tree[node_id]['price']
        steps_remaining = self.T_steps - t
        
        terminal_prices = []
        for i in range(steps_remaining + 1):
            S_T = S * (self.u ** (steps_remaining - i)) * (self.d ** i)
            terminal_prices.append(S_T)
            
        return np.array(terminal_prices)
    
    def _evaluate_hedge(self, node_id, binary_positions):
        """
        Evaluate quality of hedge at a node.
        For backward induction: match value at child nodes, not terminal payoffs.
        
        Returns:
        --------
        cost: Cost of the hedge
        violation: Sum of squared constraint violations
        """
        node = self.tree[node_id]
        t = node['time']
        
        # If terminal node, match option payoff directly
        if node['terminal']:
            S = node['price']
            target_payoff = self._payoff_function(S)
            # For terminal node, we should have 1 binary that pays 1
            replicated_payoff = binary_positions[0] if len(binary_positions) > 0 else 0
            violation = (replicated_payoff - target_payoff) ** 2
            
            # Cost: discounted price of binary
            discount = np.exp(-self.r * (self.T_steps - t) * self.dt)
            cost = binary_positions[0] * discount if len(binary_positions) > 0 else 0
            
            return cost, violation
        
        # For non-terminal nodes: match continuation values at child nodes
        child_up_id = node['child_up']
        child_down_id = node['child_down']
        
        # Get continuation values from child nodes (if already computed)
        # Otherwise use terminal payoffs
        if child_up_id in self.optimal_hedges:
            value_up = self._get_node_value(child_up_id)
        else:
            # Use terminal payoffs from this node
            terminal_prices = self._get_terminal_states_from_node(node_id)
            value_up = self._payoff_function(terminal_prices[0])
        
        if child_down_id in self.optimal_hedges:
            value_down = self._get_node_value(child_down_id)
        else:
            terminal_prices = self._get_terminal_states_from_node(node_id)
            value_down = self._payoff_function(terminal_prices[-1])
        
        target_values = np.array([value_up, value_down])
        
        # Binaries should replicate these values
        # Assume: binary_positions[0] for up state, binary_positions[1] for down state
        if len(binary_positions) == 2:
            replicated_values = binary_positions[:2]
        else:
            # Handle case where we have more binaries (for terminal states)
            terminal_prices = self._get_terminal_states_from_node(node_id)
            num_terminal = len(terminal_prices)
            
            # Aggregate binary positions to up/down values
            # Up path: first half of terminals, Down path: second half
            mid = (num_terminal + 1) // 2
            value_up_rep = np.sum(binary_positions[:mid])
            value_down_rep = np.sum(binary_positions[mid:])
            replicated_values = np.array([value_up_rep, value_down_rep])
        
        violation = np.sum((replicated_values - target_values) ** 2)
        
        # Cost: discounted risk-neutral price of binaries
        discount = np.exp(-self.r * self.dt)
        
        # Binary prices using risk-neutral probabilities
        binary_price_up = discount * self.p
        binary_price_down = discount * (1 - self.p)
        
        if len(binary_positions) == 2:
            cost = binary_positions[0] * binary_price_up + binary_positions[1] * binary_price_down
        else:
            # Price each terminal binary
            steps_remaining = self.T_steps - t
            terminal_prices = self._get_terminal_states_from_node(node_id)
            binary_prices = []
            for i in range(len(terminal_prices)):
                prob = (self.p ** (steps_remaining - i)) * ((1 - self.p) ** i)
                price = discount * prob
                binary_prices.append(price)
            cost = np.sum(binary_positions * np.array(binary_prices))
        
        return cost, violation
    
    def _get_node_value(self, node_id):
        """Get the value at a node (sum of optimal hedge positions)"""
        if node_id in self.optimal_hedges:
            return np.sum(self.optimal_hedges[node_id])
        return 0.0
    
    def _thompson_sampling_at_node(self, node_id):
        """
        Run Thompson Sampling at a specific node to learn optimal hedge.
        
        Returns:
        --------
        best_hedge: Best binary positions found
        best_reward: Best reward achieved
        """
        print(f"\n{'='*60}")
        print(f"Thompson Sampling at Node {node_id}")
        print(f"Time: {self.tree[node_id]['time']}, Price: ${self.tree[node_id]['price']:.2f}")
        print(f"{'='*60}")
        
        posterior = self.posteriors[node_id]
        best_hedge = None
        best_reward = -np.inf
        episode_rewards = []
        
        for episode in range(self.episodes_per_node):
            # Sample hedge from posterior (Thompson Sampling step)
            mean = posterior['mean']
            cov = posterior['cov']
            sampled_hedge = np.random.multivariate_normal(mean, cov)
            
            # Evaluate sampled hedge
            cost, violation = self._evaluate_hedge(node_id, sampled_hedge)
            
            # Reward: heavily penalize violations, lightly penalize cost
            # Make violation the PRIMARY objective
            if violation < 0.01:  # Nearly perfect replication
                reward = 100000 - cost  # Minimize cost among feasible solutions
            else:
                reward = -100000 * violation  # Focus purely on reducing violations
            
            # Track best
            if reward > best_reward:
                best_reward = reward
                best_hedge = sampled_hedge.copy()
            
            episode_rewards.append(reward)
            
            # Bayesian update of posterior (simplified: running mean and cov)
            posterior['observations'].append((sampled_hedge, reward))
            
            # Update posterior every 50 episodes (batch update for stability)
            if (episode + 1) % 100 == 0:
                self._update_posterior(node_id)
            
            # Progress reporting
            if (episode + 1) % 200 == 0:
                recent_rewards = episode_rewards[-200:]
                cost_best, viol_best = self._evaluate_hedge(node_id, best_hedge)
                print(f"Episode {episode+1}/{self.episodes_per_node}")
                print(f"  Avg Reward (last 200): {np.mean(recent_rewards):.2f}")
                print(f"  Best Reward: {best_reward:.2f}")
                print(f"  Best Cost: ${cost_best:.4f}, Violation: {viol_best:.6f}")
        
        # Final update
        self._update_posterior(node_id)
        
        # Store results
        cost_final, viol_final = self._evaluate_hedge(node_id, best_hedge)
        print(f"\n✓ FINAL RESULTS:")
        print(f"  Best Hedge: {best_hedge}")
        print(f"  Cost: ${cost_final:.4f}")
        print(f"  Violation: {viol_final:.6f}")
        
        self.learning_history.append({
            'node_id': node_id,
            'time': self.tree[node_id]['time'],
            'best_hedge': best_hedge,
            'cost': cost_final,
            'violation': viol_final,
            'reward_history': episode_rewards
        })
        
        return best_hedge, best_reward
    
    def _update_posterior(self, node_id):
        """Update posterior distribution using observed rewards (weighted by reward)"""
        posterior = self.posteriors[node_id]
        observations = posterior['observations']
        
        if len(observations) == 0:
            return
        
        # Weight samples by their rewards (softmax weighting)
        hedges = np.array([obs[0] for obs in observations])
        rewards = np.array([obs[1] for obs in observations])
        
        # Normalize rewards to weights (higher reward = higher weight)
        rewards_shifted = rewards - np.min(rewards) + 1e-6
        weights = np.exp(rewards_shifted / np.std(rewards_shifted) if np.std(rewards_shifted) > 0 else rewards_shifted)
        weights = weights / np.sum(weights)
        
        # Weighted mean and covariance
        new_mean = np.sum(hedges * weights[:, np.newaxis], axis=0)
        
        # Weighted covariance
        centered = hedges - new_mean
        new_cov = np.dot(centered.T * weights, centered)
        new_cov += np.eye(len(new_mean)) * 0.5  # More exploration noise
        
        # Update (moving average with old posterior)
        alpha = 0.7  # Higher learning rate for faster convergence
        posterior['mean'] = (1 - alpha) * posterior['mean'] + alpha * new_mean
        posterior['cov'] = (1 - alpha) * posterior['cov'] + alpha * new_cov
    
    def backward_induction_with_ts(self):
        """
        Main algorithm: Backward Induction with Thompson Sampling at each node.
        
        Works backward from terminal nodes (T) to root (0).
        """
        print(f"\n{'#'*60}")
        print(f"BACKWARD INDUCTION WITH THOMPSON SAMPLING")
        print(f"T_steps = {self.T_steps}, Episodes per node = {self.episodes_per_node}")
        print(f"{'#'*60}")
        
        # Process nodes backward (from T to 0)
        for t in range(self.T_steps, -1, -1):
            print(f"\n{'*'*60}")
            print(f"PROCESSING TIME STEP t = {t}")
            print(f"{'*'*60}")
            
            # Find all nodes at time t
            nodes_at_t = [node_id for node_id, info in self.tree.items() 
                         if info['time'] == t]
            
            for node_id in nodes_at_t:
                # Run Thompson Sampling at this node
                best_hedge, best_reward = self._thompson_sampling_at_node(node_id)
                self.optimal_hedges[node_id] = best_hedge
        
        print(f"\n{'#'*60}")
        print(f"BACKWARD INDUCTION COMPLETE!")
        print(f"{'#'*60}")
        
        return self.optimal_hedges
    
    def plot_learning_curves(self):
        """Plot learning curves for each node"""
        num_nodes = len(self.learning_history)
        fig, axes = plt.subplots(num_nodes, 1, figsize=(12, 4 * num_nodes))
        
        if num_nodes == 1:
            axes = [axes]
        
        for idx, history in enumerate(self.learning_history):
            ax = axes[idx]
            rewards = history['reward_history']
            
            # Plot raw rewards
            ax.plot(rewards, alpha=0.3, label='Episode Reward')
            
            # Plot moving average
            window = 50
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 
                   linewidth=2, label=f'Moving Avg (window={window})')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title(f"Node {history['node_id']} (t={history['time']}, " + 
                        f"Final Cost=${history['cost']:.4f}, Violation={history['violation']:.6f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ts_backward_induction_learning.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✓ Learning curves saved to 'ts_backward_induction_learning.png'")
    
    def summarize_results(self):
        """Print summary of learned hedging strategy"""
        print(f"\n{'='*60}")
        print(f"LEARNED HEDGING STRATEGY SUMMARY")
        print(f"{'='*60}")
        
        total_cost = 0
        total_violation = 0
        
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
                
                total_cost += cost
                total_violation += viol
        
        print(f"\n{'='*60}")
        print(f"TOTALS:")
        print(f"  Total Cost: ${total_cost:.4f}")
        print(f"  Total Violation: {total_violation:.6f}")
        print(f"{'='*60}")


# Run the algorithm
if __name__ == "__main__":
    print("Initializing Thompson Sampling + Backward Induction...")
    
    # Create instance with T=2 initially
    ts_bi = ThompsonSamplingBackwardInduction(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T_steps=2,  # Start with T=2, can scale to T=5
        dt=1.0,
        episodes_per_node=5000,  # More episodes for pure exploration
        prior_mean=0.0,  # No bias
        prior_std=15.0  # Moderate exploration
    )
    
    # Run backward induction with Thompson Sampling
    print("\nStarting Backward Induction with Thompson Sampling...\n")
    optimal_hedges = ts_bi.backward_induction_with_ts()
    
    # Summarize results (no plots)
    ts_bi.summarize_results()
    
    print("\n✓ Algorithm complete!")
    print("\nTo scale to T=5, simply change T_steps=5 in the initialization.")