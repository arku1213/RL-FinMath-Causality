import numpy as np
from typing import Dict, List, Tuple

class LinearSARSABackwardInduction:
    """
    Linear Function Approximation using SARSA with Backward Induction
    for hedging options using binary options in a binomial tree model.
    
    Pure RL: No bias, no analytical solutions.
    """
    
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T_steps=2, dt=1.0,
                 episodes_per_node=10000, learning_rate=0.001, epsilon=0.3):
        """
        Parameters:
        -----------
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T_steps: Number of time steps
        dt: Time increment
        episodes_per_node: Number of SARSA episodes per node
        learning_rate: Learning rate for weight updates (lower for stability)
        epsilon: Exploration rate (epsilon-greedy, higher for more exploration)
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_steps = T_steps
        self.dt = dt
        self.episodes_per_node = episodes_per_node
        self.alpha = learning_rate
        self.epsilon = epsilon
        
        # Binomial tree parameters
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Build tree structure
        self.tree = self._build_tree()
        
        # Linear function approximation: separate weights for each node
        # Each node has its own weight vector for Q(s,a)
        self.weights = {}  # {node_id: weight_vector}
        self._initialize_weights()
        
        # Store learned optimal hedges
        self.optimal_hedges = {}
        
        # Track learning progress
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
    
    def _initialize_weights(self):
        """Initialize weight vectors for linear Q-function at each node"""
        for node_id in self.tree.keys():
            num_binaries = self._get_num_binaries(node_id)
            # Feature dimension: expanded features for better expressiveness
            # State: [S, t, normalized_price, S^2, moneyness^2]
            # Action: [b1, b2, ..., bn]
            # Cross terms: state * action
            # Quadratic action terms: b_i^2
            feature_dim = 5 + num_binaries + (5 * num_binaries) + num_binaries
            
            # Initialize weights to zero (no bias)
            self.weights[node_id] = np.zeros(feature_dim)
    
    def _get_num_binaries(self, node_id):
        """Get number of binary options available at this node"""
        t = self.tree[node_id]['time']
        steps_remaining = self.T_steps - t
        return steps_remaining + 1 if steps_remaining > 0 else 1
    
    def _payoff_function(self, S_T):
        """Option payoff at maturity (European Call)"""
        return max(S_T - self.K, 0)
    
    def _get_state_features(self, node_id):
        """Extract state features from a node (expanded for better expressiveness)"""
        node = self.tree[node_id]
        S = node['price']
        t = node['time']
        
        # More expressive normalized features
        norm_price = (S - self.S0) / self.S0
        norm_time = t / self.T_steps
        moneyness = S / self.K - 1.0
        
        features = np.array([
            norm_price,
            norm_time,
            moneyness,
            norm_price ** 2,  # Quadratic term
            moneyness ** 2    # Quadratic moneyness
        ])
        
        return features
    
    def _compute_features(self, node_id, action):
        """
        Compute richer feature vector φ(s,a) for linear Q-function.
        
        Features include:
        - State features (5 dimensions)
        - Action features (binary positions)
        - Cross terms (state × action) - all combinations
        - Quadratic action terms (b_i^2 to capture cost structure)
        """
        state_features = self._get_state_features(node_id)
        action_array = np.array(action)
        
        # Cross terms: each state feature × each action
        cross_terms = []
        for s_feat in state_features:
            for a_feat in action_array:
                cross_terms.append(s_feat * a_feat)
        cross_terms = np.array(cross_terms)
        
        # Quadratic action terms (captures cost structure better)
        quad_action = action_array ** 2
        
        # Concatenate all features
        features = np.concatenate([
            state_features,     # 5 dims
            action_array,       # num_binaries dims
            cross_terms,        # 5 * num_binaries dims
            quad_action         # num_binaries dims
        ])
        
        return features
    
    def _Q_value(self, node_id, action):
        """
        Compute Q(s,a) = φ(s,a)ᵀw
        Linear function approximation
        """
        features = self._compute_features(node_id, action)
        weights = self.weights[node_id]
        
        return np.dot(features, weights)
    
    def _sample_action(self, node_id, epsilon=None):
        """
        Sample action using epsilon-greedy policy with adaptive sampling range.
        Action = binary positions (continuous)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        num_binaries = self._get_num_binaries(node_id)
        
        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            # Explore: sample random action from wider range
            action = np.random.randn(num_binaries) * 30.0  # Increased exploration range
        else:
            # Exploit: use policy gradient to find best action
            action = self._policy_gradient_action(node_id)
        
        return action
    
    def _policy_gradient_action(self, node_id, num_samples=100):
        """
        Find action that maximizes Q(s,a) using policy gradient sampling.
        Sample multiple actions and pick the one with highest Q-value.
        Increased samples for better exploitation.
        """
        num_binaries = self._get_num_binaries(node_id)
        
        best_action = np.zeros(num_binaries)
        best_Q = -np.inf
        
        # Sample multiple candidate actions (more samples for better search)
        for _ in range(num_samples):
            candidate = np.random.randn(num_binaries) * 30.0
            Q = self._Q_value(node_id, candidate)
            
            if Q > best_Q:
                best_Q = Q
                best_action = candidate
        
        return best_action
    
    def _evaluate_hedge(self, node_id, binary_positions):
        """
        Evaluate hedge quality (same as Thompson Sampling version).
        Returns cost and violation.
        """
        node = self.tree[node_id]
        t = node['time']
        
        # Terminal node: match option payoff
        if node['terminal']:
            S = node['price']
            target_payoff = self._payoff_function(S)
            replicated_payoff = binary_positions[0] if len(binary_positions) > 0 else 0
            violation = (replicated_payoff - target_payoff) ** 2
            
            discount = np.exp(-self.r * (self.T_steps - t) * self.dt)
            cost = binary_positions[0] * discount if len(binary_positions) > 0 else 0
            
            return cost, violation
        
        # Non-terminal: match continuation values at child nodes
        child_up_id = node['child_up']
        child_down_id = node['child_down']
        
        # Get continuation values
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
        
        # Aggregate binary positions to up/down values
        num_binaries = len(binary_positions)
        if num_binaries == 2:
            replicated_values = binary_positions[:2]
        else:
            mid = (num_binaries + 1) // 2
            value_up_rep = np.sum(binary_positions[:mid])
            value_down_rep = np.sum(binary_positions[mid:])
            replicated_values = np.array([value_up_rep, value_down_rep])
        
        violation = np.sum((replicated_values - target_values) ** 2)
        
        # Cost
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
        """
        Compute reward for taking action at node_id.
        Reward = -(cost + heavy_penalty * violation)
        """
        cost, violation = self._evaluate_hedge(node_id, action)
        
        # Two-stage reward (same as Thompson Sampling)
        if violation < 0.01:
            reward = 100000 - cost
        else:
            reward = -100000 * violation
        
        return reward, cost, violation
    
    def _sarsa_update(self, node_id, action, reward, next_action):
        """
        SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        For our problem: no next state (single-stage per node)
        So: Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
        """
        # Current Q-value
        Q_current = self._Q_value(node_id, action)
        
        # TD error (no next state, so target = reward)
        td_error = reward - Q_current
        
        # Compute features
        features = self._compute_features(node_id, action)
        
        # Update weights: w ← w + α * td_error * φ(s,a)
        self.weights[node_id] += self.alpha * td_error * features
    
    def _train_node_with_sarsa(self, node_id):
        """
        Train a single node using SARSA.
        """
        print(f"\n{'='*60}")
        print(f"SARSA Training at Node {node_id}")
        print(f"Time: {self.tree[node_id]['time']}, Price: ${self.tree[node_id]['price']:.2f}")
        print(f"{'='*60}")
        
        num_binaries = self._get_num_binaries(node_id)
        best_action = np.zeros(num_binaries)  # Initialize properly
        best_reward = -np.inf
        episode_rewards = []
        
        for episode in range(self.episodes_per_node):
            # Sample action (epsilon-greedy)
            action = self._sample_action(node_id)
            
            # Get reward
            reward, cost, violation = self._get_reward(node_id, action)
            
            # SARSA update (no next action needed for single-stage)
            self._sarsa_update(node_id, action, reward, None)
            
            # Track best
            if reward > best_reward:
                best_reward = reward
                best_action = action.copy()
            
            episode_rewards.append(reward)
            
            # Decay exploration more gradually
            if (episode + 1) % 2000 == 0:
                self.epsilon = max(0.05, self.epsilon * 0.9)  # Slower decay, higher floor
            
            # Progress
            if (episode + 1) % 1000 == 0:
                recent_rewards = episode_rewards[-1000:]
                cost_best, viol_best = self._evaluate_hedge(node_id, best_action)
                print(f"Episode {episode+1}/{self.episodes_per_node}")
                print(f"  Avg Reward (last 1000): {np.mean(recent_rewards):.2f}")
                print(f"  Best Reward: {best_reward:.2f}")
                print(f"  Best Cost: ${cost_best:.4f}, Violation: {viol_best:.6f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
        
        # Final evaluation
        cost_final, viol_final = self._evaluate_hedge(node_id, best_action)
        print(f"\n✓ FINAL RESULTS:")
        print(f"  Best Hedge: {best_action}")
        print(f"  Cost: ${cost_final:.4f}")
        print(f"  Violation: {viol_final:.6f}")
        
        self.learning_history.append({
            'node_id': node_id,
            'time': self.tree[node_id]['time'],
            'best_hedge': best_action,
            'cost': cost_final,
            'violation': viol_final,
            'reward_history': episode_rewards
        })
        
        return best_action
    
    def backward_induction_with_sarsa(self):
        """
        Main algorithm: Backward Induction with SARSA at each node.
        """
        print(f"\n{'#'*60}")
        print(f"BACKWARD INDUCTION WITH LINEAR SARSA")
        print(f"T_steps = {self.T_steps}, Episodes per node = {self.episodes_per_node}")
        print(f"{'#'*60}")
        
        # Process nodes backward (from T to 0)
        for t in range(self.T_steps, -1, -1):
            print(f"\n{'*'*60}")
            print(f"PROCESSING TIME STEP t = {t}")
            print(f"{'*'*60}")
            
            # Reset epsilon for each time step (more exploration at new nodes)
            self.epsilon = 0.3
            
            nodes_at_t = [node_id for node_id, info in self.tree.items() 
                         if info['time'] == t]
            
            for node_id in nodes_at_t:
                best_hedge = self._train_node_with_sarsa(node_id)
                self.optimal_hedges[node_id] = best_hedge
        
        print(f"\n{'#'*60}")
        print(f"BACKWARD INDUCTION COMPLETE!")
        print(f"{'#'*60}")
        
        return self.optimal_hedges
    
    def summarize_results(self):
        """Print summary of learned hedging strategy"""
        print(f"\n{'='*60}")
        print(f"LEARNED HEDGING STRATEGY SUMMARY")
        print(f"{'='*60}")
        
        total_violation = 0
        
        # Only count initial cost (t=0)
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
                    initial_cost = cost  # Only cost at root matters
                total_violation += viol
        
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Initial Hedge Cost (t=0): ${initial_cost:.4f}")
        print(f"  Total Violation: {total_violation:.6f}")
        print(f"{'='*60}")


# Run the algorithm
if __name__ == "__main__":
    print("Initializing Linear SARSA + Backward Induction...")
    
    sarsa_bi = LinearSARSABackwardInduction(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T_steps=2,
        dt=1.0,
        episodes_per_node=10000,  # More episodes
        learning_rate=0.001,  # Lower learning rate for stability
        epsilon=0.3  # Higher exploration
    )
    
    print("\nStarting Backward Induction with Linear SARSA...\n")
    optimal_hedges = sarsa_bi.backward_induction_with_sarsa()
    
    sarsa_bi.summarize_results()
    
    print("\n✓ Algorithm complete!")