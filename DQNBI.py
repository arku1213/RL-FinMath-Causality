import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple

class BranchingDQNBackwardInduction:
    """
    Branching Deep Q-Network with Backward Induction
    for hedging options using binary options in a binomial tree model.
    
    Pure RL: No bias, no analytical solutions.
    Handles multi-dimensional continuous-like action spaces via discretization.
    """
    
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T_steps=2, dt=1.0,
                 episodes_per_node=10000, learning_rate=0.001, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, batch_size=64,
                 action_levels=11, action_range=50):
        """
        Parameters:
        -----------
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T_steps: Number of time steps
        dt: Time increment
        episodes_per_node: Training episodes per node
        learning_rate: Learning rate for Q-network updates
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay rate
        batch_size: Mini-batch size for experience replay
        action_levels: Number of discrete levels per binary (e.g., 11)
        action_range: Range of binary positions (e.g., [-50, 50])
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_steps = T_steps
        self.dt = dt
        self.episodes_per_node = episodes_per_node
        self.alpha = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Action space discretization
        self.action_levels = action_levels
        self.action_range = action_range
        self.action_values = np.linspace(-action_range, action_range, action_levels)
        
        # Binomial tree parameters
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Build tree
        self.tree = self._build_tree()
        
        # Q-networks: separate for each node
        # Each node has branching Q-network: one branch per binary
        self.Q_networks = {}
        self._initialize_networks()
        
        # Experience replay buffers (one per node)
        self.replay_buffers = {}
        for node_id in self.tree.keys():
            self.replay_buffers[node_id] = deque(maxlen=10000)
        
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
    
    def _initialize_networks(self):
        """
        Initialize Branching Q-networks for each node.
        Each network has separate branches for each binary position.
        """
        for node_id in self.tree.keys():
            num_binaries = self._get_num_binaries(node_id)
            state_dim = 3  # [normalized_price, time, moneyness]
            
            # Branching Q-network structure:
            # Input: state features (3 dims)
            # Output: Q-values for each action level for each binary
            # Shape: num_binaries × action_levels
            
            # We'll use simple linear networks (one per branch)
            # Each branch: state → hidden → Q-values for that binary's actions
            
            self.Q_networks[node_id] = {
                'num_binaries': num_binaries,
                'branches': []
            }
            
            # Initialize weights for each branch (simple 2-layer network)
            for b in range(num_binaries):
                branch = {
                    'W1': np.zeros((state_dim, 16)),  # Input to hidden (no bias)
                    'W2': np.zeros((16, self.action_levels))  # Hidden to output
                }
                self.Q_networks[node_id]['branches'].append(branch)
    
    def _get_num_binaries(self, node_id):
        """Get number of binary options at this node"""
        t = self.tree[node_id]['time']
        steps_remaining = self.T_steps - t
        return steps_remaining + 1 if steps_remaining > 0 else 1
    
    def _payoff_function(self, S_T):
        """Option payoff (European Call)"""
        return max(S_T - self.K, 0)
    
    def _get_state_features(self, node_id):
        """Extract normalized state features"""
        node = self.tree[node_id]
        S = node['price']
        t = node['time']
        
        features = np.array([
            (S - self.S0) / self.S0,  # Normalized price
            t / self.T_steps,          # Normalized time
            S / self.K - 1.0           # Moneyness
        ])
        
        return features
    
    def _forward_pass(self, node_id, state):
        """
        Forward pass through branching Q-network.
        Returns Q-values for each action level for each binary.
        
        Returns:
        --------
        Q_values: list of arrays, one per binary
                 Each array has shape (action_levels,)
        """
        network = self.Q_networks[node_id]
        Q_values_all_branches = []
        
        for branch in network['branches']:
            # Simple 2-layer feedforward
            h = np.maximum(0, state @ branch['W1'])  # ReLU activation
            Q_vals = h @ branch['W2']
            Q_values_all_branches.append(Q_vals)
        
        return Q_values_all_branches
    
    def _select_action(self, node_id, state, epsilon=None):
        """
        Epsilon-greedy action selection with branching.
        Each binary position chosen independently.
        
        Returns:
        --------
        action_indices: index of chosen action level for each binary
        action_values: actual values of binary positions
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        num_binaries = self.Q_networks[node_id]['num_binaries']
        action_indices = []
        
        Q_values_all = self._forward_pass(node_id, state)
        
        for b in range(num_binaries):
            if np.random.rand() < epsilon:
                # Explore: random action
                action_idx = np.random.randint(0, self.action_levels)
            else:
                # Exploit: best action for this branch
                action_idx = np.argmax(Q_values_all[b])
            
            action_indices.append(action_idx)
        
        # Convert indices to actual values
        action_values = np.array([self.action_values[idx] for idx in action_indices])
        
        return action_indices, action_values
    
    def _evaluate_hedge(self, node_id, binary_positions):
        """Evaluate hedge (same as previous implementations)"""
        node = self.tree[node_id]
        t = node['time']
        
        # Terminal node
        if node['terminal']:
            S = node['price']
            target_payoff = self._payoff_function(S)
            replicated_payoff = binary_positions[0] if len(binary_positions) > 0 else 0
            violation = (replicated_payoff - target_payoff) ** 2
            
            discount = np.exp(-self.r * (self.T_steps - t) * self.dt)
            cost = binary_positions[0] * discount if len(binary_positions) > 0 else 0
            
            return cost, violation
        
        # Non-terminal
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
    
    def _get_reward(self, node_id, action_values):
        """Compute reward"""
        cost, violation = self._evaluate_hedge(node_id, action_values)
        
        if violation < 0.01:
            reward = 100000 - cost
        else:
            reward = -100000 * violation
        
        return reward, cost, violation
    
    def _update_networks(self, node_id):
        """
        Update Q-networks using experience replay and mini-batch gradient descent.
        """
        if len(self.replay_buffers[node_id]) < self.batch_size:
            return
        
        # Sample mini-batch
        batch = random.sample(self.replay_buffers[node_id], self.batch_size)
        
        network = self.Q_networks[node_id]
        num_binaries = network['num_binaries']
        
        # Update each branch separately
        for b in range(num_binaries):
            branch = network['branches'][b]
            
            for experience in batch:
                state, action_indices, reward = experience
                
                # Current Q-value for this branch's action
                Q_values = self._forward_pass(node_id, state)[b]
                action_idx = action_indices[b]
                Q_current = Q_values[action_idx]
                
                # Target (no next state in our single-stage setup)
                target = reward
                
                # TD error
                td_error = target - Q_current
                
                # Gradient descent update (simplified backprop)
                # ∂L/∂W2 = ∂L/∂Q × ∂Q/∂W2
                h = np.maximum(0, state @ branch['W1'])
                
                # Output layer gradient
                delta_out = np.zeros(self.action_levels)
                delta_out[action_idx] = -td_error
                
                # Update W2
                grad_W2 = np.outer(h, delta_out)
                branch['W2'] -= self.alpha * grad_W2
                
                # Hidden layer gradient (simplified)
                delta_hidden = (delta_out[action_idx] * branch['W2'][:, action_idx]) * (h > 0)
                
                # Update W1
                grad_W1 = np.outer(state, delta_hidden)
                branch['W1'] -= self.alpha * grad_W1
    
    def _train_node_with_dqn(self, node_id):
        """Train a node using Branching DQN"""
        print(f"\n{'='*60}")
        print(f"Branching DQN Training at Node {node_id}")
        print(f"Time: {self.tree[node_id]['time']}, Price: ${self.tree[node_id]['price']:.2f}")
        print(f"{'='*60}")
        
        num_binaries = self._get_num_binaries(node_id)
        best_action = np.zeros(num_binaries)
        best_reward = -np.inf
        episode_rewards = []
        
        state = self._get_state_features(node_id)
        
        for episode in range(self.episodes_per_node):
            # Select action
            action_indices, action_values = self._select_action(node_id, state)
            
            # Get reward
            reward, cost, violation = self._get_reward(node_id, action_values)
            
            # Store experience
            self.replay_buffers[node_id].append((state, action_indices, reward))
            
            # Update networks
            if episode > self.batch_size:
                self._update_networks(node_id)
            
            # Track best
            if reward > best_reward:
                best_reward = reward
                best_action = action_values.copy()
            
            episode_rewards.append(reward)
            
            # Decay epsilon
            if episode % 100 == 0:
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Progress
            if (episode + 1) % 2000 == 0:
                recent = episode_rewards[-2000:]
                cost_best, viol_best = self._evaluate_hedge(node_id, best_action)
                print(f"Episode {episode+1}/{self.episodes_per_node}")
                print(f"  Avg Reward (last 2000): {np.mean(recent):.2f}")
                print(f"  Best Reward: {best_reward:.2f}")
                print(f"  Best Cost: ${cost_best:.4f}, Violation: {viol_best:.6f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
        
        # Final
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
    
    def backward_induction_with_dqn(self):
        """Main algorithm: Backward Induction with Branching DQN"""
        print(f"\n{'#'*60}")
        print(f"BACKWARD INDUCTION WITH BRANCHING DQN")
        print(f"T_steps = {self.T_steps}, Episodes per node = {self.episodes_per_node}")
        print(f"Action discretization: {self.action_levels} levels in [{-self.action_range}, {self.action_range}]")
        print(f"{'#'*60}")
        
        for t in range(self.T_steps, -1, -1):
            print(f"\n{'*'*60}")
            print(f"PROCESSING TIME STEP t = {t}")
            print(f"{'*'*60}")
            
            # Reset epsilon
            self.epsilon = 1.0
            
            nodes_at_t = [node_id for node_id, info in self.tree.items() 
                         if info['time'] == t]
            
            for node_id in nodes_at_t:
                best_hedge = self._train_node_with_dqn(node_id)
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
    print("Initializing Branching DQN + Backward Induction...")
    
    dqn_bi = BranchingDQNBackwardInduction(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T_steps=2,
        dt=1.0,
        episodes_per_node=10000,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        action_levels=11,  # Discretization: 11 levels per binary
        action_range=50    # Range: [-50, 50]
    )
    
    print("\nStarting Backward Induction with Branching DQN...\n")
    optimal_hedges = dqn_bi.backward_induction_with_dqn()
    
    dqn_bi.summarize_results()
    
    print("\n✓ Algorithm complete!")
    print("\nTo scale to T=5, change T_steps=5.")
    print("DQN handles scaling well due to branching structure!")