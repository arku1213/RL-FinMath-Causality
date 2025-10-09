import numpy as np
from typing import Dict, List, Tuple

class PPOBackwardInduction:
    """
    Proximal Policy Optimization (PPO) with Backward Induction
    for hedging options using binary options in a binomial tree model.
    
    Pure RL: No bias, no analytical solutions.
    Continuous actions via policy gradient.
    """
    
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T_steps=2, dt=1.0,
                 episodes_per_node=None, learning_rate=0.01, clip_epsilon=0.2,
                 ppo_epochs=10, batch_size=64):
        """
        Parameters:
        -----------
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T_steps: Number of time steps
        dt: Time increment
        episodes_per_node: Training episodes per node (auto-scales if None)
        learning_rate: Learning rate for policy/value networks
        clip_epsilon: PPO clipping parameter (typically 0.2)
        ppo_epochs: Number of PPO update epochs per batch
        batch_size: Mini-batch size for PPO updates
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_steps = T_steps
        self.dt = dt
        self.alpha = learning_rate
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Binomial tree parameters
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Calculate max payoff for auto-scaling
        max_stock_price = S0 * (self.u ** T_steps)
        max_payoff = max(max_stock_price - K, 0)
        
        # AUTO-SCALE episodes (doubled for better learning)
        if episodes_per_node is None:
            self.episodes_per_node = int(10000 * (1 + 0.5 * T_steps))  # 2× more episodes
        else:
            self.episodes_per_node = episodes_per_node
        
        print(f"\n{'='*60}")
        print(f"PPO AUTO-SCALING FOR T={T_steps}")
        print(f"{'='*60}")
        print(f"Max Stock Price: ${max_stock_price:.2f}")
        print(f"Max Payoff: ${max_payoff:.2f}")
        print(f"Episodes per Node: {self.episodes_per_node}")
        print(f"Learning Rate: {self.alpha}")
        print(f"PPO Clip ε: {self.clip_epsilon}")
        print(f"{'='*60}\n")
        
        # Build tree
        self.tree = self._build_tree()
        
        # Policy and Value networks (simple linear for now)
        self.networks = {}
        self._initialize_networks()
        
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
        Initialize Policy and Value networks for each node.
        
        Policy Network: state → (mean, log_std) for Gaussian policy
        Value Network: state → V(state)
        
        IMPROVED: Better initialization near expected payoff range
        """
        for node_id in self.tree.keys():
            num_binaries = self._get_num_binaries(node_id)
            state_dim = 3  # [normalized_price, time, moneyness]
            
            # Better initialization: small random weights instead of zeros
            self.networks[node_id] = {
                'num_binaries': num_binaries,
                # Policy network (outputs mean and log_std)
                'policy_W1': np.random.randn(state_dim, 16) * 0.1,  # Small random init
                'policy_W2_mean': np.random.randn(16, num_binaries) * 10,  # Init near payoff scale
                'policy_W2_logstd': np.random.randn(16, num_binaries) * 0.1 - 1,  # Init std ~exp(-1)≈0.37
                # Value network
                'value_W1': np.random.randn(state_dim, 16) * 0.1,
                'value_W2': np.random.randn(16, 1) * 0.1
            }
    
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
            (S - self.S0) / self.S0,
            t / self.T_steps,
            S / self.K - 1.0
        ])
        
        return features
    
    def _policy_forward(self, node_id, state):
        """
        Forward pass through policy network.
        Returns mean and std for Gaussian policy.
        """
        net = self.networks[node_id]
        
        # Hidden layer with ReLU
        h = np.maximum(0, state @ net['policy_W1'])
        
        # Output layer
        mean = h @ net['policy_W2_mean']
        log_std = h @ net['policy_W2_logstd']
        std = np.exp(np.clip(log_std, -2, 2))  # Clip for stability
        
        return mean, std
    
    def _value_forward(self, node_id, state):
        """Forward pass through value network."""
        net = self.networks[node_id]
        
        h = np.maximum(0, state @ net['value_W1'])
        value = (h @ net['value_W2'])[0]
        
        return value
    
    def _sample_action(self, node_id, state):
        """Sample action from policy (Gaussian)."""
        mean, std = self._policy_forward(node_id, state)
        action = mean + std * np.random.randn(len(mean))
        return action, mean, std
    
    def _log_prob(self, action, mean, std):
        """Compute log probability of action under Gaussian policy."""
        var = std ** 2
        log_prob = -0.5 * np.sum(
            ((action - mean) ** 2) / var + np.log(2 * np.pi * var)
        )
        return log_prob
    
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
        """
        Compute reward with improved shaping.
        Uses sqrt for smoother gradient signals.
        """
        cost, violation = self._evaluate_hedge(node_id, action)
        
        # IMPROVED: Smoother reward shaping with sqrt
        if violation < 0.01:
            reward = 100000 - cost
        else:
            # Use sqrt instead of linear for smoother gradients
            reward = -10000 * np.sqrt(violation)
        
        return reward, cost, violation
    
    def _ppo_update(self, node_id, trajectories):
        """
        PPO update using collected trajectories.
        
        trajectories: list of (state, action, old_log_prob, reward, advantage)
        """
        if len(trajectories) < self.batch_size:
            return
        
        net = self.networks[node_id]
        
        # Convert to arrays
        states = np.array([t[0] for t in trajectories])
        actions = np.array([t[1] for t in trajectories])
        old_log_probs = np.array([t[2] for t in trajectories])
        advantages = np.array([t[4] for t in trajectories])
        returns = np.array([t[3] for t in trajectories])  # reward-to-go
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = np.random.permutation(len(trajectories))
            
            for start in range(0, len(trajectories), self.batch_size):
                end = min(start + self.batch_size, len(trajectories))
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Update policy
                for i in range(len(batch_states)):
                    state = batch_states[i]
                    action = batch_actions[i]
                    old_log_prob = batch_old_log_probs[i]
                    advantage = batch_advantages[i]
                    
                    # Forward pass
                    mean, std = self._policy_forward(node_id, state)
                    new_log_prob = self._log_prob(action, mean, std)
                    
                    # Ratio
                    ratio = np.exp(new_log_prob - old_log_prob)
                    
                    # Clipped objective
                    surr1 = ratio * advantage
                    surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                    policy_loss = -min(surr1, surr2)
                    
                    # Simplified gradient update (policy)
                    # In practice, you'd compute full gradients via backprop
                    h = np.maximum(0, state @ net['policy_W1'])
                    
                    # Gradient w.r.t mean (simplified)
                    grad_mean = (action - mean) / (std ** 2) * policy_loss
                    net['policy_W2_mean'] -= self.alpha * np.outer(h, grad_mean)
                
                # Update value function
                for i in range(len(batch_states)):
                    state = batch_states[i]
                    target_value = batch_returns[i]
                    
                    # Forward pass
                    pred_value = self._value_forward(node_id, state)
                    
                    # MSE loss
                    value_loss = (pred_value - target_value) ** 2
                    
                    # Gradient update (simplified)
                    h = np.maximum(0, state @ net['value_W1'])
                    grad_v = 2 * (pred_value - target_value)
                    net['value_W2'] -= self.alpha * grad_v * h.reshape(-1, 1)
    
    def _train_node_with_ppo(self, node_id):
        """Train a node using PPO"""
        print(f"\n{'='*60}")
        print(f"PPO Training at Node {node_id}")
        print(f"Time: {self.tree[node_id]['time']}, Price: ${self.tree[node_id]['price']:.2f}")
        print(f"{'='*60}")
        
        num_binaries = self._get_num_binaries(node_id)
        best_action = np.zeros(num_binaries)
        best_reward = -np.inf
        episode_rewards = []
        
        state = self._get_state_features(node_id)
        trajectories = []
        
        for episode in range(self.episodes_per_node):
            # Sample action from policy
            action, mean, std = self._sample_action(node_id, state)
            log_prob = self._log_prob(action, mean, std)
            
            # Get reward
            reward, cost, violation = self._get_reward(node_id, action)
            
            # Compute value and advantage
            value = self._value_forward(node_id, state)
            advantage = reward - value
            
            # Store trajectory
            trajectories.append((state, action, log_prob, reward, advantage))
            
            # Track best
            if reward > best_reward:
                best_reward = reward
                best_action = action.copy()
            
            episode_rewards.append(reward)
            
            # PPO update every batch_size episodes
            if (episode + 1) % self.batch_size == 0:
                self._ppo_update(node_id, trajectories)
                trajectories = []
            
            # Progress
            if (episode + 1) % 2000 == 0:
                recent = episode_rewards[-2000:]
                cost_best, viol_best = self._evaluate_hedge(node_id, best_action)
                print(f"Episode {episode+1}/{self.episodes_per_node}")
                print(f"  Avg Reward (last 2000): {np.mean(recent):.2f}")
                print(f"  Best Reward: {best_reward:.2f}")
                print(f"  Best Cost: ${cost_best:.4f}, Violation: {viol_best:.6f}")
        
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
    
    def backward_induction_with_ppo(self):
        """Main algorithm: Backward Induction with PPO"""
        print(f"\n{'#'*60}")
        print(f"BACKWARD INDUCTION WITH PPO")
        print(f"T_steps = {self.T_steps}, Episodes per node = {self.episodes_per_node}")
        print(f"{'#'*60}")
        
        for t in range(self.T_steps, -1, -1):
            print(f"\n{'*'*60}")
            print(f"PROCESSING TIME STEP t = {t}")
            print(f"{'*'*60}")
            
            nodes_at_t = [node_id for node_id, info in self.tree.items() 
                         if info['time'] == t]
            
            for node_id in nodes_at_t:
                best_hedge = self._train_node_with_ppo(node_id)
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
    print("Initializing PPO + Backward Induction...")
    
    ppo_bi = PPOBackwardInduction(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T_steps=2,  # Just change this! Everything auto-scales
        dt=1.0
        # All parameters auto-scale with T
    )
    
    print("\nStarting Backward Induction with PPO...\n")
    optimal_hedges = ppo_bi.backward_induction_with_ppo()
    
    ppo_bi.summarize_results()
    
    print("\n✓ Algorithm complete!")
    print("\nPPO uses policy gradient for continuous actions!")