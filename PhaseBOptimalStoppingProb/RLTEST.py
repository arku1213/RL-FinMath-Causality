import numpy as np
from collections import defaultdict
import random

class RLOptimalStopping:
    """
    Solve Causal Optimal Stopping using Tabular Q-Learning
    
    This provides an RL-based alternative to backward induction,
    learning the optimal policy through simulated experience.
    """
    
    def __init__(self, X0=10):
        # ========================================================================
        # CONFIGURATION (same as your original)
        # ========================================================================
        self.T = 6
        self.X_min, self.X_max = 1, 20
        self.X0 = X0
        self.safe_min = 3
        self.safe_max = 17
        self.U_values = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        
        # Shock probabilities
        self.U_probs = self._compute_U_probabilities()
        
        # RL-specific parameters
        self.Q = defaultdict(lambda: defaultdict(float))  # Q-table: Q[state][action]
        self.policy = {}  # Derived greedy policy
        self.optimal_intervention_target = {}  # Target for each intervention state
        
        # Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 1.0  # Discount factor (no discounting for finite horizon)
        self.epsilon = 0.1  # Exploration rate
        self.num_episodes = 50000  # Number of training episodes
        
    def _compute_U_probabilities(self):
        """Same as original - extremely negative-biased distribution"""
        probs = np.array([0.15, 0.20, 0.25, 0.20, 0.10, 0.05, 0.03, 0.01, 0.01])
        return probs / probs.sum()
    
    def transition(self, X, U_current, U_next):
        """Same transition dynamics as original"""
        if X < self.safe_min or X > self.safe_max:
            return int(np.clip(X, self.X_min, self.X_max))
        
        X_next = np.floor(X + U_current/3 + U_next/2)
        return int(np.clip(X_next, self.X_min, self.X_max))
    
    def compute_Y(self, XT):
        """Same terminal reward as original"""
        if XT < self.safe_min or XT > self.safe_max:
            return 0  # Death
        else:
            return 1  # Survival
    
    def sample_shock(self):
        """Sample a shock value from the distribution"""
        return np.random.choice(self.U_values, p=self.U_probs)
    
    def get_state_key(self, t, X, U, I):
        """Create hashable state key for Q-table"""
        return (t, X, U, I)
    
    def get_possible_actions(self, state):
        """Get valid actions for a state"""
        t, X, U, I = state
        
        # At terminal time, no actions
        if t >= self.T:
            return []
        
        # Already intervened, can only WAIT
        if I == 1:
            return ['WAIT']
        
        # At death boundary, effectively no action
        if X < self.safe_min or X > self.safe_max:
            return ['WAIT']  # Doesn't matter, already dead
        
        # Haven't intervened yet, can do either
        return ['WAIT', 'INTERVENE']
    
    def choose_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        possible_actions = self.get_possible_actions(state)
        
        if not possible_actions:
            return None
        
        # Epsilon-greedy
        if random.random() < epsilon:
            return random.choice(possible_actions)
        else:
            # Greedy: choose action with highest Q-value
            q_values = {action: self.Q[state][action] for action in possible_actions}
            return max(q_values, key=q_values.get)
    
    def choose_intervention_target(self, state):
        """
        When intervening, choose the best target X' based on Q-values
        
        We'll try all possible targets and pick the one with highest expected Q-value
        """
        t, X, U, I = state
        
        best_target = None
        best_value = -np.inf
        
        # Try all possible intervention targets
        for X_target in range(self.safe_min, self.safe_max + 1):
            # Estimate value of intervening to X_target
            # Sample a few next shocks and average
            total_value = 0
            n_samples = 10
            for _ in range(n_samples):
                U_next = self.sample_shock()
                X_next = self.transition(X_target, U, U_next)
                next_state = self.get_state_key(t+1, X_next, U_next, 1)
                
                # Get max Q-value for next state
                next_actions = self.get_possible_actions(next_state)
                if next_actions:
                    next_q = max([self.Q[next_state][a] for a in next_actions])
                else:
                    next_q = self.compute_Y(X_next)
                
                total_value += next_q
            
            avg_value = total_value / n_samples
            
            if avg_value > best_value:
                best_value = avg_value
                best_target = X_target
        
        return best_target if best_target is not None else 10  # Default to center
    
    def train_q_learning(self, verbose=True):
        """
        Train Q-table using Q-Learning
        
        Each episode:
        1. Start from initial state
        2. Take actions according to epsilon-greedy policy
        3. Update Q-values using Bellman update
        4. Get terminal reward
        """
        
        if verbose:
            print(f"Training Q-Learning for {self.num_episodes} episodes...")
            print(f"Learning rate: {self.alpha}, Epsilon: {self.epsilon}")
        
        for episode in range(self.num_episodes):
            # Decay epsilon over time (exploration -> exploitation)
            epsilon = self.epsilon * (1 - episode / self.num_episodes)
            
            # Initialize episode
            t = 1
            U_current = self.sample_shock()
            X_current = int(np.floor(self.X0 + U_current/2))
            X_current = np.clip(X_current, self.X_min, self.X_max)
            I_current = 0  # Haven't intervened yet
            
            episode_trajectory = []  # Store (state, action, reward) tuples
            
            # Run episode
            while t < self.T:
                state = self.get_state_key(t, X_current, U_current, I_current)
                
                # Choose action
                action = self.choose_action(state, epsilon)
                
                if action is None:
                    break
                
                # Sample next shock
                U_next = self.sample_shock()
                
                # Execute action
                if action == 'INTERVENE':
                    # Choose best target
                    X_target = self.choose_intervention_target(state)
                    X_next = self.transition(X_target, U_current, U_next)
                    I_next = 1
                else:  # WAIT
                    X_next = self.transition(X_current, U_current, U_next)
                    I_next = I_current
                
                # Move to next state
                next_state = self.get_state_key(t+1, X_next, U_next, I_next)
                
                # Store trajectory
                episode_trajectory.append((state, action, next_state))
                
                # Update current state
                t += 1
                X_current = X_next
                U_current = U_next
                I_current = I_next
            
            # Get terminal reward
            terminal_reward = self.compute_Y(X_current)
            
            # Backward update through trajectory (Monte Carlo-style for terminal reward)
            # Start from the end and work backwards
            G = terminal_reward  # Return
            
            for state, action, next_state in reversed(episode_trajectory):
                # Q-Learning update
                next_actions = self.get_possible_actions(next_state)
                if next_actions:
                    max_next_q = max([self.Q[next_state][a] for a in next_actions])
                else:
                    max_next_q = terminal_reward
                
                # Temporal difference error
                td_target = max_next_q
                td_error = td_target - self.Q[state][action]
                
                # Update Q-value
                self.Q[state][action] += self.alpha * td_error
            
            # Print progress
            if verbose and (episode + 1) % 10000 == 0:
                print(f"Episode {episode + 1}/{self.num_episodes} completed")
        
        if verbose:
            print("Training completed!")
            print(f"Q-table size: {len(self.Q)} states")
        
        # Extract greedy policy from Q-table
        self.extract_policy_from_q_table()
    
    def extract_policy_from_q_table(self):
        """Extract greedy policy from learned Q-table"""
        print("\nExtracting greedy policy from Q-table...")
        
        for t in range(1, self.T):
            for X in range(self.X_min, self.X_max + 1):
                for U in self.U_values:
                    for I in [0, 1]:
                        state = self.get_state_key(t, X, U, I)
                        actions = self.get_possible_actions(state)
                        
                        if not actions:
                            self.policy[state] = 'no_action'
                            continue
                        
                        # Choose action with highest Q-value
                        q_values = {action: self.Q[state][action] for action in actions}
                        best_action = max(q_values, key=q_values.get)
                        
                        self.policy[state] = best_action
                        
                        # If intervening, determine target
                        if best_action == 'INTERVENE' and I == 0:
                            target = self.choose_intervention_target(state)
                            self.optimal_intervention_target[(t, X, U)] = target
        
        print("Policy extraction completed!")
    
    def print_results(self, output_file='RESULTS_RL.txt'):
        """Print results in same format as original code"""
        
        # Make sure we have trained
        if not self.policy:
            print("Training Q-Learning first...")
            self.train_q_learning()
        
        with open(output_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write("Q-LEARNING OPTIMAL STOPPING RESULTS\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Training: {self.num_episodes} episodes\n")
            f.write(f"Learning rate: {self.alpha}, Epsilon: {self.epsilon}\n")
            f.write(f"Q-table size: {len(self.Q)} states\n\n")
            
            # ================================================================
            # POLICY BY TIME PERIOD
            # ================================================================
            f.write(f"{'='*80}\n")
            f.write("LEARNED POLICY\n")
            f.write(f"{'='*80}\n\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*40}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*40}\n")
                
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX_{t}={X:2d}:\n")
                    
                    for U in self.U_values:
                        # State (X, U, I=0)
                        state = self.get_state_key(t, X, U, 0)
                        
                        if t == self.T:
                            reward = self.compute_Y(X)
                            f.write(f"  U_{t}={U:2d}, I=0: Terminal, Y={reward}\n")
                        else:
                            policy = self.policy.get(state, '?')
                            q_val = self.Q[state].get(policy, 0) if policy != '?' else 0
                            
                            if policy == 'INTERVENE':
                                X_target = self.optimal_intervention_target.get((t, X, U), '?')
                                f.write(f"  U_{t}={U:2d}, I=0: 🔴 INTERVENE → X becomes {X_target}, Q={q_val:.4f}\n")
                            elif policy == 'WAIT':
                                f.write(f"  U_{t}={U:2d}, I=0: ⚪ WAIT, Q={q_val:.4f}\n")
                            elif policy == 'no_action':
                                f.write(f"  U_{t}={U:2d}, I=0: ☠️  DEATH (boundary), Q={q_val:.4f}\n")
                            else:
                                f.write(f"  U_{t}={U:2d}, I=0: ?, Q={q_val:.4f}\n")
                        
                        # State (X, U, I=1)
                        state_used = self.get_state_key(t, X, U, 1)
                        policy_used = self.policy.get(state_used, '?')
                        q_val_used = self.Q[state_used].get(policy_used, 0) if policy_used != '?' else 0
                        
                        if policy_used == 'no_action' and (X < self.safe_min or X > self.safe_max):
                            f.write(f"  U_{t}={U:2d}, I=1: ☠️  DEATH (boundary), Q={q_val_used:.4f}\n")
                        else:
                            f.write(f"  U_{t}={U:2d}, I=1: {policy_used}, Q={q_val_used:.4f}\n")
            
            # ================================================================
            # THRESHOLD POLICY
            # ================================================================
            f.write(f"\n\n{'='*80}\n")
            f.write("THRESHOLD POLICY\n")
            f.write(f"{'='*80}\n\n")
            
            for t in range(1, self.T):
                f.write(f"\nAt time t={t}:\n")
                for U in self.U_values:
                    intervene_states = []
                    for X in range(self.X_min, self.X_max + 1):
                        state = self.get_state_key(t, X, U, 0)
                        if self.policy.get(state) == 'INTERVENE':
                            intervene_states.append(X)
                    
                    if intervene_states:
                        states_str = '{' + ', '.join(map(str, intervene_states)) + '}'
                    else:
                        states_str = '∅'
                    f.write(f"  U_{t}={U:2d}: Intervene if X_{t} ∈ {states_str}\n")
            
            # ================================================================
            # SUMMARY
            # ================================================================
            f.write(f"\n{'─'*80}\n")
            f.write("SUMMARY\n")
            f.write(f"{'─'*80}\n\n")
            
            intervene_conditions = {}
            
            for t in range(1, self.T):
                for U in self.U_values:
                    for X in range(self.X_min, self.X_max + 1):
                        state = self.get_state_key(t, X, U, 0)
                        if self.policy.get(state) == 'INTERVENE':
                            if X not in intervene_conditions:
                                intervene_conditions[X] = {}
                            target = self.optimal_intervention_target.get((t, X, U), None)
                            if target is not None:
                                if U not in intervene_conditions[X]:
                                    intervene_conditions[X][U] = []
                                intervene_conditions[X][U].append(target)
            
            if intervene_conditions:
                from collections import Counter
                for X in sorted(intervene_conditions.keys()):
                    U_target_map = intervene_conditions[X]
                    
                    all_targets = []
                    for targets_list in U_target_map.values():
                        all_targets.extend(targets_list)
                    
                    if all_targets:
                        target_counts = Counter(all_targets)
                        most_common_target = target_counts.most_common(1)[0][0]
                        
                        U_set = sorted(U_target_map.keys())
                        
                        if len(U_set) == len(self.U_values):
                            f.write(f"  X_n = {X} (for any U), intervene with X_n' = {most_common_target}\n")
                        else:
                            U_min = min(U_set)
                            U_max = max(U_set)
                            
                            if U_set == list(range(U_min, U_max + 1)):
                                if U_min == min(self.U_values):
                                    f.write(f"  X_n = {X} and U ≤ {U_max}, intervene with X_n' = {most_common_target}\n")
                                elif U_max == max(self.U_values):
                                    f.write(f"  X_n = {X} and U ≥ {U_min}, intervene with X_n' = {most_common_target}\n")
                                else:
                                    f.write(f"  X_n = {X} and {U_min} ≤ U ≤ {U_max}, intervene with X_n' = {most_common_target}\n")
                            else:
                                U_str = '{' + ', '.join(map(str, U_set)) + '}'
                                f.write(f"  X_n = {X} and U ∈ {U_str}, intervene with X_n' = {most_common_target}\n")
            else:
                f.write("Never intervene\n")
            
            f.write(f"\n{'='*80}\n")
        
        print(f"Results saved to {output_file}")


# Run Q-Learning
if __name__ == "__main__":
    # Train RL model
    rl_model = RLOptimalStopping(X0=10)
    rl_model.train_q_learning(verbose=True)
    rl_model.print_results('RESULTS_RL.txt')
    
    print("\n" + "="*80)
    print("RL Training Complete!")
    print("="*80)
    print("\nYou can now compare:")
    print("  - RESULTS.txt (Exact Dynamic Programming)")
    print("  - RESULTS_RL.txt (Q-Learning)")