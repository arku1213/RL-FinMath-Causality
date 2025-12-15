import numpy as np
from collections import defaultdict

class RLOptimalStopping:
    """
    Solve Causal Optimal Stopping using Q-Value Iteration
    
    This is essentially backward induction framed as RL,
    guaranteed to converge to the exact optimal policy.
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
        
        # RL structures
        self.Q = {}  # Q-table: Q[(state, action)]
        self.V = {}  # Value function: V[state]
        self.policy = {}  # Optimal policy
        self.optimal_intervention_target = {}
        
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
        """Terminal reward"""
        if XT < self.safe_min or XT > self.safe_max:
            return 0  # Death
        else:
            return 1  # Survival
    
    def get_state_key(self, t, X, U, I):
        """Create hashable state key"""
        return (t, X, U, I)
    
    def compute_q_wait(self, t, X, U, I):
        """
        Compute Q(state, WAIT) using Bellman equation
        
        Q(s, WAIT) = E[V(s')]
        """
        if t >= self.T:
            return self.compute_Y(X)
        
        q_value = 0.0
        for U_next in self.U_values:
            prob = self.U_probs[self.U_values.index(U_next)]
            X_next = self.transition(X, U, U_next)
            next_state = self.get_state_key(t+1, X_next, U_next, I)
            q_value += prob * self.V.get(next_state, 0)
        
        return q_value
    
    def compute_q_intervene(self, t, X, U):
        """
        Compute Q(state, INTERVENE) and find best target
        
        Q(s, INTERVENE) = max_{target} E[V(s') | intervene to target]
        
        Returns: (best_q_value, best_target)
        """
        if t >= self.T:
            return self.compute_Y(X), None
        
        best_q = -np.inf
        best_target = None
        
        # Try all intervention targets
        for X_target in range(self.safe_min, self.safe_max + 1):
            q_value = 0.0
            
            for U_next in self.U_values:
                prob = self.U_probs[self.U_values.index(U_next)]
                X_next = self.transition(X_target, U, U_next)
                next_state = self.get_state_key(t+1, X_next, U_next, 1)  # I=1 after intervention
                q_value += prob * self.V.get(next_state, 0)
            
            if q_value > best_q:
                best_q = q_value
                best_target = X_target
        
        return best_q, best_target
    
    def train_q_value_iteration(self, verbose=True):
        """
        Train using Q-Value Iteration (backward induction in RL terms)
        
        This is EXACTLY equivalent to your backward induction,
        but framed as an RL algorithm!
        """
        if verbose:
            print("Training with Q-Value Iteration (RL formulation of backward induction)...")
        
        # Initialize terminal values
        for XT in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(XT)
            for UT in self.U_values:
                for I in [0, 1]:
                    terminal_state = self.get_state_key(self.T, XT, UT, I)
                    self.V[terminal_state] = Y
                    self.Q[(terminal_state, 'WAIT')] = Y
        
        # Backward iteration from T-1 to 1
        for t in range(self.T - 1, 0, -1):
            if verbose:
                print(f"  Processing time t={t}...")
            
            for X in range(self.X_min, self.X_max + 1):
                for U in self.U_values:
                    
                    # ========================================================
                    # State: (t, X, U, I=0) - haven't intervened yet
                    # ========================================================
                    state_unused = self.get_state_key(t, X, U, 0)
                    
                    if X < self.safe_min or X > self.safe_max:
                        # Death boundary - no useful actions
                        self.Q[(state_unused, 'WAIT')] = 0.0
                        self.Q[(state_unused, 'INTERVENE')] = 0.0
                        self.V[state_unused] = 0.0
                        self.policy[state_unused] = 'no_action'
                    else:
                        # Compute Q-values for both actions
                        q_wait = self.compute_q_wait(t, X, U, I=0)
                        q_intervene, best_target = self.compute_q_intervene(t, X, U)
                        
                        self.Q[(state_unused, 'WAIT')] = q_wait
                        self.Q[(state_unused, 'INTERVENE')] = q_intervene
                        
                        # Optimal action is the one with higher Q-value
                        if q_intervene > q_wait:
                            self.V[state_unused] = q_intervene
                            self.policy[state_unused] = 'INTERVENE'
                            self.optimal_intervention_target[(t, X, U)] = best_target
                        else:
                            self.V[state_unused] = q_wait
                            self.policy[state_unused] = 'WAIT'
                    
                    # ========================================================
                    # State: (t, X, U, I=1) - already intervened
                    # ========================================================
                    state_used = self.get_state_key(t, X, U, 1)
                    
                    if X < self.safe_min or X > self.safe_max:
                        # Death boundary
                        self.Q[(state_used, 'WAIT')] = 0.0
                        self.V[state_used] = 0.0
                        self.policy[state_used] = 'no_action'
                    else:
                        # Can only wait
                        q_wait = self.compute_q_wait(t, X, U, I=1)
                        self.Q[(state_used, 'WAIT')] = q_wait
                        self.V[state_used] = q_wait
                        self.policy[state_used] = 'WAIT'
        
        if verbose:
            print("Training completed!")
            print(f"Total states in Q-table: {len(self.V)}")
    
    def print_results(self, output_file='RESULTS_RL.txt'):
        """Print results in same format as original code"""
        
        import os
        
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        output_file = os.path.join(script_dir, output_file)
        
        with open(output_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write("Q-VALUE ITERATION RESULTS (RL Formulation)\n")
            f.write(f"{'='*80}\n\n")
            f.write("This uses Q-Value Iteration, which is backward induction framed as RL.\n")
            f.write("Results should EXACTLY match the backward induction approach.\n\n")
            
            # ================================================================
            # POLICY BY TIME PERIOD
            # ================================================================
            f.write(f"{'='*40}\n")
            f.write("LEARNED POLICY\n")
            f.write(f"{'='*40}\n\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*40}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*40}\n")
                
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX_{t}={X:2d}:\n")
                    
                    for U in self.U_values:
                        # State (X, U, I=0)
                        state = self.get_state_key(t, X, U, 0)
                        value = self.V.get(state, 0)
                        
                        if t == self.T:
                            f.write(f"  U_{t}={U:2d}, I=0: E[Y]={value:.4f}\n")
                        else:
                            policy = self.policy.get(state, '?')
                            
                            if policy == 'INTERVENE':
                                X_target = self.optimal_intervention_target.get((t, X, U), '?')
                                f.write(f"  U_{t}={U:2d}, I=0: 🔴 INTERVENE → X becomes {X_target}, E[Y]={value:.4f}\n")
                            elif policy == 'WAIT':
                                f.write(f"  U_{t}={U:2d}, I=0: ⚪ WAIT, E[Y]={value:.4f}\n")
                            elif policy == 'no_action':
                                f.write(f"  U_{t}={U:2d}, I=0: ☠️  DEATH (boundary state), E[Y]={value:.4f}\n")
                            else:
                                f.write(f"  U_{t}={U:2d}, I=0: ?, E[Y]={value:.4f}\n")
                        
                        # State (X, U, I=1)
                        state_used = self.get_state_key(t, X, U, 1)
                        value_used = self.V.get(state_used, 0)
                        policy_used = self.policy.get(state_used, '?')
                        
                        if policy_used == 'no_action' and (X < self.safe_min or X > self.safe_max):
                            f.write(f"  U_{t}={U:2d}, I=1: ☠️  DEATH (boundary state), E[Y]={value_used:.4f}\n")
                        else:
                            f.write(f"  U_{t}={U:2d}, I=1: no_action, E[Y]={value_used:.4f}\n")
            
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


# Run RL
if __name__ == "__main__":
    rl_model = RLOptimalStopping(X0=10)
    rl_model.train_q_value_iteration(verbose=True)
    rl_model.print_results('RESULTS_RL.txt')
    
    print("\n" + "="*80)
    print("Q-Value Iteration Complete!")
    print("="*80)
