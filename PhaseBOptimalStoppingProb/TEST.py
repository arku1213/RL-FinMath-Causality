import numpy as np
import matplotlib.pyplot as plt

class SimplifiedOptimalStopping:
    """
    Simplified Causal Optimal Stopping with 2D Visualization
    
    Key differences from original:
    - Simpler dynamics: X_{i+1} = floor(X_i + U_{i+1})
    - Longer horizon: T = 10
    - Focus on 2D (t, X) heatmap showing intervention regions
    """
    
    def __init__(self, X0=10):
        # ========================================================================
        # CONFIGURATION
        # ========================================================================
        
        # Time horizon
        self.T = 10  # Total time periods
        
        # State space
        self.X_min, self.X_max = 1, 20
        self.X0 = X0
        
        # Safe zone boundaries
        self.safe_min = 3
        self.safe_max = 17
        
        # Shock distribution (same as before)
        self.U_values = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        
        # Shock probabilities
        self.U_probs = self._compute_U_probabilities()
        
        # Storage
        self.value_function = {}
        self.policy = {}
        self.optimal_intervention_target = {}
        
    def _compute_U_probabilities(self):
        """Same extremely negative-biased distribution"""
        probs = np.array([0.15, 0.20, 0.25, 0.20, 0.10, 0.05, 0.03, 0.01, 0.01])
        return probs / probs.sum()
    
    def transition(self, X, U_next):
        """
        SIMPLIFIED transition dynamics: X_{i+1} = floor(X_i + U_{i+1})
        No dampening - just direct shock
        """
        # Boundary behavior - absorbing states
        if X < self.safe_min or X > self.safe_max:
            return int(np.clip(X, self.X_min, self.X_max))
        
        # Simplified transition
        X_next = np.floor(X + U_next)
        return int(np.clip(X_next, self.X_min, self.X_max))
    
    def compute_Y(self, XT):
        """Binary outcome based on final health marker"""
        if XT < self.safe_min or XT > self.safe_max:
            return 0  # Death
        else:
            return 1  # Survival
    
    def solve_optimal_stopping(self):
        """Backward induction to find optimal policy"""
        
        # Terminal condition
        for XT in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(XT)
            for UT in self.U_values:
                self.value_function[(self.T, XT, UT, 0)] = Y
                self.value_function[(self.T, XT, UT, 1)] = Y
        
        # Backward induction
        for t in range(self.T - 1, 0, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    
                    # Boundary check
                    if Xt < self.safe_min or Xt > self.safe_max:
                        self.value_function[(t, Xt, Ut, 0)] = 0.0
                        self.value_function[(t, Xt, Ut, 1)] = 0.0
                        self.policy[(t, Xt, Ut, 0)] = 'no_action'
                        self.policy[(t, Xt, Ut, 1)] = 'no_action'
                        self.optimal_intervention_target[(t, Xt, Ut)] = None
                    
                    else:
                        # Already intervened (I=1)
                        cont_used = self._continuation_value(t, Xt, Ut, True)
                        self.value_function[(t, Xt, Ut, 1)] = cont_used
                        self.policy[(t, Xt, Ut, 1)] = 'no_action'
                        
                        # Haven't intervened (I=0)
                        intervene_val, best_target = self._intervention_value_optimal(t, Xt, Ut)
                        wait_val = self._continuation_value(t, Xt, Ut, False)
                        
                        if intervene_val > wait_val:
                            self.value_function[(t, Xt, Ut, 0)] = intervene_val
                            self.policy[(t, Xt, Ut, 0)] = 'INTERVENE'
                            self.optimal_intervention_target[(t, Xt, Ut)] = best_target
                        else:
                            self.value_function[(t, Xt, Ut, 0)] = wait_val
                            self.policy[(t, Xt, Ut, 0)] = 'WAIT'
                            self.optimal_intervention_target[(t, Xt, Ut)] = None
    
    def _intervention_value_optimal(self, t, Xt, Ut):
        """Find best intervention target and its value"""
        best_value = -np.inf
        best_target = None
        
        for X_target in range(self.safe_min, self.safe_max + 1):
            total = 0.0
            
            for U_next in self.U_values:
                prob = self.U_probs[self.U_values.index(U_next)]
                X_next = self.transition(X_target, U_next)
                future = self.value_function.get((t+1, X_next, U_next, 1), 0)
                total += prob * future
            
            if total > best_value:
                best_value = total
                best_target = X_target
        
        return best_value, best_target
    
    def _continuation_value(self, t, Xt, Ut, already_intervened):
        """Expected value if we wait"""
        total = 0.0
        
        for U_next in self.U_values:
            prob = self.U_probs[self.U_values.index(U_next)]
            X_next = self.transition(Xt, U_next)
            I_next = 1 if already_intervened else 0
            future = self.value_function.get((t+1, X_next, U_next, I_next), 0)
            total += prob * future
        
        return total
    
    def find_all_optimal_targets(self, t, Xt, Ut, tolerance=1e-6):
        """
        Find ALL intervention targets that achieve (approximately) optimal value
        
        This creates the "intervention region" at later times when multiple
        targets give the same E[Y]
        
        Returns:
        --------
        optimal_targets : list of int
            All X' values that achieve the maximum E[Y]
        """
        best_value = -np.inf
        target_values = {}
        
        # Compute value for each possible target
        for X_target in range(self.safe_min, self.safe_max + 1):
            total = 0.0
            
            for U_next in self.U_values:
                prob = self.U_probs[self.U_values.index(U_next)]
                X_next = self.transition(X_target, U_next)
                future = self.value_function.get((t+1, X_next, U_next, 1), 0)
                total += prob * future
            
            target_values[X_target] = total
            if total > best_value:
                best_value = total
        
        # Find all targets within tolerance of best value
        optimal_targets = [X_target for X_target, val in target_values.items() 
                          if abs(val - best_value) < tolerance]
        
        return optimal_targets
    
    def create_2d_heatmap(self, filename='intervention_heatmap_2d.png'):
        """
        Create 2D heatmap showing:
        - X-axis: Time (t)
        - Y-axis: Health state (X)
        - Color: Policy (DEATH/WAIT/INTERVENE) averaged over U
        - Green overlay: Intervention target region
        """
        
        print("Creating 2D intervention heatmap...")
        
        # Make sure we've solved the problem
        if not self.policy:
            print("Solving optimal stopping problem first...")
            self.solve_optimal_stopping()
        
        # Create grid for heatmap
        times = np.arange(1, self.T)
        states = np.arange(self.X_min, self.X_max + 1)
        
        # Policy matrix: average policy across all U values
        policy_matrix = np.zeros((len(states), len(times)))
        
        # Intervention target region matrix
        target_region = np.zeros((len(states), len(times)))
        
        for t_idx, t in enumerate(times):
            for X_idx, X in enumerate(states):
                
                # Check if this is a death boundary
                if X < self.safe_min or X > self.safe_max:
                    policy_matrix[X_idx, t_idx] = -1  # DEATH
                    continue
                
                # Count policies across all U values
                intervene_count = 0
                wait_count = 0
                
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), 'WAIT')
                    if policy == 'INTERVENE':
                        intervene_count += 1
                    elif policy == 'WAIT':
                        wait_count += 1
                
                # Assign policy based on majority
                if intervene_count > wait_count:
                    policy_matrix[X_idx, t_idx] = 1  # INTERVENE
                else:
                    policy_matrix[X_idx, t_idx] = 0  # WAIT
                
                # Find intervention target region
                # For each (t, X), find all optimal targets across all U
                all_targets = set()
                for U in self.U_values:
                    if self.policy.get((t, X, U, 0)) == 'INTERVENE':
                        targets = self.find_all_optimal_targets(t, X, U)
                        all_targets.update(targets)
                
                # Mark target region
                for target_X in all_targets:
                    target_X_idx = target_X - self.X_min
                    if 0 <= target_X_idx < len(states):
                        target_region[target_X_idx, t_idx] += 1
        
        # Normalize target region (darker = more states target this X)
        if target_region.max() > 0:
            target_region = target_region / target_region.max()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot policy heatmap
        from matplotlib.colors import ListedColormap
        
        # Custom colormap: black (death) -> blue (wait) -> red (intervene)
        colors = ['black', 'lightblue', 'lightcoral']
        cmap = ListedColormap(colors)
        
        im = ax.imshow(policy_matrix, aspect='auto', origin='lower',
                      extent=[times[0]-0.5, times[-1]+0.5, 
                             states[0]-0.5, states[-1]+0.5],
                      cmap=cmap, vmin=-1, vmax=1, alpha=0.7)
        
        # Overlay intervention target region (green)
        target_overlay = ax.imshow(target_region, aspect='auto', origin='lower',
                                   extent=[times[0]-0.5, times[-1]+0.5,
                                          states[0]-0.5, states[-1]+0.5],
                                   cmap='Greens', alpha=0.6, vmin=0, vmax=1)
        
        # Add death zone boundaries
        ax.axhline(y=self.safe_min - 0.5, color='red', linestyle='--', 
                  linewidth=2, label='Death Boundary')
        ax.axhline(y=self.safe_max + 0.5, color='red', linestyle='--', 
                  linewidth=2)
        
        # Labels and title
        ax.set_xlabel('Time (t)', fontsize=14)
        ax.set_ylabel('Health State (X)', fontsize=14)
        ax.set_title('Intervention Policy Heatmap: Evolution Over Time\n' +
                    'Blue=WAIT, Red=INTERVENE, Black=DEATH, Green=Target Region',
                    fontsize=16, pad=20)
        
        # Set ticks
        ax.set_xticks(times)
        ax.set_yticks(range(self.X_min, self.X_max + 1))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', color='gray')
        
        # Add colorbars
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', label='DEATH (boundary)'),
            Patch(facecolor='lightblue', label='WAIT (optimal)'),
            Patch(facecolor='lightcoral', label='INTERVENE (optimal)'),
            Patch(facecolor='green', alpha=0.6, label='Intervention Target Region')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"2D heatmap saved to {filename}")
        return filename



# Run the simplified analysis
if __name__ == "__main__":
    model = SimplifiedOptimalStopping(X0=10)
    model.solve_optimal_stopping()
    model.create_2d_heatmap('intervention_heatmap_2d.png')
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
