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
    
    def solve_no_intervention_baseline(self):
        """Compute E[Y] under NO INTERVENTION policy"""
        value_no_int = {}
        
        # Terminal condition
        for XT in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(XT)
            for UT in self.U_values:
                value_no_int[(self.T, XT, UT)] = Y
        
        # Backward induction - always wait
        for t in range(self.T - 1, 0, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    if Xt < self.safe_min or Xt > self.safe_max:
                        value_no_int[(t, Xt, Ut)] = 0.0
                    else:
                        total = 0.0
                        for U_next in self.U_values:
                            prob = self.U_probs[self.U_values.index(U_next)]
                            X_next = self.transition(Xt, U_next)
                            future = value_no_int.get((t+1, X_next, U_next), 0)
                            total += prob * future
                        value_no_int[(t, Xt, Ut)] = total
        
        # Compute E[Y] from initial state
        E_Y = 0.0
        for U1 in self.U_values:
            prob = self.U_probs[self.U_values.index(U1)]
            X1 = int(np.floor(self.X0 + U1))
            X1 = np.clip(X1, self.X_min, self.X_max)
            E_Y += prob * value_no_int.get((1, X1, U1), 0)
        
        return E_Y

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
        
        UPDATED: White background for WAIT states
        """
        
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, filename)

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
                all_targets = set()
                for U in self.U_values:
                    if self.policy.get((t, X, U, 0)) == 'INTERVENE':
                        targets = self.find_all_optimal_targets(t, X, U)
                        all_targets.update(targets)
                
                # Mark target region
                for target_X in all_targets:
                    target_X_idx = target_X - self.X_min
                    if 0 <= target_X_idx < len(states):
                        target_region[target_X_idx, t_idx] = 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot policy heatmap with WHITE for WAIT
        from matplotlib.colors import ListedColormap
        
        # Custom colormap: black (death) -> white (wait) -> red (intervene)
        colors = ['black', 'white', 'lightcoral']
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
                    'White=WAIT, Red=INTERVENE, Black=DEATH, Green=Target Region',
                    fontsize=16, pad=20)
        
        # Set ticks
        ax.set_xticks(times)
        ax.set_yticks(range(self.X_min, self.X_max + 1))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', color='gray')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', label='DEATH (boundary)'),
            Patch(facecolor='white', edgecolor='gray', label='WAIT (optimal)'),
            Patch(facecolor='lightcoral', label='INTERVENE (optimal)'),
            Patch(facecolor='green', alpha=0.6, label='Intervention Target Region')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"2D heatmap saved to {filename}")
        return filename

    def simulate_trajectories(self, n_sims=5, filename='intervention_simulations.png'):
    
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, filename)
        
        print(f"Simulating {n_sims} trajectories...")
        
        if not self.policy:
            print("Solving optimal stopping problem first...")
            self.solve_optimal_stopping()
        
        # Run simulations
        trajectories = []
        
        for sim in range(n_sims):
            trajectory = {
                't': [], 
                'X': [], 
                'intervened_at': None, 
                'intervention_target': None,
                'survived': None
            }
            
            # Initialize - start at X0 at t=1
            # Initialize - apply initial transition from t=0 to t=1
            U_initial = np.random.choice(self.U_values, p=self.U_probs)
            X_current = int(np.floor(self.X0 + U_initial))
            X_current = np.clip(X_current, self.X_min, self.X_max)

            t = 1
            I_current = 0

            trajectory['t'].append(t)
            trajectory['X'].append(X_current)
            
            # Run trajectory
            while t < self.T:
                # Sample the shock that arrives NOW
                U_current = np.random.choice(self.U_values, p=self.U_probs)
                
                # Get policy for current state (t, X_current, U_current, I_current)
                policy = self.policy.get((t, X_current, U_current, I_current), 'WAIT')
                
                # Execute action
                if policy == 'INTERVENE' and I_current == 0:
                    X_target = self.optimal_intervention_target.get((t, X_current, U_current), X_current)
                    
                    # Verify target is valid
                    if X_target < self.safe_min or X_target > self.safe_max:
                        print(f"WARNING: Invalid target X'={X_target} at t={t}, X={X_current}, U={U_current}")
                        X_target = np.clip(X_target, self.safe_min, self.safe_max)
                    
                    trajectory['intervened_at'] = t
                    trajectory['intervention_target'] = X_target
                    
                    # Record the TARGET position (before shock)
                    trajectory['t'].append(t)
                    trajectory['X'].append(X_target)
                    
                    X_current = X_target
                    I_current = 1
                
                # Transition to next state using the current shock
                X_current = self.transition(X_current, U_current)
                
                t += 1
                trajectory['t'].append(t)
                trajectory['X'].append(X_current)
            
            # Check final outcome
            trajectory['survived'] = self.compute_Y(X_current)
            trajectories.append(trajectory)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create heatmap background (consistent with create_2d_heatmap)
        times = np.arange(1, self.T)
        states = np.arange(self.X_min, self.X_max + 1)
        
        policy_matrix = np.zeros((len(states), len(times)))
        target_region = np.zeros((len(states), len(times)))
        
        for t_idx, t in enumerate(times):
            for X_idx, X in enumerate(states):
                if X < self.safe_min or X > self.safe_max:
                    policy_matrix[X_idx, t_idx] = -1
                    continue
                
                intervene_count = 0
                wait_count = 0
                
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), 'WAIT')
                    if policy == 'INTERVENE':
                        intervene_count += 1
                    elif policy == 'WAIT':
                        wait_count += 1
                
                if intervene_count > wait_count:
                    policy_matrix[X_idx, t_idx] = 1
                else:
                    policy_matrix[X_idx, t_idx] = 0
                
                all_targets = set()
                for U in self.U_values:
                    if self.policy.get((t, X, U, 0)) == 'INTERVENE':
                        targets = self.find_all_optimal_targets(t, X, U)
                        all_targets.update(targets)
                
                for target_X in all_targets:
                    target_X_idx = target_X - self.X_min
                    if 0 <= target_X_idx < len(states):
                        target_region[target_X_idx, t_idx] = 1
        
        # Plot heatmap with WHITE for WAIT (same as create_2d_heatmap)
        from matplotlib.colors import ListedColormap
        colors = ['black', 'white', 'lightcoral']
        cmap = ListedColormap(colors)
        
        ax.imshow(policy_matrix, aspect='auto', origin='lower',
                extent=[times[0]-0.5, times[-1]+0.5, 
                        states[0]-0.5, states[-1]+0.5],
                cmap=cmap, vmin=-1, vmax=1, alpha=0.4, zorder=1)
        
        ax.imshow(target_region, aspect='auto', origin='lower',
                extent=[times[0]-0.5, times[-1]+0.5,
                        states[0]-0.5, states[-1]+0.5],
                cmap='Greens', alpha=0.3, vmin=0, vmax=1, zorder=1)
        
        # Death zone boundaries (same as create_2d_heatmap)
        ax.axhline(y=self.safe_min - 0.5, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, zorder=2)
        ax.axhline(y=self.safe_max + 0.5, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, zorder=2)
        
        # Plot trajectories
        colors_sim = plt.cm.tab10(np.linspace(0, 1, n_sims))

        for idx, traj in enumerate(trajectories):
            color = colors_sim[idx]
            survived = traj['survived']
            intervened_at = traj['intervened_at']
            intervention_target = traj['intervention_target']
            
            # Plot line with thicker width
            linestyle = '-' if survived else ':'
            label = f"Sim {idx+1} ({'survived' if survived else 'died'})"
            
            ax.plot(traj['t'], traj['X'], 'o-', color=color, 
                linewidth=3, markersize=6, alpha=0.9, label=label, 
                linestyle=linestyle, zorder=10)
            
            # Mark intervention
            if intervened_at is not None:
                # Find indices
                t_indices = [i for i, t_val in enumerate(traj['t']) if t_val == intervened_at]
                
                if len(t_indices) >= 2:
                    # First occurrence = pre-intervention, Second = target
                    pre_idx = t_indices[0]
                    target_idx = t_indices[1]
                    
                    X_before = traj['X'][pre_idx]
                    X_target = traj['X'][target_idx]
                    
                    # Vertical dotted line: pre-intervention → target
                    ax.plot([intervened_at, intervened_at], 
                        [X_before, X_target],
                        color=color, linestyle=':', linewidth=3, 
                        alpha=0.8, zorder=12, label='_nolegend_')
                    
                    # Star at pre-intervention
                    ax.plot(intervened_at, X_before, 
                        '*', color=color, markersize=30, markeredgecolor='black',
                        markeredgewidth=2, zorder=15, label='_nolegend_')
                    
                    # If there's a next point (post-shock), draw connecting line
                    if target_idx + 1 < len(traj['t']):
                        t_next = traj['t'][target_idx + 1]
                        X_post_shock = traj['X'][target_idx + 1]
                        
                        # Horizontal line: target → next time point showing shock effect
                        ax.plot([intervened_at, t_next], 
                            [X_target, X_post_shock],
                            color=color, linestyle='-', linewidth=3, 
                            alpha=0.9, zorder=10, label='_nolegend_')
        
        # Labels and formatting
        ax.set_xlabel('Time (t)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Health State (X)', fontsize=16, fontweight='bold')
        ax.set_title(f'{n_sims} Simulated Trajectories with Optimal Policy\n' +
                    'Stars (★) mark interventions | Solid line = survived, Dotted line = died',
                    fontsize=16, pad=20)
        
        ax.set_xticks(range(1, self.T + 1))
        ax.set_yticks(range(self.X_min, self.X_max + 1))
        ax.set_xlim(0.5, self.T + 0.5)
        ax.set_ylim(self.X_min - 0.5, self.X_max + 0.5)
        
        ax.grid(True, alpha=0.3, linestyle=':', color='gray', zorder=0)
        
        # Simple legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Simulation plot saved to {filename}")
        
        # Print summary
        print(f"\nSimulation Summary:")
        for idx, traj in enumerate(trajectories):
            status = "SURVIVED" if traj['survived'] else "DIED"
            intervention = f"intervened at t={traj['intervened_at']}, target X'={traj['intervention_target']}" if traj['intervened_at'] else "never intervened"
            print(f"  Sim {idx+1}: {status}, {intervention}, final X={traj['X'][-1]}")
        
        return trajectories
    
    def print_results(self, output_file='RESULTS.txt'):
        """Print comprehensive results to file"""
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, output_file)
        
        # Solve both problems
        if not self.policy:
            self.solve_optimal_stopping()
        
        V_no_intervention = self.solve_no_intervention_baseline()
        
        # Compute V_optimal from initial state
        V_optimal = 0.0
        for U1 in self.U_values:
            prob = self.U_probs[self.U_values.index(U1)]
            X1 = int(np.floor(self.X0 + U1))
            X1 = np.clip(X1, self.X_min, self.X_max)
            V_optimal += prob * self.value_function.get((1, X1, U1, 0), 0)
        
        ATE = V_optimal - V_no_intervention
        
        with open(output_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write("SIMPLIFIED OPTIMAL STOPPING - RESULTS\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"{'='*60}\n")
            f.write("EXPECTED OUTCOMES AT t=0 (Starting from X₀=10)\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"E[Y¹] (Optimal Intervention):     {V_optimal:.6f}\n")
            f.write(f"E[Y⁰] (No Intervention):          {V_no_intervention:.6f}\n")
            f.write(f"{'─'*60}\n")
            f.write(f"Average Treatment Effect (ATE):  {ATE:.6f}\n")
            f.write(f"  ({ATE*100:.1f} percentage point improvement)\n")
            f.write(f"{'─'*60}\n\n")
            
            f.write(f"{'='*60}\n")
            f.write("OPTIMAL STOPPING POLICY - DETAILED RESULTS\n")
            f.write(f"{'='*60}\n\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*60}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*60}\n")
                
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX_{t}={X:2d}:\n")
                    
                    for U in self.U_values:
                        value = self.value_function.get((t, X, U, 0), 0)
                        
                        if t == self.T:
                            f.write(f"  U_{t}={U:2d}, I=0: E[Y]={value:.4f}\n")
                        else:
                            policy = self.policy.get((t, X, U, 0), '?')
                            if policy == 'INTERVENE':
                                X_target = self.optimal_intervention_target.get((t, X, U), '?')
                                f.write(f"  U_{t}={U:2d}, I=0: 🔴 INTERVENE → X'={X_target}, E[Y]={value:.4f}\n")
                            elif policy == 'WAIT':
                                f.write(f"  U_{t}={U:2d}, I=0: ⚪ WAIT, E[Y]={value:.4f}\n")
                            elif policy == 'no_action':
                                f.write(f"  U_{t}={U:2d}, I=0: ☠️  DEATH (boundary), E[Y]={value:.4f}\n")
                            else:
                                f.write(f"  U_{t}={U:2d}, I=0: ?, E[Y]={value:.4f}\n")
                        
                        value_used = self.value_function.get((t, X, U, 1), 0)
                        policy_used = self.policy.get((t, X, U, 1), '?')
                        if policy_used == 'no_action' and (X < self.safe_min or X > self.safe_max):
                            f.write(f"  U_{t}={U:2d}, I=1: ☠️  DEATH (boundary), E[Y]={value_used:.4f}\n")
                        else:
                            f.write(f"  U_{t}={U:2d}, I=1: no_action, E[Y]={value_used:.4f}\n")
        
        print(f"Results saved to {output_file}")
        print(f"\nKey Findings:")
        print(f"  E[Y with optimal policy] = {V_optimal:.4f}")
        print(f"  E[Y with no intervention] = {V_no_intervention:.4f}")
        print(f"  Average Treatment Effect = {ATE:.4f} ({ATE*100:.1f}% improvement)")

    def monte_carlo_validation(self, n_sims=100000, print_results=True, save_to_file=True):
        """
        Run large-scale Monte Carlo simulation to validate backward induction
        
        Parameters:
        -----------
        n_sims : int
            Number of simulations (default 100,000)
        print_results : bool
            Whether to print detailed results to console
        save_to_file : bool
            Whether to append results to RESULTS.txt
        
        Returns:
        --------
        results : dict
            Complete statistics from Monte Carlo validation
        """
        import os
        from collections import Counter
        
        print(f"\nRunning Monte Carlo validation with {n_sims:,} simulations...")
        print("This may take a few seconds...\n")
        
        if not self.policy:
            print("Solving optimal stopping problem first...")
            self.solve_optimal_stopping()
        
        # Storage for results
        optimal_results = {
            'survived': 0,
            'died': 0,
            'intervention_times': [],
            'intervention_targets': [],
            'never_intervened': 0,
            'final_states': []
        }
        
        no_intervention_results = {
            'survived': 0,
            'died': 0,
            'final_states': []
        }
        
        # ========================================================================
        # RUN SIMULATIONS WITH OPTIMAL POLICY
        # ========================================================================
        
        for sim in range(n_sims):
            # Progress indicator
            if (sim + 1) % 10000 == 0:
                print(f"  Progress: {sim + 1:,} / {n_sims:,} simulations complete...")
            
            # Initialize - Apply initial transition from t=0 to t=1
            U_initial = np.random.choice(self.U_values, p=self.U_probs)
            X_current = int(np.floor(self.X0 + U_initial))
            X_current = np.clip(X_current, self.X_min, self.X_max)

            t = 1
            I_current = 0
            intervened = False
            intervention_time = None
            intervention_target = None
            
            # Run trajectory with optimal policy
            while t < self.T:
                # Sample shock
                U_current = np.random.choice(self.U_values, p=self.U_probs)
                
                # Get policy
                policy = self.policy.get((t, X_current, U_current, I_current), 'WAIT')
                
                # Execute action
                if policy == 'INTERVENE' and I_current == 0:
                    X_target = self.optimal_intervention_target.get((t, X_current, U_current), X_current)
                    intervened = True
                    intervention_time = t
                    intervention_target = X_target
                    X_current = X_target
                    I_current = 1
                
                # Transition
                X_current = self.transition(X_current, U_current)
                t += 1
            
            # Record outcome
            survived = self.compute_Y(X_current)
            if survived:
                optimal_results['survived'] += 1
            else:
                optimal_results['died'] += 1
            
            optimal_results['final_states'].append(X_current)
            
            if intervened:
                optimal_results['intervention_times'].append(intervention_time)
                optimal_results['intervention_targets'].append(intervention_target)
            else:
                optimal_results['never_intervened'] += 1
        
        # ========================================================================
        # RUN SIMULATIONS WITH NO INTERVENTION
        # ========================================================================
        
        print(f"\n  Running no-intervention baseline simulations...")
        
        for sim in range(n_sims):
            # Initialize
            # Initialize - Apply initial transition from t=0 to t=1
            U_initial = np.random.choice(self.U_values, p=self.U_probs)
            X_current = int(np.floor(self.X0 + U_initial))
            X_current = np.clip(X_current, self.X_min, self.X_max)

            t = 1
            
            # Run trajectory WITHOUT intervention
            while t < self.T:
                # Sample shock
                U_current = np.random.choice(self.U_values, p=self.U_probs)
                
                # Just transition (never intervene)
                X_current = self.transition(X_current, U_current)
                t += 1
            
            # Record outcome
            survived = self.compute_Y(X_current)
            if survived:
                no_intervention_results['survived'] += 1
            else:
                no_intervention_results['died'] += 1
            
            no_intervention_results['final_states'].append(X_current)
        
        # ========================================================================
        # COMPUTE STATISTICS
        # ========================================================================
        
        # Survival rates
        optimal_rate = optimal_results['survived'] / n_sims
        no_int_rate = no_intervention_results['survived'] / n_sims
        mc_ate = optimal_rate - no_int_rate
        
        # Expected values from backward induction
        V_optimal = 0.0
        for U1 in self.U_values:
            prob = self.U_probs[self.U_values.index(U1)]
            X1 = int(np.floor(self.X0 + U1))
            X1 = np.clip(X1, self.X_min, self.X_max)
            V_optimal += prob * self.value_function.get((1, X1, U1, 0), 0)
        
        V_no_int = self.solve_no_intervention_baseline()
        expected_ate = V_optimal - V_no_int
        
        # Standard errors
        se_optimal = np.sqrt(optimal_rate * (1 - optimal_rate) / n_sims)
        se_no_int = np.sqrt(no_int_rate * (1 - no_int_rate) / n_sims)
        
        # Intervention statistics
        intervention_time_dist = Counter(optimal_results['intervention_times'])
        intervention_target_dist = Counter(optimal_results['intervention_targets'])
        
        avg_intervention_time = (np.mean(optimal_results['intervention_times']) 
                                if optimal_results['intervention_times'] else None)
        
        # ========================================================================
        # PREPARE OUTPUT
        # ========================================================================
        
        output_lines = []
        output_lines.append("\n" + "="*70)
        output_lines.append(f"MONTE CARLO VALIDATION (n={n_sims:,} simulations)")
        output_lines.append("="*70 + "\n")
        
        output_lines.append("OPTIMAL POLICY RESULTS:")
        output_lines.append(f"  Survived:      {optimal_results['survived']:,} / {n_sims:,} ({optimal_rate*100:.3f}%)")
        output_lines.append(f"  Died:          {optimal_results['died']:,} / {n_sims:,} ({(1-optimal_rate)*100:.3f}%)")
        output_lines.append(f"  Expected (BI): {V_optimal*100:.3f}%")
        output_lines.append(f"  Difference:    {(optimal_rate - V_optimal)*100:+.3f}% (SE: ±{se_optimal*100:.3f}%)")
        
        if abs(optimal_rate - V_optimal) < 2 * se_optimal:
            output_lines.append(f"  Status:        ✓ Within 2 standard errors")
        else:
            output_lines.append(f"  Status:        ⚠ Outside 2 standard errors")
        
        output_lines.append("\nNO INTERVENTION BASELINE:")
        output_lines.append(f"  Survived:      {no_intervention_results['survived']:,} / {n_sims:,} ({no_int_rate*100:.3f}%)")
        output_lines.append(f"  Died:          {no_intervention_results['died']:,} / {n_sims:,} ({(1-no_int_rate)*100:.3f}%)")
        output_lines.append(f"  Expected (BI): {V_no_int*100:.3f}%")
        output_lines.append(f"  Difference:    {(no_int_rate - V_no_int)*100:+.3f}% (SE: ±{se_no_int*100:.3f}%)")
        
        if abs(no_int_rate - V_no_int) < 2 * se_no_int:
            output_lines.append(f"  Status:        ✓ Within 2 standard errors")
        else:
            output_lines.append(f"  Status:        ⚠ Outside 2 standard errors")
        
        output_lines.append("\nAVERAGE TREATMENT EFFECT:")
        output_lines.append(f"  Monte Carlo:   {mc_ate:.6f} ({mc_ate*100:.3f} pp)")
        output_lines.append(f"  Expected (BI): {expected_ate:.6f} ({expected_ate*100:.3f} pp)")
        output_lines.append(f"  Difference:    {(mc_ate - expected_ate)*100:+.3f} pp")
        
        output_lines.append("\nINTERVENTION STATISTICS:")
        output_lines.append(f"  Never intervened:  {optimal_results['never_intervened']:,} / {n_sims:,} ({optimal_results['never_intervened']/n_sims*100:.2f}%)")
        output_lines.append(f"  Avg. intervention time: t = {avg_intervention_time:.2f}" if avg_intervention_time else "  Avg. intervention time: N/A")
        
        output_lines.append("\n  Intervention Time Distribution:")
        for t in sorted(intervention_time_dist.keys()):
            count = intervention_time_dist[t]
            pct = count / n_sims * 100
            output_lines.append(f"    t={t}: {count:,} ({pct:.2f}%)")
        
        output_lines.append("\n  Most Common Targets:")
        top_targets = intervention_target_dist.most_common(5)
        for target, count in top_targets:
            pct = count / len(optimal_results['intervention_targets']) * 100 if optimal_results['intervention_targets'] else 0
            output_lines.append(f"    X'={target}: {count:,} ({pct:.2f}% of interventions)")
        
        output_lines.append("\n" + "="*70)
        output_lines.append("VALIDATION SUMMARY:")
        
        validation_passed = (
            abs(optimal_rate - V_optimal) < 2 * se_optimal and
            abs(no_int_rate - V_no_int) < 2 * se_no_int
        )
        
        if validation_passed:
            output_lines.append("✓ Monte Carlo results match backward induction within statistical error!")
            output_lines.append("✓ Model implementation is CORRECT!")
        else:
            output_lines.append("⚠ Monte Carlo results differ from backward induction")
            output_lines.append("⚠ Check model implementation")
        
        output_lines.append("="*70 + "\n")
        
        # ========================================================================
        # PRINT AND SAVE
        # ========================================================================
        
        output_text = "\n".join(output_lines)
        
        if print_results:
            print(output_text)
        
        if save_to_file:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            results_file = os.path.join(script_dir, 'RESULTS.txt')
            
            with open(results_file, 'a') as f:
                f.write("\n\n")
                f.write(output_text)
            
            print(f"Monte Carlo results appended to RESULTS.txt")
        
        # ========================================================================
        # RETURN RESULTS DICTIONARY
        # ========================================================================
        
        results = {
            'n_simulations': n_sims,
            'optimal_policy': {
                'survived': optimal_results['survived'],
                'died': optimal_results['died'],
                'rate': optimal_rate,
                'expected_rate': V_optimal,
                'standard_error': se_optimal
            },
            'no_intervention': {
                'survived': no_intervention_results['survived'],
                'died': no_intervention_results['died'],
                'rate': no_int_rate,
                'expected_rate': V_no_int,
                'standard_error': se_no_int
            },
            'ate': {
                'monte_carlo': mc_ate,
                'expected': expected_ate
            },
            'intervention_stats': {
                'never_intervened': optimal_results['never_intervened'],
                'avg_time': avg_intervention_time,
                'time_distribution': dict(intervention_time_dist),
                'target_distribution': dict(intervention_target_dist)
            },
            'validation_passed': validation_passed
        }
        
        return results

if __name__ == "__main__":
    model = SimplifiedOptimalStopping(X0=10)
    model.solve_optimal_stopping()
    model.print_results('RESULTS.txt')
    model.create_2d_heatmap('intervention_heatmap_2d.png')
    model.simulate_trajectories(n_sims=5, filename='intervention_simulations.png')
    
    # Run Monte Carlo validation
    mc_results = model.monte_carlo_validation(n_sims=100000, print_results=True, save_to_file=True)
    
    print("\nAnalysis Complete!")