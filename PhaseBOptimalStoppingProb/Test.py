import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class CausalOptimalStopping:
    """
    Causal Optimal Stopping with 2 Algorithms:
    1. Standard Optimal Stopping (with optimal intervention choice)
    2. Threshold Policy Extraction: Extract intervention thresholds from optimal policy
    
    Boundary states are absorbing death states (no rescue possible)
    """
    
    def __init__(self, X0=10):
        # ========================================================================
        # CONFIGURATION - Change these parameters to modify the model
        # ========================================================================
        
        # Time horizon
        self.T = 6  # Total time periods (decisions at t=1,...,T-1, outcome at t=T)
        
        # State space
        self.X_min, self.X_max = 1, 20  # Health can range from 1 to 20
        self.X0 = X0  # Starting health marker
        
        # Safe zone boundaries (death if outside this range)
        self.safe_min = 3   # Minimum safe health (below this is death)
        self.safe_max = 17  # Maximum safe health (above this is death)
        
        # Intervention parameters
        self.intervention_center = 10     # Only used as fallback
        
        # Shock distribution (negative bias = health tends to worsen)
        # U = -3:  5%,  U = -2: 15%,  U = -1: 30%
        # U =  0: 20%,  U =  1: 20%,  U =  2:  8%,  U =  3:  2%
        self.U_values = [-3, -2, -1, 0, 1, 2, 3]
        
        # ========================================================================
        # DERIVED ATTRIBUTES (computed from configuration above)
        # ========================================================================
        
        # Boundary states (death zones)
        self.boundary_states = (
            list(range(self.X_min, self.safe_min)) +           # Low death zone
            list(range(self.safe_max + 1, self.X_max + 1))     # High death zone
        )
        
        # Shock probabilities
        self.U_probs = self._compute_U_probabilities()
        
        # Storage
        self.value_function = {}         # Standard Optimal Stopping
        self.policy = {}                 # Standard Optimal Stopping
        self.optimal_intervention_target = {}  # Optimal X target for each state
        
    def _compute_U_probabilities(self):
        """
        Asymmetric distribution biased toward negative shocks
        
        U = -3:  5%
        U = -2: 15%
        U = -1: 30%  } 50% negative
        U =  0: 20%
        U =  1: 20%  } 30% positive
        U =  2:  8%
        U =  3:  2%
        
        Expected value ≈ -0.5 (negative bias)
        """
        probs = np.array([0.05, 0.15, 0.30, 0.20, 0.20, 0.08, 0.02])
        return probs / probs.sum()
    
    def transition(self, X, U_current, U_next, intervene=False):
        """
        State transition with optional intervention
        
        If intervene=True: X is already the intervention target
        
        Dynamics: X_{t+1} = floor(X_t + U_t/3 + U_{t+1}/2)
        """
        # Note: If intervene=True, X is already the target state
        # No transformation needed here
        
        # Boundary behavior - absorbing states (stay at boundary)
        if X < self.safe_min or X > self.safe_max:
            return int(np.clip(X, self.X_min, self.X_max))
        
        # Standard transition: X_{t+1} = floor(X_t + U_t/3 + U_{t+1}/2)
        X_next = np.floor(X + U_current/3 + U_next/2)
        return int(np.clip(X_next, self.X_min, self.X_max))
    
    def compute_Y(self, XT):
        """
        Binary outcome based on final health marker
        
        Y = 0 if outside safe zone (death)
        Y = 1 if in safe zone (survival)
        
        Returns: 0 or 1
        """
        if XT < self.safe_min or XT > self.safe_max:
            return 0  # Death
        else:
            return 1  # Survival
    
    # ========================================================================
    # STANDARD OPTIMAL STOPPING (with optimal intervention choice)
    # ========================================================================
    
    def solve_standard_optimal_stopping(self):
        """Standard Optimal Stopping: Find optimal intervention time and target
        
        NEW: When intervening, tries all possible targets in safe zone and picks best
        
        Boundary states are absorbing death states - no rescue
        """
        
        # Terminal condition: value at final time equals outcome
        for XT in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(XT)
            for UT in self.U_values:
                self.value_function[(self.T, XT, UT, 0)] = Y
                self.value_function[(self.T, XT, UT, 1)] = Y
        
        # Backward induction from T-1 down to 1
        for t in range(self.T - 1, 0, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    
                    # ============================================================
                    # BOUNDARY CHECK: Death states (no rescue possible)
                    # ============================================================
                    if Xt < self.safe_min or Xt > self.safe_max:
                        # At boundary - certain death regardless of I
                        self.value_function[(t, Xt, Ut, 0)] = 0.0
                        self.value_function[(t, Xt, Ut, 1)] = 0.0
                        self.policy[(t, Xt, Ut, 0)] = 'no_action'
                        self.policy[(t, Xt, Ut, 1)] = 'no_action'
                        self.optimal_intervention_target[(t, Xt, Ut)] = None
                    
                    # ============================================================
                    # NON-BOUNDARY: Normal optimal stopping logic
                    # ============================================================
                    else:
                        # Already intervened (I=1) - can only continue
                        cont_used = self._continuation_value(t, Xt, Ut, True)
                        self.value_function[(t, Xt, Ut, 1)] = cont_used
                        self.policy[(t, Xt, Ut, 1)] = 'no_action'
                        
                        # Haven't intervened (I=0) - OPTIMAL STOPPING DECISION
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
        
        # Value at t=0
        V0 = 0.0
        for U1 in self.U_values:
            prob = self.U_probs[self.U_values.index(U1)]
            X1 = int(np.floor(self.X0 + U1/2))
            X1 = np.clip(X1, self.X_min, self.X_max)
            V0 += prob * self.value_function.get((1, X1, U1, 0), 0)
        
        return V0
    
    def _intervention_value_optimal(self, t, Xt, Ut):
        """Expected value if we intervene optimally now
        
        NEW: Tries all possible intervention targets in safe zone,
        returns best value and corresponding target
        
        Returns:
        --------
        (best_value, best_target) : tuple
        """
        best_value = -np.inf
        best_target = None
        
        # Try all possible intervention targets in safe zone
        for X_target in range(self.safe_min, self.safe_max + 1):
            total = 0.0
            
            for U_next in self.U_values:
                prob = self.U_probs[self.U_values.index(U_next)]
                # Transition from intervention target
                X_next = self.transition(X_target, Ut, U_next, intervene=False)
                future = self.value_function.get((t+1, X_next, U_next, 1), 0)
                total += prob * future
            
            # Is this the best target so far?
            if total > best_value:
                best_value = total
                best_target = X_target
        
        return best_value, best_target
    
    def _continuation_value(self, t, Xt, Ut, already_intervened):
        """Expected value if we wait (don't intervene now)"""
        total = 0.0
        
        for U_next in self.U_values:
            prob = self.U_probs[self.U_values.index(U_next)]
            X_next = self.transition(Xt, Ut, U_next, intervene=False)
            I_next = 1 if already_intervened else 0
            future = self.value_function.get((t+1, X_next, U_next, I_next), 0)
            total += prob * future
        
        return total
    
    # ========================================================================
    # THRESHOLD POLICY EXTRACTION
    # ========================================================================
    
    def extract_threshold_policy(self):
        """
        Extract intervention thresholds from optimal policy
        
        For each time t, finds:
        - x_low: highest X where we intervene due to low health
        - x_high: lowest X where we intervene due to high health
        
        Returns:
        --------
        thresholds : dict
            {t: (x_low, x_high)} for each decision time
        """
        thresholds = {}
        
        for t in range(1, self.T):
            # Find intervention boundaries by looking at policy across all (X, U)
            # Average over U to get typical threshold
            
            low_boundary = self.safe_min  # Start at lowest safe state
            high_boundary = self.safe_max  # Start at highest safe state
            
            # For each X, check if intervention happens for any U
            for X in range(self.safe_min, self.safe_max + 1):
                intervene_count = 0
                total_count = 0
                
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), 'WAIT')
                    if policy == 'INTERVENE':
                        intervene_count += 1
                    total_count += 1
                
                # If majority of shocks lead to intervention, this X is in intervention region
                if intervene_count > total_count / 2:
                    if X < self.intervention_center:
                        low_boundary = max(low_boundary, X)
                    else:
                        high_boundary = min(high_boundary, X)
            
            # Threshold is boundary + 1 (intervene if X < threshold or X > threshold)
            x_low = low_boundary + 1
            x_high = high_boundary - 1
            
            thresholds[t] = (x_low, x_high)
        
        return thresholds
    
    # ========================================================================
    # 3D MESH SURFACE VISUALIZATION
    # ========================================================================
    
    def plot_intervention_boundaries_mesh(self, filename='intervention_boundaries_3d.png'):
        """
        Create 3D mesh surface plot showing intervention boundaries
        
        Two surfaces:
        - Lower boundary: intervene if X < this threshold
        - Upper boundary: intervene if X > this threshold
        
        Similar to the example image with mesh grid and color gradient
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid for time and shock
        times = np.arange(1, self.T)  # t = 1, 2, 3, 4, 5
        U_grid = np.array(self.U_values)
        
        T_grid, U_mesh = np.meshgrid(times, U_grid)
        
        # Initialize surfaces
        lower_boundary = np.zeros_like(T_grid, dtype=float)
        upper_boundary = np.zeros_like(T_grid, dtype=float)
        
        # For each (t, U) combination, find intervention boundaries
        for i, t in enumerate(times):
            for j, U in enumerate(self.U_values):
                # Find lowest X where INTERVENE (lower boundary)
                lower_X = self.safe_min
                for X in range(self.safe_min, self.intervention_center + 1):
                    policy = self.policy.get((t, X, U, 0), 'WAIT')
                    if policy == 'INTERVENE':
                        lower_X = X
                    else:
                        break  # Found the boundary
                
                # Find highest X where INTERVENE (upper boundary)
                upper_X = self.safe_max
                for X in range(self.safe_max, self.intervention_center - 1, -1):
                    policy = self.policy.get((t, X, U, 0), 'WAIT')
                    if policy == 'INTERVENE':
                        upper_X = X
                    else:
                        break  # Found the boundary
                
                lower_boundary[j, i] = lower_X
                upper_boundary[j, i] = upper_X
        
        # Plot lower boundary surface (red/pink gradient)
        surf1 = ax.plot_surface(T_grid, lower_boundary, U_mesh, 
                                cmap=cm.Reds, alpha=0.8, 
                                edgecolor='black', linewidth=0.3,
                                vmin=self.safe_min, vmax=self.safe_max)
        
        # Plot upper boundary surface (blue/purple gradient)
        surf2 = ax.plot_surface(T_grid, upper_boundary, U_mesh, 
                                cmap=cm.Blues, alpha=0.8, 
                                edgecolor='black', linewidth=0.3,
                                vmin=self.safe_min, vmax=self.safe_max)
        
        # Labels
        ax.set_xlabel('Time (t)', fontsize=14, labelpad=10)
        ax.set_ylabel('Health State (X)', fontsize=14, labelpad=10)
        ax.set_zlabel('Shock (U)', fontsize=14, labelpad=10)
        ax.set_title('Intervention Boundary Surfaces', fontsize=16, pad=20)
        
        # Set limits
        ax.set_xlim(0.5, self.T - 0.5)
        ax.set_ylim(self.safe_min - 0.5, self.safe_max + 0.5)
        ax.set_zlim(min(self.U_values) - 0.5, max(self.U_values) + 0.5)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Viewing angle for best visualization
        ax.view_init(elev=20, azim=45)
        
        # Save
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    def print_results(self, output_file='RESULTS.txt'):
        """
        Print results - saves analysis to file
        
        Shows:
        1. Standard Optimal Stopping (all states) with optimal intervention targets
        2. Threshold Policy extracted from optimal policy
        """
        
        # Solve algorithms
        V_optimal = self.solve_standard_optimal_stopping()
        
        # Extract thresholds
        thresholds = self.extract_threshold_policy()
        
        # Generate visualization
        viz_filename = self.plot_intervention_boundaries_mesh()
        
        # Open file for output
        with open(output_file, 'w') as f:
            # ================================================================
            # STANDARD OPTIMAL STOPPING
            # ================================================================
            f.write(f"{'='*40}\n")
            f.write("STANDARD OPTIMAL STOPPING\n")
            f.write(f"{'='*40}\n\n")
            
            # Show ALL states for each time period
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*40}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*40}\n")
                
                # Loop through ALL X values
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX_{t}={X:2d}:\n")
                    
                    # Loop through ALL U values
                    for U in self.U_values:
                        # State (X, U, I=0) - haven't intervened yet
                        value = self.value_function.get((t, X, U, 0), 0)
                        
                        if t == self.T:
                            # Terminal time - no policy, just show value
                            f.write(f"  U_{t}={U:2d}, I=0: E[Y]={value:.4f}\n")
                        else:
                            # Non-terminal - show policy
                            policy = self.policy.get((t, X, U, 0), '?')
                            if policy == 'INTERVENE':
                                X_target = self.optimal_intervention_target.get((t, X, U), '?')
                                f.write(f"  U_{t}={U:2d}, I=0: 🔴 INTERVENE → X becomes {X_target}, E[Y]={value:.4f}\n")
                            elif policy == 'WAIT':
                                f.write(f"  U_{t}={U:2d}, I=0: ⚪ WAIT, E[Y]={value:.4f}\n")
                            elif policy == 'no_action':
                                f.write(f"  U_{t}={U:2d}, I=0: ☠️  DEATH (boundary state), E[Y]={value:.4f}\n")
                            else:
                                f.write(f"  U_{t}={U:2d}, I=0: ?, E[Y]={value:.4f}\n")
                        
                        # State (X, U, I=1) - already intervened
                        value_used = self.value_function.get((t, X, U, 1), 0)
                        policy_used = self.policy.get((t, X, U, 1), '?')
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
                x_low, x_high = thresholds[t]
                f.write(f"At time t={t}: Intervene if X < {x_low} or X > {x_high}\n")
            
            # ================================================================
            # 3D VISUALIZATION
            # ================================================================
            f.write(f"\n\n{'='*80}\n")
            f.write("3D VISUALIZATION\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"3D visualization of intervention boundaries saved to: {viz_filename}\n")
            
            f.write(f"\n{'='*80}\n")
        
        print(f"Results saved to {output_file}")
        print(f"3D visualization saved to {viz_filename}")


# Run algorithms
if __name__ == "__main__":
    model = CausalOptimalStopping(X0=10)
    
    # Solve optimal policies and generate output files
    output_file = 'RESULTS.txt'
    model.print_results(output_file)