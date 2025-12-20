import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class CausalOptimalStopping:
    """
    Causal Optimal Stopping with 3 Algorithms:
    1. Standard Optimal Stopping (with optimal intervention choice)
    2. Threshold Policy Extraction: Extract intervention thresholds from optimal policy
    3. Optimal Intervention at Fixed Time k (mandatory intervention with optimal target)
    
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
        self.U_values = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        
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
        Extremely asymmetric distribution heavily biased toward negative shocks
        
        U = -4: 15%
        U = -3: 20%
        U = -2: 25%
        U = -1: 20%  } 80% negative
        U =  0: 10%
        U =  1:  5%  } 15% positive
        U =  2:  3%
        U =  3:  1%
        U =  4:  1%
        
        Expected value ≈ -1.8 (very strong negative bias)
        """
        probs = np.array([0.15, 0.20, 0.25, 0.20, 0.10, 0.05, 0.03, 0.01, 0.01])
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
    # NO INTERVENTION BASELINE
    # ========================================================================
    
    def solve_no_intervention_baseline(self):
        """
        Compute E[Y] under NO INTERVENTION policy (always wait)
        
        This gives us the baseline to compute Average Treatment Effect:
        ATE = E[Y^I] - E[Y^{no intervention}]
        
        Returns:
        --------
        E_Y_no_intervention : float
            Expected outcome when never intervening
        """
        # Storage for no-intervention values
        value_no_int = {}
        
        # Terminal condition
        for XT in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(XT)
            for UT in self.U_values:
                value_no_int[(self.T, XT, UT)] = Y
        
        # Backward induction - always wait (never intervene)
        for t in range(self.T - 1, 0, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    # Boundary states - certain death
                    if Xt < self.safe_min or Xt > self.safe_max:
                        value_no_int[(t, Xt, Ut)] = 0.0
                    else:
                        # Just wait (no intervention allowed)
                        total = 0.0
                        for U_next in self.U_values:
                            prob = self.U_probs[self.U_values.index(U_next)]
                            X_next = self.transition(Xt, Ut, U_next, intervene=False)
                            future = value_no_int.get((t+1, X_next, U_next), 0)
                            total += prob * future
                        
                        value_no_int[(t, Xt, Ut)] = total
        
        # Compute E[Y] from initial state
        E_Y = 0.0
        for U1 in self.U_values:
            prob = self.U_probs[self.U_values.index(U1)]
            X1 = int(np.floor(self.X0 + U1/2))
            X1 = np.clip(X1, self.X_min, self.X_max)
            E_Y += prob * value_no_int.get((1, X1, U1), 0)
        
        return E_Y
    
    # ========================================================================
    # ALGORITHM 3: OPTIMAL INTERVENTION AT FIXED TIME k
    # ========================================================================
    
    def solve_optimal_intervention_at_fixed_k(self, k):
        """
        Algorithm 3: Optimal Intervention at Fixed Time k
        
        Everyone MUST intervene at time k, but they choose the optimal target
        based on their state (Xk, Uk).
        
        Parameters:
        -----------
        k : int
            Time to intervene (1, 2, ..., T-1)
        
        Returns:
        --------
        E[Y] : float
            Expected outcome under this policy
        optimal_targets : dict
            {(Xk, Uk): optimal_target} for each state at time k
        """
        
        # Storage for this specific k
        value_k = {}
        optimal_targets_k = {}
        
        # Step 1: Backward induction from T down to k+1
        # Terminal condition
        for XT in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(XT)
            for UT in self.U_values:
                value_k[(self.T, XT, UT)] = Y
        
        # Backward from T-1 to k+1 (no intervention yet)
        for t in range(self.T - 1, k, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    # Check if at boundary
                    if Xt < self.safe_min or Xt > self.safe_max:
                        value_k[(t, Xt, Ut)] = 0.0
                        continue
                    
                    # Just transition (no intervention possible yet)
                    V = 0.0
                    for U_next in self.U_values:
                        prob = self.U_probs[self.U_values.index(U_next)]
                        X_next = self.transition(Xt, Ut, U_next, intervene=False)
                        V += prob * value_k[(t+1, X_next, U_next)]
                    
                    value_k[(t, Xt, Ut)] = V
        
        # Step 2: At time k, choose optimal intervention target for each state
        for Xk in range(self.X_min, self.X_max + 1):
            for Uk in self.U_values:
                # Check if at boundary
                if Xk < self.safe_min or Xk > self.safe_max:
                    value_k[(k, Xk, Uk)] = 0.0
                    optimal_targets_k[(Xk, Uk)] = None
                    continue
                
                # Try all intervention targets, pick best
                best_value = -np.inf
                best_target = None
                
                for r in range(self.safe_min, self.safe_max + 1):
                    # Intervene to r, then see what happens
                    V = 0.0
                    for U_next in self.U_values:
                        prob = self.U_probs[self.U_values.index(U_next)]
                        X_next = self.transition(r, Uk, U_next, intervene=False)
                        V += prob * value_k[(k+1, X_next, U_next)]
                    
                    if V > best_value:
                        best_value = V
                        best_target = r
                
                value_k[(k, Xk, Uk)] = best_value
                optimal_targets_k[(Xk, Uk)] = best_target
        
        # Step 3: Backward induction from k-1 down to 1
        for t in range(k - 1, 0, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    # Check if at boundary
                    if Xt < self.safe_min or Xt > self.safe_max:
                        value_k[(t, Xt, Ut)] = 0.0
                        continue
                    
                    # Just transition (intervention will happen at k)
                    V = 0.0
                    for U_next in self.U_values:
                        prob = self.U_probs[self.U_values.index(U_next)]
                        X_next = self.transition(Xt, Ut, U_next, intervene=False)
                        V += prob * value_k[(t+1, X_next, U_next)]
                    
                    value_k[(t, Xt, Ut)] = V
        
        # Step 4: Compute E[Y] from initial state
        E_Y = 0.0
        for U1 in self.U_values:
            prob = self.U_probs[self.U_values.index(U1)]
            X1 = int(np.floor(self.X0 + U1/2))
            X1 = np.clip(X1, self.X_min, self.X_max)
            E_Y += prob * value_k.get((1, X1, U1), 0)
        
        return E_Y, optimal_targets_k
    
    # ========================================================================
    # THRESHOLD POLICY EXTRACTION
    # ========================================================================
    
    def extract_threshold_policy_detailed(self):
        """
        Extract detailed intervention thresholds from optimal policy
        
        For each (t, U), finds all X values where intervention happens
        
        Returns:
        --------
        thresholds : dict
            {(t, U): [list of X values where intervene]}
        """
        thresholds = {}
        
        for t in range(1, self.T):
            for U in self.U_values:
                intervene_states = []
                
                for X in range(self.X_min, self.X_max + 1):
                    policy = self.policy.get((t, X, U, 0), 'WAIT')
                    if policy == 'INTERVENE':
                        intervene_states.append(X)
                
                thresholds[(t, U)] = intervene_states
        
        return thresholds
    
    def find_all_optimal_targets(self, t, Xt, Ut, tolerance=1e-6):
        """
        Find ALL intervention targets that achieve (approximately) optimal value
        
        This creates the "intervention region" at later times when multiple
        targets give the same E[Y]
        
        Parameters:
        -----------
        t : int
            Current time
        Xt : int
            Current health state
        Ut : int
            Current shock
        tolerance : float
            Tolerance for considering values equal
        
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
                X_next = self.transition(X_target, Ut, U_next, intervene=False)
                future = self.value_function.get((t+1, X_next, U_next, 1), 0)
                total += prob * future
            
            target_values[X_target] = total
            if total > best_value:
                best_value = total
        
        # Find all targets within tolerance of best value
        optimal_targets = [X_target for X_target, val in target_values.items() 
                          if abs(val - best_value) < tolerance]
        
        return optimal_targets
    
    # ========================================================================
    # 2D HEATMAP
    # ========================================================================
    
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
            self.solve_standard_optimal_stopping()
        
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
                    policy = self.policy.get((t, X, U, 0))
                    if policy == 'INTERVENE':
                        # Only add target if THIS specific (t, X, U) intervenes
                        target = self.optimal_intervention_target.get((t, X, U))
                        if target is not None:
                            all_targets.add(target)

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
        
        # Overlay intervention target region (green)
        target_overlay = ax.imshow(target_region, aspect='auto', origin='lower',
                                extent=[times[0]-0.5, times[-1]+0.5,
                                        states[0]-0.5, states[-1]+0.5],
                                cmap='Greens', alpha=0.6, vmin=0, vmax=1)

# Track which arrows we've drawn to avoid duplicates
        drawn_arrows = set()

        for t_idx, t in enumerate(times):
            for X_idx, X in enumerate(states):
                
                # Only draw arrows from RED (intervention) states
                if policy_matrix[X_idx, t_idx] == 1:  # INTERVENE
                    
                    # Find most common target across all U values for this (t, X)
                    target_counts = {}
                    
                    for U in self.U_values:
                        if self.policy.get((t, X, U, 0)) == 'INTERVENE':
                            target = self.optimal_intervention_target.get((t, X, U), None)
                            if target is not None:
                                target_counts[target] = target_counts.get(target, 0) + 1
                    
                    if target_counts:
                        # Get most common target
                        most_common_target = max(target_counts, key=target_counts.get)
                        
                        # Only draw arrow if source ≠ target (no self-loops)
                        if X != most_common_target:
                            
                            # Create unique key for this arrow to avoid duplicates
                            arrow_key = (t, X, most_common_target)
                            
                            if arrow_key not in drawn_arrows:
                                drawn_arrows.add(arrow_key)
                                
                                # Calculate arrow thickness based on "vote strength"
                                # More U values agreeing = thicker arrow
                                vote_strength = target_counts[most_common_target] / len(self.U_values)
                                arrow_width = 0.5 + 1.5 * vote_strength  # Range: 0.5 to 2.0
                                
                                # Draw arrow from (t, X) → (t, most_common_target)
                                # Offset slightly in X direction to avoid overlapping with cells
                                ax.annotate('', 
                                        xy=(t + 0.15, most_common_target),  # Arrow points TO target
                                        xytext=(t - 0.15, X),  # Arrow starts FROM intervention state
                                        arrowprops=dict(
                                            arrowstyle='-|>',  # Arrow with head
                                            color='black',      # Black arrows
                                            lw=arrow_width,
                                            alpha=0.7,
                                            shrinkA=0,
                                            shrinkB=0
                                        ),
                                        zorder=13)  # Draw on top of heatmap
        
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
            self.solve_standard_optimal_stopping()
        
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
            X_current = int(np.floor(self.X0 + U_initial/2))
            X_current = np.clip(X_current, self.X_min, self.X_max)

            t = 1
            U_prev = U_initial  # Track previous shock for dampened dynamics
            I_current = 0
            intervened = False
            intervention_time = None
            intervention_target = None
            
            # Run trajectory with optimal policy
            while t < self.T:
                # Sample current shock
                U_current = np.random.choice(self.U_values, p=self.U_probs)
                
                # Get policy
                policy = self.policy.get((t, X_current, U_prev, I_current), 'WAIT')
                
                # Execute action
                if policy == 'INTERVENE' and I_current == 0:
                    X_target = self.optimal_intervention_target.get((t, X_current, U_prev), X_current)
                    intervened = True
                    intervention_time = t
                    intervention_target = X_target
                    X_current = X_target
                    I_current = 1
                
                # Transition with dampened dynamics
                X_current = self.transition(X_current, U_prev, U_current)
                U_prev = U_current  # Update previous shock
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
            # Initialize - Apply initial transition from t=0 to t=1
            U_initial = np.random.choice(self.U_values, p=self.U_probs)
            X_current = int(np.floor(self.X0 + U_initial/2))
            X_current = np.clip(X_current, self.X_min, self.X_max)

            t = 1
            U_prev = U_initial  # Track previous shock
            
            # Run trajectory WITHOUT intervention
            while t < self.T:
                # Sample current shock
                U_current = np.random.choice(self.U_values, p=self.U_probs)
                
                # Just transition (never intervene)
                X_current = self.transition(X_current, U_prev, U_current)
                U_prev = U_current  # Update previous shock
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
            X1 = int(np.floor(self.X0 + U1/2))
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

    # ========================================================================
    # POLISHED 3D VISUALIZATIONS
    # ========================================================================
    
    def plot_intervention_boundaries_3d_polished(self, filename='intervention_boundaries_3d_polished.png'):
        """
        POLISHED 3D scatter plot with professional styling
        """
        # Set professional style
        sns.set_style("white")
        sns.set_context("paper", font_scale=1.3)
        
        # Colorblind-friendly palette
        colors_palette = sns.color_palette("colorblind")
        color_intervene = colors_palette[3]  # Red
        color_wait = colors_palette[0]       # Blue
        color_death = (0.2, 0.2, 0.2)        # Dark gray
        
        fig = plt.figure(figsize=(16, 12), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # Clean background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        print("Creating polished 3D visualization...")
        
        if not self.policy:
            print("Solving optimal stopping problem first...")
            self.solve_standard_optimal_stopping()
        
        # Collect points
        intervene_points = {'t': [], 'X': [], 'U': []}
        wait_points = {'t': [], 'X': [], 'U': []}
        death_points = {'t': [], 'X': [], 'U': []}
        
        for t in range(1, self.T):
            for X in range(self.X_min, self.X_max + 1):
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), '?')
                    
                    if policy == 'INTERVENE':
                        intervene_points['t'].append(t)
                        intervene_points['X'].append(X)
                        intervene_points['U'].append(U)
                    elif policy == 'WAIT':
                        wait_points['t'].append(t)
                        wait_points['X'].append(X)
                        wait_points['U'].append(U)
                    elif policy == 'no_action':
                        death_points['t'].append(t)
                        death_points['X'].append(X)
                        death_points['U'].append(U)
        
        # Plot WAIT (background, subtle)
        if len(wait_points['t']) > 0:
            ax.scatter(wait_points['t'], wait_points['X'], wait_points['U'],
                      c=[color_wait], marker='o', s=40, alpha=0.25, 
                      edgecolors='none', label='WAIT')
        
        # Plot DEATH (visible but not dominant)
        if len(death_points['t']) > 0:
            ax.scatter(death_points['t'], death_points['X'], death_points['U'],
                      c=[color_death], marker='x', s=50, alpha=0.7, 
                      linewidths=1.5, label='DEATH')
        
        # Plot INTERVENE (prominent)
        if len(intervene_points['t']) > 0:
            ax.scatter(intervene_points['t'], intervene_points['X'], intervene_points['U'],
                      c=[color_intervene], marker='o', s=70, alpha=0.85,
                      edgecolors='darkred', linewidths=0.5, label='INTERVENE')
        
        # Labels
        ax.set_xlabel('Time (t)', fontsize=16, labelpad=12, fontweight='normal')
        ax.set_ylabel('Health State (X)', fontsize=16, labelpad=12, fontweight='normal')
        ax.set_zlabel('Shock (U)', fontsize=16, labelpad=12, fontweight='normal')
        ax.set_title('Intervention Policy in (t, X, U) Space', 
                     fontsize=18, pad=25, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(0.5, self.T - 0.5)
        ax.set_ylim(self.X_min - 0.5, self.X_max + 0.5)
        ax.set_zlim(min(self.U_values) - 0.5, max(self.U_values) + 0.5)
        ax.set_yticks(range(0, self.X_max + 1, 2))
        
        # Better legend
        ax.legend(loc='upper left', fontsize=13, frameon=True, 
                 fancybox=True, shadow=True, framealpha=0.95)
        
        # Set better viewing angle
        ax.view_init(elev=25, azim=50)
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Polished 3D visualization saved to {filename}")
        return filename
    
    def plot_intervention_boundaries_3d_with_targets_polished(self, filename='intervention_boundaries_3d_with_targets_polished.png'):
        """
        Polished 3D visualization with intervention targets
        """
        # Set professional style
        sns.set_style("white")
        sns.set_context("paper", font_scale=1.3)
        
        # Colorblind-friendly palette
        colors_palette = sns.color_palette("colorblind")
        color_intervene = colors_palette[3]  # Red
        color_wait = colors_palette[0]       # Blue
        color_death = (0.2, 0.2, 0.2)        # Dark gray
        color_target = colors_palette[2]     # Green
        
        fig = plt.figure(figsize=(16, 12), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # Clean background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        print("Creating polished 3D visualization with targets...")
        
        if not self.policy:
            print("Solving optimal stopping problem first...")
            self.solve_standard_optimal_stopping()
        
        # Collect points
        intervene_points = {'t': [], 'X': [], 'U': [], 'target': []}
        wait_points = {'t': [], 'X': [], 'U': []}
        death_points = {'t': [], 'X': [], 'U': []}
        
        for t in range(1, self.T):
            for X in range(self.X_min, self.X_max + 1):
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), '?')
                    
                    if policy == 'INTERVENE':
                        target = self.optimal_intervention_target.get((t, X, U), None)
                        intervene_points['t'].append(t)
                        intervene_points['X'].append(X)
                        intervene_points['U'].append(U)
                        intervene_points['target'].append(target)
                    elif policy == 'WAIT':
                        wait_points['t'].append(t)
                        wait_points['X'].append(X)
                        wait_points['U'].append(U)
                    elif policy == 'no_action':
                        death_points['t'].append(t)
                        death_points['X'].append(X)
                        death_points['U'].append(U)
        
        # Plot background (WAIT and DEATH) - very subtle
        if len(wait_points['t']) > 0:
            ax.scatter(wait_points['t'], wait_points['X'], wait_points['U'],
                      c=[color_wait], marker='o', s=25, alpha=0.15,
                      edgecolors='none', label='WAIT')
        
        if len(death_points['t']) > 0:
            ax.scatter(death_points['t'], death_points['X'], death_points['U'],
                      c=[color_death], marker='x', s=35, alpha=0.4,
                      linewidths=1, label='DEATH')
        
        # Plot intervention points and targets
        if len(intervene_points['t']) > 0:
            # Current states (red circles)
            ax.scatter(intervene_points['t'], intervene_points['X'], intervene_points['U'],
                      c=[color_intervene], marker='o', s=65, alpha=0.8,
                      edgecolors='darkred', linewidths=1, label='Intervene (current state)')
            
            # Target states (green stars)
            target_X = intervene_points['target']
            ax.scatter(intervene_points['t'], target_X, intervene_points['U'],
                      c=[color_target], marker='*', s=120, alpha=0.85,
                      edgecolors='darkgreen', linewidths=1, label='Target state (X′)')
            
            # Draw arrows
            for i in range(len(intervene_points['t'])):
                t = intervene_points['t'][i]
                X_from = intervene_points['X'][i]
                X_to = target_X[i]
                U = intervene_points['U'][i]
                
                if X_from != X_to:
                    ax.plot([t, t], [X_from, X_to], [U, U],
                           color=color_target, alpha=0.5, linewidth=2.5)
        
        # Labels
        ax.set_xlabel('Time (t)', fontsize=16, labelpad=12, fontweight='normal')
        ax.set_ylabel('Health State (X)', fontsize=16, labelpad=12, fontweight='normal')
        ax.set_zlabel('Shock (U)', fontsize=16, labelpad=12, fontweight='normal')
        ax.set_title('Intervention States → Optimal Targets in (t, X, U) Space',
                     fontsize=18, pad=25, fontweight='bold')
        
        # Axis limits
        ax.set_xlim(0.5, self.T - 0.5)
        ax.set_ylim(self.X_min - 0.5, self.X_max + 0.5)
        ax.set_zlim(min(self.U_values) - 0.5, max(self.U_values) + 0.5)
        ax.set_yticks(range(0, self.X_max + 1, 2))
        
        # Legend
        ax.legend(loc='upper left', fontsize=12, frameon=True,
                 fancybox=True, shadow=True, framealpha=0.95)
        
        # Viewing angle
        ax.view_init(elev=25, azim=50)
        
        # Save
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Polished 3D visualization with targets saved to {filename}")
        return filename
    
    def plot_intervention_boundaries_2d_slices_polished(self, filename='intervention_boundaries_2d_slices_polished.png'):
        """
        Polished 2D slices for each time period
        """
        # Set professional style
        sns.set_style("white")
        sns.set_context("paper", font_scale=1.2)
        
        # Colorblind-friendly palette
        colors_palette = sns.color_palette("colorblind")
        color_intervene = colors_palette[3]
        color_wait = colors_palette[0]
        color_death = (0.2, 0.2, 0.2)
        
        if not self.policy:
            self.solve_standard_optimal_stopping()
        
        # Create figure with subplots
        n_times = self.T - 1
        fig, axes = plt.subplots(1, n_times, figsize=(5*n_times, 5), dpi=150)
        
        if n_times == 1:
            axes = [axes]
        
        for idx, t in enumerate(range(1, self.T)):
            ax = axes[idx]
            
            # Collect data for this time slice
            for X in range(self.X_min, self.X_max + 1):
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), '?')
                    
                    if policy == 'INTERVENE':
                        ax.scatter(X, U, c=[color_intervene], marker='o', s=120, 
                                 alpha=0.8, edgecolors='darkred', linewidths=1)
                    elif policy == 'WAIT':
                        ax.scatter(X, U, c=[color_wait], marker='o', s=70, 
                                 alpha=0.4, edgecolors='none')
                    elif policy == 'no_action':
                        ax.scatter(X, U, c=[color_death], marker='x', s=90, 
                                 alpha=0.8, linewidths=2)
            
            # Format subplot
            ax.set_xlabel('Health State (X)', fontsize=13, fontweight='normal')
            ax.set_ylabel('Shock (U)', fontsize=13, fontweight='normal')
            ax.set_title(f't = {t}', fontsize=15, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.set_xlim(self.X_min - 0.5, self.X_max + 0.5)
            ax.set_ylim(min(self.U_values) - 0.5, max(self.U_values) + 0.5)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_intervene, alpha=0.8, edgecolor='darkred', label='INTERVENE'),
            Patch(facecolor=color_wait, alpha=0.4, label='WAIT'),
            Patch(facecolor=color_death, alpha=0.8, label='DEATH')
        ]
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fig.legend(handles=legend_elements, loc='lower center', 
                   bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=13,
                   frameon=True, fancybox=True, shadow=True)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Polished 2D slices saved to {filename}")
        return filename
    
    def plot_intervention_boundaries_plotly(self, filename='intervention_boundaries_3d_interactive.html'):
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return None
        
        print("Creating polished interactive Plotly visualization...")
        
        if not self.policy:
            self.solve_standard_optimal_stopping()
        
        fig = go.Figure()
        
        # Collect all points by policy type
        intervene_points = {'t': [], 'X': [], 'U': [], 'target': []}
        wait_points = {'t': [], 'X': [], 'U': []}
        death_points = {'t': [], 'X': [], 'U': []}
        
        for t in range(1, self.T):
            for X in range(self.X_min, self.X_max + 1):
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), '?')
                    
                    if policy == 'INTERVENE':
                        target = self.optimal_intervention_target.get((t, X, U), None)
                        intervene_points['t'].append(t)
                        intervene_points['X'].append(X)
                        intervene_points['U'].append(U)
                        intervene_points['target'].append(target)
                    elif policy == 'WAIT':
                        wait_points['t'].append(t)
                        wait_points['X'].append(X)
                        wait_points['U'].append(U)
                    elif policy == 'no_action':
                        death_points['t'].append(t)
                        death_points['X'].append(X)
                        death_points['U'].append(U)
        
        # Plot WAIT points (blue, subtle)
        if len(wait_points['t']) > 0:
            fig.add_trace(go.Scatter3d(
                x=wait_points['t'],
                y=wait_points['X'],
                z=wait_points['U'],
                mode='markers',
                marker=dict(
                    size=4,
                    color='rgb(31, 119, 180)',  # Blue
                    opacity=0.3,
                    symbol='circle'
                ),
                name='WAIT',
                hovertemplate='<b>WAIT</b><br>Time: %{x}<br>Health: %{y}<br>Shock: %{z}<extra></extra>'
            ))
        
        # Plot DEATH points (black, visible)
        if len(death_points['t']) > 0:
            fig.add_trace(go.Scatter3d(
                x=death_points['t'],
                y=death_points['X'],
                z=death_points['U'],
                mode='markers',
                marker=dict(
                    size=5,
                    color='rgb(50, 50, 50)',  # Dark gray
                    opacity=0.7,
                    symbol='x'
                ),
                name='DEATH',
                hovertemplate='<b>DEATH</b><br>Time: %{x}<br>Health: %{y}<br>Shock: %{z}<extra></extra>'
            ))
        
        # Plot INTERVENE points (red, prominent)
        if len(intervene_points['t']) > 0:
            # Create hover text with target information
            hover_text = [f"<b>INTERVENE</b><br>Time: {t}<br>Health: {X}<br>Shock: {U}<br>Target: X'={target}"
                        for t, X, U, target in zip(intervene_points['t'], intervene_points['X'], 
                                                    intervene_points['U'], intervene_points['target'])]
            
            fig.add_trace(go.Scatter3d(
                x=intervene_points['t'],
                y=intervene_points['X'],
                z=intervene_points['U'],
                mode='markers',
                marker=dict(
                    size=7,
                    color='rgb(214, 39, 40)',  # Red
                    opacity=0.85,
                    symbol='circle',
                    line=dict(color='rgb(150, 0, 0)', width=1)
                ),
                name='INTERVENE',
                hovertemplate='%{text}<extra></extra>',
                text=hover_text
            ))
            
            # Plot target points (green stars)
            target_X = intervene_points['target']
            target_hover = [f"<b>TARGET</b><br>Time: {t}<br>Target Health: {X_t}<br>Shock: {U}"
                        for t, X_t, U in zip(intervene_points['t'], target_X, intervene_points['U'])]
            
            fig.add_trace(go.Scatter3d(
                x=intervene_points['t'],
                y=target_X,
                z=intervene_points['U'],
                mode='markers',
                marker=dict(
                    size=9,
                    color='rgb(44, 160, 44)',  # Green
                    opacity=0.8,
                    symbol='diamond',
                    line=dict(color='rgb(0, 100, 0)', width=1)
                ),
                name='Target (X′)',
                hovertemplate='%{text}<extra></extra>',
                text=target_hover
            ))
            
            # Draw lines connecting intervention to targets (only where there's movement)
            for i in range(len(intervene_points['t'])):
                t = intervene_points['t'][i]
                X_from = intervene_points['X'][i]
                X_to = target_X[i]
                U = intervene_points['U'][i]
                
                if X_from != X_to:  # Only draw if there's actual movement
                    fig.add_trace(go.Scatter3d(
                        x=[t, t],
                        y=[X_from, X_to],
                        z=[U, U],
                        mode='lines',
                        line=dict(
                            color='rgb(44, 160, 44)',  # Green
                            width=5
                        ),
                        opacity=0.6,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add death zone planes (semi-transparent red)
        # Bottom death zone
        t_plane = [0.5, self.T - 0.5, self.T - 0.5, 0.5]
        u_plane = [min(self.U_values) - 0.5, min(self.U_values) - 0.5, 
                max(self.U_values) + 0.5, max(self.U_values) + 0.5]
        x_bottom = [self.safe_min - 0.5] * 4
        
        fig.add_trace(go.Mesh3d(
            x=t_plane,
            y=x_bottom,
            z=u_plane,
            color='red',
            opacity=0.15,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Top death zone
        x_top = [self.safe_max + 0.5] * 4
        fig.add_trace(go.Mesh3d(
            x=t_plane,
            y=x_top,
            z=u_plane,
            color='red',
            opacity=0.15,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Update layout with professional styling
        fig.update_layout(
            title={
                'text': '<b>Interactive Intervention Policy Visualization</b><br>' +
                        '<sub>Red circles = Intervene | Green diamonds = Target states | Lines show intervention direction</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            scene=dict(
                xaxis=dict(
                    title='<b>Time (t)</b>',
                    backgroundcolor='rgb(240, 240, 240)',
                    gridcolor='white',
                    showbackground=True,
                    gridwidth=2,
                    range=[0.5, self.T - 0.5]
                ),
                yaxis=dict(
                    title='<b>Health State (X)</b>',
                    backgroundcolor='rgb(240, 240, 240)',
                    gridcolor='white',
                    showbackground=True,
                    gridwidth=2,
                    range=[self.X_min - 0.5, self.X_max + 0.5]
                ),
                zaxis=dict(
                    title='<b>Shock (U)</b>',
                    backgroundcolor='rgb(240, 240, 240)',
                    gridcolor='white',
                    showbackground=True,
                    gridwidth=2,
                    range=[min(self.U_values) - 0.5, max(self.U_values) + 0.5]
                ),
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.3),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            width=1400,
            height=1000,
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='rgb(100, 100, 100)',
                borderwidth=2,
                font=dict(size=13, family='Arial, sans-serif')
            ),
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        # Save as HTML
        fig.write_html(filename)
        print(f"Polished interactive Plotly visualization saved to {filename}")
        print("Open this file in your web browser to interact with the 3D plot!")
        
        return filename
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    def print_results(self, output_file='RESULTS.txt'):
        """Print results - saves analysis to file"""
        
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, output_file)
        
        # Solve Algorithm 1 (Optimal intervention)
        V_optimal = self.solve_standard_optimal_stopping()
        
        # Solve No Intervention Baseline
        V_no_intervention = self.solve_no_intervention_baseline()
        
        # Compute Average Treatment Effect
        ATE = V_optimal - V_no_intervention
        
        # Extract detailed thresholds
        thresholds = self.extract_threshold_policy_detailed()
        
        # Solve Algorithm 3 for all k
        alg3_results = {}
        alg3_targets = {}
        for k in range(1, self.T):
            E_Y, targets = self.solve_optimal_intervention_at_fixed_k(k)
            alg3_results[k] = E_Y
            alg3_targets[k] = targets
        
        # Generate visualizations
        print("\n" + "="*25)
        print("Creating visualizations...")
        print("="*25 + "\n")
        
        self.plot_intervention_boundaries_3d_polished(
            filename=os.path.join(script_dir, 'intervention_boundaries_3d_polished.png'))
        self.plot_intervention_boundaries_3d_with_targets_polished(
            filename=os.path.join(script_dir, 'intervention_boundaries_3d_with_targets_polished.png'))
        self.plot_intervention_boundaries_2d_slices_polished(
            filename=os.path.join(script_dir, 'intervention_boundaries_2d_slices_polished.png'))
        self.plot_intervention_boundaries_plotly(
            filename=os.path.join(script_dir, 'intervention_boundaries_3d_interactive.html'))
        self.create_2d_heatmap(
            filename=os.path.join(script_dir, 'intervention_heatmap_2d.png'))
        
        # Open file for output
        with open(output_file, 'w') as f:
            # NEW SECTION: Expected Values at t=0
            f.write(f"{'='*25}\n")
            f.write("EXPECTED OUTCOMES AT t=0 (Starting from X₀=10)\n")
            f.write(f"{'='*25}\n\n")
            
            f.write(f"E[Y¹] (Optimal Intervention):     {V_optimal:.6f}\n")
            f.write(f"E[Y⁰] (No Intervention):          {V_no_intervention:.6f}\n")
            f.write(f"{'─'*25}\n")
            f.write(f"Average Treatment Effect (ATE):  {ATE:.6f}\n")
            f.write(f"{'─'*25}\n\n")
            
            
            f.write(f"{'='*25}\n")
            f.write("STANDARD OPTIMAL STOPPING\n")
            f.write(f"{'='*25}\n\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*25}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*25}\n")
                
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
                                f.write(f"  U_{t}={U:2d}, I=0: 🔴 INTERVENE → X becomes {X_target}, E[Y]={value:.4f}\n")
                            elif policy == 'WAIT':
                                f.write(f"  U_{t}={U:2d}, I=0: ⚪ WAIT, E[Y]={value:.4f}\n")
                            elif policy == 'no_action':
                                f.write(f"  U_{t}={U:2d}, I=0: ☠️  DEATH (boundary state), E[Y]={value:.4f}\n")
                            else:
                                f.write(f"  U_{t}={U:2d}, I=0: ?, E[Y]={value:.4f}\n")
                        
                        value_used = self.value_function.get((t, X, U, 1), 0)
                        policy_used = self.policy.get((t, X, U, 1), '?')
                        if policy_used == 'no_action' and (X < self.safe_min or X > self.safe_max):
                            f.write(f"  U_{t}={U:2d}, I=1: ☠️  DEATH (boundary state), E[Y]={value_used:.4f}\n")
                        else:
                            f.write(f"  U_{t}={U:2d}, I=1: no_action, E[Y]={value_used:.4f}\n")
            
            f.write(f"\n\n{'='*25}\n")
            f.write("THRESHOLD POLICY\n")
            f.write(f"{'='*25}\n\n")

            for t in range(1, self.T):
                f.write(f"\nAt time t={t}:\n")
                for U in self.U_values:
                    intervene_states = thresholds[(t, U)]
                    if intervene_states:
                        states_str = '{' + ', '.join(map(str, intervene_states)) + '}'
                    else:
                        states_str = '∅'
                    f.write(f"  U_{t}={U:2d}: Intervene if X_{t} ∈ {states_str}\n")

            f.write(f"\n{'─'*25}\n")
            f.write("SUMMARY\n")
            f.write(f"{'─'*25}\n\n")

            intervene_conditions = {}

            for t in range(1, self.T):
                for U in self.U_values:
                    intervene_states = thresholds[(t, U)]
                    for X in intervene_states:
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
                f.write("Never intervene (no states trigger intervention)\n")
        
        print("\n" + "="*25)
        print("Complete! All visualizations and analysis created.")
        print("="*25)


if __name__ == "__main__":
    model = CausalOptimalStopping(X0=10)
    
    # Solve optimal policies and generate output files
    output_file = 'RESULTS.txt'
    model.print_results(output_file)
    
    # Run Monte Carlo validation
    mc_results = model.monte_carlo_validation(n_sims=100000, print_results=True, save_to_file=True)
    
    print("\nAnalysis Complete!")