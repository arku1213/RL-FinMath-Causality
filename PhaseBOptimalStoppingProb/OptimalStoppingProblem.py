import numpy as np

class CausalOptimalStopping:
    """
    Causal Optimal Stopping with 3 Algorithms:
    1. Standard Optimal Stopping
    2. Bounds under Incomplete Information
    3. Robust Optimal Stopping
    
    NEW: Mandatory intervention at boundary states {1, 2, 18, 19, 20}
    NEW: Intervention timing analysis via simulation
    """
    
    def __init__(self, X0=10, prob_uncertainty=0.15, intervention_uncertainty=0.25):
        self.X0 = X0
        self.X_min, self.X_max = 1, 20
        
        # Boundary states trigger automatic intervention
        self.boundary_states = [1, 2, 18, 19, 20]
        
        # No more "healthy range" - intervention pulls toward center
        self.intervention_center = 10
        self.intervention_strength = 0.7  # How much to pull toward center
        
        # U values - NOW INCLUDING 0
        self.U_values = [-3, -2, -1, 0, 1, 2, 3]
        self.U_probs = self._compute_U_probabilities()
        
        # Uncertainty bounds
        self.prob_uncertainty = prob_uncertainty
        self.intervention_uncertainty = intervention_uncertainty
        self.U_probs_lower, self.U_probs_upper = self._compute_probability_bounds()
        
        self.T = 6
        
        # Storage
        self.value_function = {}         # Standard Optimal Stopping
        self.policy = {}                 # Standard Optimal Stopping
        self.value_lower = {}            # Bounds under Incomplete Information
        self.value_upper = {}            # Bounds under Incomplete Information
        self.robust_policy = {}          # Robust Optimal Stopping
        
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
        return probs / probs.sum()  # Ensure normalization
    
    def _compute_probability_bounds(self):
        eps = self.prob_uncertainty
        probs_lower = np.maximum(0, self.U_probs - eps)
        probs_lower = probs_lower / probs_lower.sum()
        probs_upper = np.minimum(1, self.U_probs + eps)
        probs_upper = probs_upper / probs_upper.sum()
        return probs_lower, probs_upper
    
    def apply_intervention(self, X):
        """
        Intervention pulls X toward center without a specific "healthy range"
        
        Formula: X_new = (1 - α) * X + α * center
        where α = intervention_strength = 0.7
        
        Then clip to safe bounds [3, 17]
        """
        X_pulled = (1 - self.intervention_strength) * X + self.intervention_strength * self.intervention_center
        X_intervened = int(np.round(X_pulled))
        X_intervened = np.clip(X_intervened, 3, 17)
        return X_intervened
    
    def transition(self, X, U_current, U_next, intervene=False, intervention_params=None):
        """
        State transition with optional intervention
        
        If intervene=True: Apply intervention first, then transition
        """
        if intervene:
            if intervention_params is None:
                X = self.apply_intervention(X)
            else:
                # For robust analysis with uncertainty
                center, strength = intervention_params
                X_pulled = (1 - strength) * X + strength * center
                X = int(np.round(X_pulled))
                X = np.clip(X, 3, 17)
        
        # Boundary behavior - absorbing states
        if X < 3 or X > 17:
            return int(np.clip(X, self.X_min, self.X_max))
        
        # Standard transition
        X_next = np.floor(X + U_current/3 + U_next/2)
        return int(np.clip(X_next, self.X_min, self.X_max))
    
    def compute_Y(self, X6):
        """
        Binary outcome based on final health marker X₆
        
        Y = 0 if X₆ ∈ {1, 2, 18, 19, 20} (extreme failure zones)
        Y = 1 if X₆ ∈ {3, 4, ..., 16, 17} (success zone)
        
        Returns: 0 or 1
        """
        if X6 in [1, 2, 18, 19, 20]:
            return 0  # Failure
        else:
            return 1  # Success
    
    # ========================================================================
    # STANDARD OPTIMAL STOPPING
    # ========================================================================
    
    def solve_standard_optimal_stopping(self):
        """Standard Optimal Stopping: Find optimal intervention time
        
        NEW: Automatic intervention at boundary states
        """
        
        # Terminal condition at t=6
        for X6 in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(X6)
            for U6 in self.U_values:
                self.value_function[(6, X6, U6, 0)] = Y
                self.value_function[(6, X6, U6, 1)] = Y
        
        # Backward induction
        for t in range(5, 0, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    
                    # ============================================================
                    # BOUNDARY CHECK: Automatic intervention at boundary
                    # ============================================================
                    if Xt in self.boundary_states:
                        # Already intervened (I=1) - stuck at boundary, certain death
                        self.value_function[(t, Xt, Ut, 1)] = 0.0
                        self.policy[(t, Xt, Ut, 1)] = 'no_action'
                        
                        # Haven't intervened (I=0) - AUTOMATIC INTERVENTION
                        intervene_val = self._intervention_value(t, Xt, Ut, 'standard')
                        self.value_function[(t, Xt, Ut, 0)] = intervene_val
                        self.policy[(t, Xt, Ut, 0)] = 'AUTO_INTERVENE'
                    
                    # ============================================================
                    # NON-BOUNDARY: Normal optimal stopping logic
                    # ============================================================
                    else:
                        # Already intervened (I=1) - can only continue
                        cont_used = self._continuation_value(t, Xt, Ut, True, 'standard')
                        self.value_function[(t, Xt, Ut, 1)] = cont_used
                        self.policy[(t, Xt, Ut, 1)] = 'no_action'
                        
                        # Haven't intervened (I=0) - OPTIMAL STOPPING DECISION
                        intervene_val = self._intervention_value(t, Xt, Ut, 'standard')
                        wait_val = self._continuation_value(t, Xt, Ut, False, 'standard')
                        
                        if intervene_val > wait_val:
                            self.value_function[(t, Xt, Ut, 0)] = intervene_val
                            self.policy[(t, Xt, Ut, 0)] = 'INTERVENE'
                        else:
                            self.value_function[(t, Xt, Ut, 0)] = wait_val
                            self.policy[(t, Xt, Ut, 0)] = 'WAIT'
        
        # Value at t=0
        V0 = 0.0
        for U1 in self.U_values:
            prob = self.U_probs[self.U_values.index(U1)]
            X1 = int(np.floor(self.X0 + U1/2))
            X1 = np.clip(X1, self.X_min, self.X_max)
            V0 += prob * self.value_function.get((1, X1, U1, 0), 0)
        
        return V0
    
    def _intervention_value(self, t, Xt, Ut, mode='standard'):
        """Expected value if we intervene now"""
        Xt_int = self.apply_intervention(Xt)
        total = 0.0
        
        for U_next in self.U_values:
            prob = self.U_probs[self.U_values.index(U_next)]
            X_next = self.transition(Xt_int, Ut, U_next, intervene=False)
            
            if mode == 'standard':
                future = self.value_function.get((t+1, X_next, U_next, 1), 0)
            total += prob * future
        
        return total
    
    def _continuation_value(self, t, Xt, Ut, already_intervened, mode='standard'):
        """Expected value if we wait (don't intervene now)"""
        total = 0.0
        
        for U_next in self.U_values:
            prob = self.U_probs[self.U_values.index(U_next)]
            X_next = self.transition(Xt, Ut, U_next, intervene=False)
            
            I_next = 1 if already_intervened else 0
            if mode == 'standard':
                future = self.value_function.get((t+1, X_next, U_next, I_next), 0)
            total += prob * future
        
        return total
    
    # ========================================================================
    # BOUNDS UNDER INCOMPLETE INFORMATION
    # ========================================================================
    
    def solve_bounds_incomplete_information(self):
        """Bounds under Incomplete Information: Compute upper and lower bounds on E[Y]
        
        NEW: Automatic intervention at boundary states
        """
        
        # Terminal condition
        for X6 in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(X6)
            for U6 in self.U_values:
                self.value_lower[(6, X6, U6, 0)] = Y
                self.value_lower[(6, X6, U6, 1)] = Y
                self.value_upper[(6, X6, U6, 0)] = Y
                self.value_upper[(6, X6, U6, 1)] = Y
        
        # Backward induction
        for t in range(5, 0, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    
                    # ============================================================
                    # BOUNDARY CHECK: Automatic intervention at boundary
                    # ============================================================
                    if Xt in self.boundary_states:
                        # Already intervened (I=1) - certain death
                        self.value_lower[(t, Xt, Ut, 1)] = 0.0
                        self.value_upper[(t, Xt, Ut, 1)] = 0.0
                        
                        # Haven't intervened (I=0) - automatic intervention
                        int_lower = self._intervention_bounded(t, Xt, Ut, 'lower')
                        int_upper = self._intervention_bounded(t, Xt, Ut, 'upper')
                        self.value_lower[(t, Xt, Ut, 0)] = int_lower
                        self.value_upper[(t, Xt, Ut, 0)] = int_upper
                    
                    # ============================================================
                    # NON-BOUNDARY: Normal bounds logic
                    # ============================================================
                    else:
                        # Already intervened
                        cont_lower = self._continuation_bounded(t, Xt, Ut, True, 'lower')
                        cont_upper = self._continuation_bounded(t, Xt, Ut, True, 'upper')
                        self.value_lower[(t, Xt, Ut, 1)] = cont_lower
                        self.value_upper[(t, Xt, Ut, 1)] = cont_upper
                        
                        # Haven't intervened
                        int_lower = self._intervention_bounded(t, Xt, Ut, 'lower')
                        int_upper = self._intervention_bounded(t, Xt, Ut, 'upper')
                        wait_lower = self._continuation_bounded(t, Xt, Ut, False, 'lower')
                        wait_upper = self._continuation_bounded(t, Xt, Ut, False, 'upper')
                        
                        self.value_lower[(t, Xt, Ut, 0)] = max(int_lower, wait_lower)
                        self.value_upper[(t, Xt, Ut, 0)] = max(int_upper, wait_upper)
        
        # Value at t=0
        V0_lower, V0_upper = 0.0, 0.0
        for U1 in self.U_values:
            prob_lower = self.U_probs_lower[self.U_values.index(U1)]
            prob_upper = self.U_probs_upper[self.U_values.index(U1)]
            X1 = int(np.floor(self.X0 + U1/2))
            X1 = np.clip(X1, self.X_min, self.X_max)
            V0_lower += prob_lower * self.value_lower.get((1, X1, U1, 0), 0)
            V0_upper += prob_upper * self.value_upper.get((1, X1, U1, 0), 0)
        
        return V0_lower, V0_upper
    
    def _intervention_bounded(self, t, Xt, Ut, bound_type):
        """Intervention value under uncertainty"""
        scenarios = []
        
        # Try different intervention parameters (uncertainty in effectiveness)
        intervention_params_list = [
            (self.intervention_center, self.intervention_strength),
            (self.intervention_center, self.intervention_strength * (1 - self.intervention_uncertainty)),
            (self.intervention_center, self.intervention_strength * (1 + self.intervention_uncertainty))
        ]
        
        for int_params in intervention_params_list:
            for use_lower_probs in [True, False]:
                probs = self.U_probs_lower if use_lower_probs else self.U_probs_upper
                
                # Apply intervention with these parameters
                center, strength = int_params
                X_pulled = (1 - strength) * Xt + strength * center
                Xt_int = int(np.round(X_pulled))
                Xt_int = np.clip(Xt_int, 3, 17)
                
                total = 0.0
                for U_next in self.U_values:
                    prob = probs[self.U_values.index(U_next)]
                    X_next = self.transition(Xt_int, Ut, U_next, intervene=False)
                    
                    if bound_type == 'lower':
                        future = self.value_lower.get((t+1, X_next, U_next, 1), 0)
                    else:
                        future = self.value_upper.get((t+1, X_next, U_next, 1), 0)
                    
                    total += prob * future
                
                scenarios.append(total)
        
        return min(scenarios) if bound_type == 'lower' else max(scenarios)
    
    def _continuation_bounded(self, t, Xt, Ut, already_intervened, bound_type):
        """Continuation value under uncertainty"""
        scenarios = []
        
        for use_lower_probs in [True, False]:
            probs = self.U_probs_lower if use_lower_probs else self.U_probs_upper
            total = 0.0
            
            for U_next in self.U_values:
                prob = probs[self.U_values.index(U_next)]
                X_next = self.transition(Xt, Ut, U_next, intervene=False)
                
                I_next = 1 if already_intervened else 0
                if bound_type == 'lower':
                    future = self.value_lower.get((t+1, X_next, U_next, I_next), 0)
                else:
                    future = self.value_upper.get((t+1, X_next, U_next, I_next), 0)
                
                total += prob * future
            
            scenarios.append(total)
        
        return min(scenarios) if bound_type == 'lower' else max(scenarios)
    
    # ========================================================================
    # ROBUST OPTIMAL STOPPING
    # ========================================================================
    
    def solve_robust_optimal_stopping(self):
        """Robust Optimal Stopping: Robust policy under uncertainty
        
        NEW: Automatic intervention at boundary states
        """
        
        # First solve Bounds if not already done
        if not self.value_lower:
            self.solve_bounds_incomplete_information()
        
        # Derive robust policy
        for t in range(1, self.T + 1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    
                    # ============================================================
                    # BOUNDARY CHECK: Automatic intervention at boundary
                    # ============================================================
                    if Xt in self.boundary_states:
                        self.robust_policy[(t, Xt, Ut, 1)] = 'no_action'
                        self.robust_policy[(t, Xt, Ut, 0)] = 'AUTO_INTERVENE'
                    
                    # ============================================================
                    # NON-BOUNDARY: Normal robust logic
                    # ============================================================
                    else:
                        self.robust_policy[(t, Xt, Ut, 1)] = 'no_action'
                        
                        # Get bounds
                        int_lower = self._intervention_bounded(t, Xt, Ut, 'lower')
                        int_upper = self._intervention_bounded(t, Xt, Ut, 'upper')
                        wait_lower = self._continuation_bounded(t, Xt, Ut, False, 'lower')
                        wait_upper = self._continuation_bounded(t, Xt, Ut, False, 'upper')
                        
                        # Robust decision rule
                        if int_lower > wait_upper:
                            self.robust_policy[(t, Xt, Ut, 0)] = 'INTERVENE'
                        elif int_upper < wait_lower:
                            self.robust_policy[(t, Xt, Ut, 0)] = 'WAIT'
                        else:
                            self.robust_policy[(t, Xt, Ut, 0)] = 'AMBIGUOUS'
    
    # ========================================================================
    # INTERVENTION TIMING ANALYSIS
    # ========================================================================
    
    def characterize_intervention_boundary(self):
        """
        Option B: Characterize the intervention decision boundary at each time
        
        Returns which (X, U) combinations trigger intervention at each time t
        """
        boundary = {}
        
        for t in range(1, 6):  # t=1,2,3,4,5
            boundary[t] = {
                'auto_intervene': [],  # Boundary states
                'intervene': [],       # Policy says INTERVENE
                'wait': []             # Policy says WAIT (for reference)
            }
            
            for X in range(self.X_min, self.X_max + 1):
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), 'UNKNOWN')
                    
                    if policy == 'AUTO_INTERVENE':
                        boundary[t]['auto_intervene'].append((X, U))
                    elif policy == 'INTERVENE':
                        boundary[t]['intervene'].append((X, U))
                    elif policy == 'WAIT':
                        boundary[t]['wait'].append((X, U))
        
        return boundary
    
    def compute_path_counterfactuals(self, n_paths=10):
        """
        Option C: For sample paths, compute E[Y] under different intervention times
        
        Shows what would happen if we intervened at different times along specific paths
        """
        np.random.seed(42)  # For reproducibility
        path_analyses = []
        
        for path_id in range(n_paths):
            # Generate a single path
            X_sequence = [self.X0]
            U_sequence = []
            I = 0
            
            # Generate U1 and compute X1
            U1 = np.random.choice(self.U_values, p=self.U_probs)
            U_sequence.append(U1)
            X1 = int(np.floor(self.X0 + U1/2))
            X1 = np.clip(X1, self.X_min, self.X_max)
            X_sequence.append(X1)
            
            # Generate rest of path (t=2 to t=6)
            X = X1
            U_prev = U1
            for t in range(2, 7):
                U_current = np.random.choice(self.U_values, p=self.U_probs)
                U_sequence.append(U_current)
                
                # Transition
                if X < 3 or X > 17:
                    X_next = X  # Boundary absorption
                else:
                    X_next = int(np.floor(X + U_prev/3 + U_current/2))
                    X_next = np.clip(X_next, self.X_min, self.X_max)
                
                X_sequence.append(X_next)
                X = X_next
                U_prev = U_current
            
            # Now compute counterfactuals: what if we intervened at each time?
            counterfactuals = {}
            
            # Counterfactual: Never intervene
            counterfactuals['never'] = self._evaluate_path_outcome(
                X_sequence, U_sequence, intervene_at=None
            )
            
            # Counterfactual: Intervene at each time t=1,2,3,4,5
            for t_intervene in range(1, 6):
                counterfactuals[f't={t_intervene}'] = self._evaluate_path_outcome(
                    X_sequence, U_sequence, intervene_at=t_intervene
                )
            
            # Optimal intervention time for this path
            optimal_t = self._find_optimal_intervention_time(X_sequence, U_sequence)
            
            path_analyses.append({
                'path_id': path_id,
                'X_sequence': X_sequence,
                'U_sequence': U_sequence,
                'counterfactuals': counterfactuals,
                'optimal_t': optimal_t
            })
        
        return path_analyses
    
    def _evaluate_path_outcome(self, X_sequence, U_sequence, intervene_at):
        """
        Evaluate expected outcome for a path given intervention at specific time
        (or never if intervene_at=None)
        
        Uses the value function to get expected continuation value
        """
        if intervene_at is None:
            # Never intervene - follow path with I=1 (as if already used)
            # Start from t=1
            X1 = X_sequence[1]
            U1 = U_sequence[0]
            return self.value_function.get((1, X1, U1, 1), 0)
        
        else:
            # Intervene at specific time
            # Before intervention: I=0, after: I=1
            Xt = X_sequence[intervene_at]
            Ut = U_sequence[intervene_at - 1] if intervene_at > 0 else U_sequence[0]
            
            # Get value of intervening at this time
            return self._intervention_value(intervene_at, Xt, Ut, 'standard')
    
    def _find_optimal_intervention_time(self, X_sequence, U_sequence):
        """
        Find optimal intervention time by following the policy along the path
        """
        I = 0
        for t in range(1, 6):
            X = X_sequence[t]
            U = U_sequence[t-1]
            
            if I == 0:
                policy = self.policy.get((t, X, U, 0), 'WAIT')
                if policy in ['INTERVENE', 'AUTO_INTERVENE']:
                    return t
        
        return None  # Never intervene
    
    def compute_intervention_timing(self, n_simulations=10000):
        """
        Simulate optimal policy from X0 and analyze when intervention occurs
        
        Returns:
            dict with timing statistics
        """
        intervention_times = []
        intervention_states = []  # Track (X, U) at intervention
        
        for sim in range(n_simulations):
            X = self.X0
            I = 0  # Start with intervention available
            
            # Generate U1 and transition to t=1
            U_prev = np.random.choice(self.U_values, p=self.U_probs)
            X = int(np.floor(X + U_prev/2))
            X = np.clip(X, self.X_min, self.X_max)
            
            for t in range(1, 6):  # t=1,2,3,4,5
                U_current = U_prev
                
                # Check if at boundary -> auto-intervention
                if X in self.boundary_states and I == 0:
                    intervention_times.append(t)
                    intervention_states.append((X, U_current))
                    I = 1
                    X = self.apply_intervention(X)
                
                # Otherwise check policy
                elif I == 0:
                    policy = self.policy.get((t, X, U_current, 0), 'WAIT')
                    
                    if policy == 'INTERVENE' or policy == 'AUTO_INTERVENE':
                        intervention_times.append(t)
                        intervention_states.append((X, U_current))
                        I = 1
                        X = self.apply_intervention(X)
                
                # Transition to next period (if not at t=5)
                if t < 5:
                    U_next = np.random.choice(self.U_values, p=self.U_probs)
                    if X < 3 or X > 17:
                        # Boundary absorption
                        pass
                    else:
                        X = int(np.floor(X + U_current/3 + U_next/2))
                        X = np.clip(X, self.X_min, self.X_max)
                    U_prev = U_next
            
            # If never intervened, record as None
            if I == 0:
                intervention_times.append(None)
        
        # Compute statistics
        intervened = [t for t in intervention_times if t is not None]
        never_intervened = intervention_times.count(None)
        
        time_distribution = {
            1: intervention_times.count(1),
            2: intervention_times.count(2),
            3: intervention_times.count(3),
            4: intervention_times.count(4),
            5: intervention_times.count(5),
            'never': never_intervened
        }
        
        return {
            'mean_time': np.mean(intervened) if intervened else None,
            'median_time': np.median(intervened) if intervened else None,
            'std_time': np.std(intervened) if intervened else None,
            'intervention_rate': len(intervened) / n_simulations,
            'never_rate': never_intervened / n_simulations,
            'time_distribution': time_distribution,
            'n_simulations': n_simulations,
            'intervention_states': intervention_states
        }
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    def print_results(self, output_file='RESULTS.txt', n_simulations=10000):
        """
        Print results - saves ALL analysis to file including timing analysis
        
        Shows all 280 states per time period (20 X × 7 U × 2 I)
        """
        
        # Solve all algorithms
        V2 = self.solve_standard_optimal_stopping()
        V3_lower, V3_upper = self.solve_bounds_incomplete_information()
        self.solve_robust_optimal_stopping()
        
        # Compute intervention timing
        timing = self.compute_intervention_timing(n_simulations)
        
        # Compute intervention boundary characterization (Option B)
        boundary = self.characterize_intervention_boundary()
        
        # Compute path counterfactuals (Option C)
        path_counterfactuals = self.compute_path_counterfactuals(n_paths=10)
        
        # Open file for ALL output
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
                
                # Loop through ALL X values (no filtering)
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX_{t}={X:2d}:\n")
                    
                    # Loop through ALL U values
                    for U in self.U_values:
                        # State (X, U, I=0) - haven't intervened yet
                        value = self.value_function.get((t, X, U, 0), 0)
                        
                        if t == 6:
                            # Terminal time - no policy, just show value
                            f.write(f"  U_{t}={U:2d}, I=0: E[Y]={value:.4f}\n")
                        else:
                            # Non-terminal - show policy
                            policy = self.policy.get((t, X, U, 0), '?')
                            if policy == 'INTERVENE':
                                X_intervened = self.apply_intervention(X)
                                f.write(f"  U_{t}={U:2d}, I=0: 🔴 INTERVENE → X becomes {X_intervened}, E[Y]={value:.4f}\n")
                            elif policy == 'AUTO_INTERVENE':
                                X_intervened = self.apply_intervention(X)
                                f.write(f"  U_{t}={U:2d}, I=0: 🚨 AUTO_INTERVENE → X becomes {X_intervened}, E[Y]={value:.4f}\n")
                            elif policy == 'WAIT':
                                f.write(f"  U_{t}={U:2d}, I=0: ⚪ WAIT, E[Y]={value:.4f}\n")
                            else:
                                f.write(f"  U_{t}={U:2d}, I=0: ?, E[Y]={value:.4f}\n")
                        
                        # State (X, U, I=1) - already intervened
                        value_used = self.value_function.get((t, X, U, 1), 0)
                        f.write(f"  U_{t}={U:2d}, I=1: no_action, E[Y]={value_used:.4f}\n")
            
            # ================================================================
            # BOUNDS UNDER INCOMPLETE INFORMATION
            # ================================================================
            f.write(f"\n\n{'='*40}\n")
            f.write("BOUNDS UNDER INCOMPLETE INFORMATION\n")
            f.write(f"{'='*40}\n\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*40}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*40}\n")
                
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX_{t}={X:2d}:\n")
                    
                    for U in self.U_values:
                        # I=0
                        V_lower = self.value_lower.get((t, X, U, 0), 0)
                        V_upper = self.value_upper.get((t, X, U, 0), 0)
                        width = V_upper - V_lower
                        f.write(f"  U_{t}={U:2d}, I=0: E[Y] ∈ [{V_lower:.4f}, {V_upper:.4f}], width={width:.4f}\n")
                        
                        # I=1
                        V_lower_used = self.value_lower.get((t, X, U, 1), 0)
                        V_upper_used = self.value_upper.get((t, X, U, 1), 0)
                        width_used = V_upper_used - V_lower_used
                        f.write(f"  U_{t}={U:2d}, I=1: E[Y] ∈ [{V_lower_used:.4f}, {V_upper_used:.4f}], width={width_used:.4f}\n")
            
            # ================================================================
            # ROBUST OPTIMAL STOPPING
            # ================================================================
            f.write(f"\n\n{'='*40}\n")
            f.write("ROBUST OPTIMAL STOPPING\n")
            f.write(f"{'='*40}\n\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*40}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*40}\n")
                
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX_{t}={X:2d}:\n")
                    
                    for U in self.U_values:
                        policy = self.robust_policy.get((t, X, U, 0), '?')
                        if policy == 'INTERVENE':
                            symbol = '🔴'
                        elif policy == 'AUTO_INTERVENE':
                            symbol = '🚨'
                        elif policy == 'WAIT':
                            symbol = '⚪'
                        elif policy == 'AMBIGUOUS':
                            symbol = '🟡'
                        else:
                            symbol = '?'
                        f.write(f"  U_{t}={U:2d}, I=0: {symbol} {policy}\n")
                        f.write(f"  U_{t}={U:2d}, I=1: no_action\n")
            
            # ================================================================
            # INTERVENTION TIMING ANALYSIS
            # ================================================================
            f.write(f"\n\n{'='*80}\n")
            f.write("INTERVENTION TIMING ANALYSIS\n")
            f.write(f"{'='*80}\n\n")
            
            # Summary statistics
            f.write(f"{'─'*80}\n")
            f.write("SUMMARY STATISTICS\n")
            f.write(f"{'─'*80}\n\n")
            
            if timing['mean_time']:
                f.write(f"Expected intervention time (given intervention occurs):\n")
                f.write(f"  Mean:   t = {timing['mean_time']:.2f}\n")
                f.write(f"  Median: t = {timing['median_time']:.1f}\n")
                f.write(f"  Std Dev:    {timing['std_time']:.2f}\n\n")
            else:
                f.write(f"No interventions occurred in any simulation!\n\n")
            
            f.write(f"Intervention rate: {timing['intervention_rate']*100:.2f}%\n")
            f.write(f"  ({int(timing['intervention_rate'] * timing['n_simulations']):,} simulations used intervention)\n\n")
            
            f.write(f"Never intervene rate: {timing['never_rate']*100:.2f}%\n")
            f.write(f"  ({timing['time_distribution']['never']:,} simulations never needed intervention)\n\n")
            
            # Time distribution
            f.write(f"{'─'*80}\n")
            f.write("DISTRIBUTION OF INTERVENTION TIMES\n")
            f.write(f"{'─'*80}\n\n")
            
            for t in range(1, 6):
                count = timing['time_distribution'][t]
                pct = count / timing['n_simulations'] * 100
                bar_length = int(pct * 0.5)  # Scale for display
                bar = '█' * bar_length
                f.write(f"  t={t}: {pct:6.2f}%  {bar}  ({count:,} times)\n")
            
            never = timing['time_distribution']['never']
            pct_never = never / timing['n_simulations'] * 100
            bar_length = int(pct_never * 0.5)
            bar = '█' * bar_length
            f.write(f"  Never: {pct_never:6.2f}%  {bar}  ({never:,} times)\n\n")
            
            f.write(f"{'='*80}\n")
            
            # ================================================================
            # OPTION B: INTERVENTION BOUNDARY CHARACTERIZATION
            # ================================================================
            f.write(f"\n\n{'='*80}\n")
            f.write("INTERVENTION BOUNDARY CHARACTERIZATION\n")
            f.write(f"{'='*80}\n\n")
            f.write("Shows which (X, U) states trigger intervention at each time t\n\n")
            
            for t in range(1, 6):
                f.write(f"{'─'*80}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*80}\n\n")
                
                # Auto-intervene (boundary states)
                auto = boundary[t]['auto_intervene']
                if auto:
                    # Group by X
                    auto_by_x = {}
                    for (X, U) in auto:
                        if X not in auto_by_x:
                            auto_by_x[X] = []
                        auto_by_x[X].append(U)
                    
                    f.write("🚨 AUTO-INTERVENE (Boundary States):\n")
                    for X in sorted(auto_by_x.keys()):
                        U_list = sorted(auto_by_x[X])
                        f.write(f"   X={X:2d}: All U values {U_list}\n")
                    f.write("\n")
                
                # Policy intervene
                intervene = boundary[t]['intervene']
                if intervene:
                    # Group by X
                    intervene_by_x = {}
                    for (X, U) in intervene:
                        if X not in intervene_by_x:
                            intervene_by_x[X] = []
                        intervene_by_x[X].append(U)
                    
                    f.write("🔴 INTERVENE (Optimal Policy):\n")
                    for X in sorted(intervene_by_x.keys()):
                        U_list = sorted(intervene_by_x[X])
                        f.write(f"   X={X:2d}: U ∈ {U_list}\n")
                    f.write("\n")
                else:
                    f.write("🔴 INTERVENE (Optimal Policy): None\n\n")
                
                # Summary
                total_intervene_states = len(auto) + len(intervene)
                total_states = len(auto) + len(intervene) + len(boundary[t]['wait'])
                pct = total_intervene_states / total_states * 100
                f.write(f"Summary: {total_intervene_states}/{total_states} states trigger intervention ({pct:.1f}%)\n\n")
            
            # ================================================================
            # OPTION C: PATH-SPECIFIC COUNTERFACTUALS
            # ================================================================
            f.write(f"\n{'='*80}\n")
            f.write("PATH-SPECIFIC COUNTERFACTUAL ANALYSIS\n")
            f.write(f"{'='*80}\n\n")
            f.write("For 10 sample paths, shows E[Y] under different intervention times\n")
            f.write("Optimal intervention time t* maximizes E[Y] for each specific path\n\n")
            
            for path in path_counterfactuals:
                f.write(f"{'─'*80}\n")
                f.write(f"PATH #{path['path_id'] + 1}\n")
                f.write(f"{'─'*80}\n\n")
                
                # Show path
                X_seq = [int(x) for x in path['X_sequence']]
                U_seq = [int(u) for u in path['U_sequence']]
                f.write(f"Health trajectory: X = {X_seq}\n")
                f.write(f"Shock sequence:    U = {U_seq}\n\n")
                
                # Show counterfactuals
                cf = path['counterfactuals']
                f.write("Expected outcomes by intervention strategy:\n")
                
                # Find max value to mark optimal
                max_val = max(cf.values())
                
                f.write(f"  Never intervene:  E[Y] = {cf['never']:.4f}")
                if cf['never'] == max_val:
                    f.write(" ← OPTIMAL")
                f.write("\n")
                
                for t in range(1, 6):
                    val = cf[f't={t}']
                    f.write(f"  Intervene at t={t}: E[Y] = {val:.4f}")
                    if val == max_val:
                        f.write(" ← OPTIMAL")
                    f.write("\n")
                
                # Show what policy actually does
                optimal_t = path['optimal_t']
                if optimal_t:
                    f.write(f"\n✓ Optimal policy intervenes at t={optimal_t}\n")
                else:
                    f.write(f"\n✓ Optimal policy never intervenes\n")
                
                f.write("\n")
            
            f.write(f"{'='*80}\n")
        
        # Simple console confirmation
        print(f"✓ Analysis complete. Results saved to: {output_file}")


# Run all algorithms
if __name__ == "__main__":
    model = CausalOptimalStopping(X0=10, prob_uncertainty=0.15, intervention_uncertainty=0.25)
    
    # Solve optimal policies and generate output file with timing analysis
    output_file = 'RESULTS.txt'
    model.print_results(output_file, n_simulations=10000)
    
    print(f"\nComplete! View results in: {output_file}")