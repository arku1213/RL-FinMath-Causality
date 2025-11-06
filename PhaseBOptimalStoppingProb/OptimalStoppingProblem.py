import numpy as np

class CausalOptimalStopping:

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
    
    #================================================================
    # STANDARD OPTIMAL STOPPING
    #================================================================

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
                    
                    #================================================
                    # BOUNDARY CHECK: Automatic intervention at boundary
                    #================================================
                    if Xt in self.boundary_states:
                        # Already intervened (I=1) - stuck at boundary, certain death
                        self.value_function[(t, Xt, Ut, 1)] = 0.0
                        self.policy[(t, Xt, Ut, 1)] = 'no_action'
                        
                        # Haven't intervened (I=0) - AUTOMATIC INTERVENTION
                        intervene_val = self._intervention_value(t, Xt, Ut, 'standard')
                        self.value_function[(t, Xt, Ut, 0)] = intervene_val
                        self.policy[(t, Xt, Ut, 0)] = 'AUTO_INTERVENE'
                    
                    #================================================
                    # NON-BOUNDARY: Normal optimal stopping logic
                    #================================================
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
    
    #================================================================
    # BOUNDS UNDER INCOMPLETE INFORMATION
    #================================================================

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
                    
                    #================================================
                    # BOUNDARY CHECK: Automatic intervention at boundary
                    #================================================
                    if Xt in self.boundary_states:
                        # Already intervened (I=1) - certain death
                        self.value_lower[(t, Xt, Ut, 1)] = 0.0
                        self.value_upper[(t, Xt, Ut, 1)] = 0.0
                        
                        # Haven't intervened (I=0) - automatic intervention
                        int_lower = self._intervention_bounded(t, Xt, Ut, 'lower')
                        int_upper = self._intervention_bounded(t, Xt, Ut, 'upper')
                        self.value_lower[(t, Xt, Ut, 0)] = int_lower
                        self.value_upper[(t, Xt, Ut, 0)] = int_upper

                    #================================================
                    # NON-BOUNDARY: Normal bounds logic
                    #================================================
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

    #================================================================
    # ROBUST OPTIMAL STOPPING
    #================================================================
    
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
                    
                    #================================================
                    # BOUNDARY CHECK: Automatic intervention at boundary
                    #================================================
                    if Xt in self.boundary_states:
                        self.robust_policy[(t, Xt, Ut, 1)] = 'no_action'
                        self.robust_policy[(t, Xt, Ut, 0)] = 'AUTO_INTERVENE'
                    
                    #================================================
                    # NON-BOUNDARY: Normal robust logic
                    #================================================
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
    
    #================================================================
    # OUTPUT
    #================================================================
    
    def print_results(self, output_file='RESULTS.txt'):
        """
        Print results - saves ALL analysis to file
        
        Shows all 280 states per time period (20 X × 7 U × 2 I)
        """
        
        # Solve all algorithms
        V2 = self.solve_standard_optimal_stopping()
        V3_lower, V3_upper = self.solve_bounds_incomplete_information()
        self.solve_robust_optimal_stopping()
        
        # Open file for ALL output
        with open(output_file, 'w') as f:
            #========================================================
            # STANDARD OPTIMAL STOPPING
            #========================================================
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
            
            #========================================================
            # BOUNDS UNDER INCOMPLETE INFORMATION
            #========================================================
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
            
            #========================================================
            # ROBUST OPTIMAL STOPPING
            #========================================================
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
        
        # Simple console confirmation
        print(f"✓ Analysis complete. Results saved to: {output_file}")


# Run all algorithms
if __name__ == "__main__":
    model = CausalOptimalStopping(X0=10, prob_uncertainty=0.15, intervention_uncertainty=0.25)
    
    # Solve optimal policies and generate output file
    output_file = 'RESULTS.txt'
    model.print_results(output_file)
    
    print(f"Complete! View results in: {output_file}")