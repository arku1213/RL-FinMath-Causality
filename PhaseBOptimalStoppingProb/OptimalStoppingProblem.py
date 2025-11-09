import numpy as np

class CausalOptimalStopping:
    """
    Causal Optimal Stopping with 2 Algorithms:
    1. Standard Optimal Stopping
    2. Fixed Intervention at Time k
    
    Boundary states {1, 2, 18, 19, 20} are absorbing death states (no rescue possible)
    """
    
    def __init__(self, X0=10):
        self.X0 = X0
        self.X_min, self.X_max = 1, 20
        
        # Boundary states are absorbing death states - no intervention can save you
        self.boundary_states = [1, 2, 18, 19, 20]
        
        # Intervention parameters
        self.intervention_center = 10
        self.intervention_strength = 0.7
        
        # U values
        self.U_values = [-3, -2, -1, 0, 1, 2, 3]
        self.U_probs = self._compute_U_probabilities()
        
        self.T = 6
        
        # Storage
        self.value_function = {}         # Standard Optimal Stopping
        self.policy = {}                 # Standard Optimal Stopping
        self.value_fixed = {}            # Fixed Intervention at Time k
        
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
    
    def apply_intervention(self, X):
        """
        Intervention pulls X toward center
        
        Formula: X_new = (1 - α) * X + α * center
        where α = intervention_strength = 0.7
        
        Then clip to safe bounds [3, 17]
        """
        X_pulled = (1 - self.intervention_strength) * X + self.intervention_strength * self.intervention_center
        X_intervened = int(np.round(X_pulled))
        X_intervened = np.clip(X_intervened, 3, 17)
        return X_intervened
    
    def transition(self, X, U_current, U_next, intervene=False):
        """
        State transition with optional intervention
        
        If intervene=True: Apply intervention first, then transition
        """
        if intervene:
            X = self.apply_intervention(X)
        
        # Boundary behavior - absorbing states (stay at boundary)
        if X < 3 or X > 17:
            return int(np.clip(X, self.X_min, self.X_max))
        
        # Standard transition
        X_next = np.floor(X + U_current/3 + U_next/2)
        return int(np.clip(X_next, self.X_min, self.X_max))
    
    def compute_Y(self, X6):
        """
        Binary outcome based on final health marker X₆
        
        Y = 0 if X₆ ∈ {1, 2, 18, 19, 20} (death states)
        Y = 1 if X₆ ∈ {3, 4, ..., 16, 17} (survival)
        
        Returns: 0 or 1
        """
        if X6 in [1, 2, 18, 19, 20]:
            return 0  # Death
        else:
            return 1  # Survival
    
    # ========================================================================
    # STANDARD OPTIMAL STOPPING
    # ========================================================================
    
    def solve_standard_optimal_stopping(self):
        """Standard Optimal Stopping: Find optimal intervention time
        
        Boundary states {1,2,18,19,20} are absorbing death states - no rescue
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
                    # BOUNDARY CHECK: Death states (no rescue possible)
                    # ============================================================
                    if Xt in self.boundary_states:
                        # At boundary - certain death regardless of I
                        self.value_function[(t, Xt, Ut, 0)] = 0.0
                        self.value_function[(t, Xt, Ut, 1)] = 0.0
                        self.policy[(t, Xt, Ut, 0)] = 'no_action'
                        self.policy[(t, Xt, Ut, 1)] = 'no_action'
                    
                    # ============================================================
                    # NON-BOUNDARY: Normal optimal stopping logic
                    # ============================================================
                    else:
                        # Already intervened (I=1) - can only continue
                        cont_used = self._continuation_value(t, Xt, Ut, True)
                        self.value_function[(t, Xt, Ut, 1)] = cont_used
                        self.policy[(t, Xt, Ut, 1)] = 'no_action'
                        
                        # Haven't intervened (I=0) - OPTIMAL STOPPING DECISION
                        intervene_val = self._intervention_value(t, Xt, Ut)
                        wait_val = self._continuation_value(t, Xt, Ut, False)
                        
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
    
    def _intervention_value(self, t, Xt, Ut):
        """Expected value if we intervene now"""
        Xt_int = self.apply_intervention(Xt)
        total = 0.0
        
        for U_next in self.U_values:
            prob = self.U_probs[self.U_values.index(U_next)]
            X_next = self.transition(Xt_int, Ut, U_next, intervene=False)
            future = self.value_function.get((t+1, X_next, U_next, 1), 0)
            total += prob * future
        
        return total
    
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
    # FIXED INTERVENTION AT TIME k
    # ========================================================================
    
    def solve_fixed_intervention_at_k(self, k):
        """
        Evaluate E[Y^I] under fixed intervention policy: intervene at time k
        
        Parameters:
        -----------
        k : int or None
            Time to intervene (1, 2, 3, 4, 5) or None for "never intervene"
        
        Returns:
        --------
        E[Y^I] : float
            Expected outcome under this fixed policy
        """
        
        # Clear storage for this specific k
        value_fixed_k = {}
        
        # Terminal condition at t=6
        for X6 in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(X6)
            for U6 in self.U_values:
                value_fixed_k[(6, X6, U6)] = Y
        
        # Backward induction
        for t in range(5, 0, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    
                    # Check if at boundary (death state)
                    if Xt in self.boundary_states:
                        value_fixed_k[(t, Xt, Ut)] = 0.0
                        continue
                    
                    # Check if this is the intervention time
                    if t == k:
                        # Intervene at this time
                        Xt_int = self.apply_intervention(Xt)
                        
                        # Compute expected future value after intervention
                        V = 0.0
                        for U_next in self.U_values:
                            prob = self.U_probs[self.U_values.index(U_next)]
                            X_next = self.transition(Xt_int, Ut, U_next, intervene=False)
                            V += prob * value_fixed_k[(t+1, X_next, U_next)]
                        
                        value_fixed_k[(t, Xt, Ut)] = V
                    
                    else:
                        # No intervention at this time, just transition
                        V = 0.0
                        for U_next in self.U_values:
                            prob = self.U_probs[self.U_values.index(U_next)]
                            X_next = self.transition(Xt, Ut, U_next, intervene=False)
                            V += prob * value_fixed_k[(t+1, X_next, U_next)]
                        
                        value_fixed_k[(t, Xt, Ut)] = V
        
        # Compute E[Y^I] starting from X_0
        E_Y_I = 0.0
        for U1 in self.U_values:
            prob = self.U_probs[self.U_values.index(U1)]
            X1 = int(np.floor(self.X0 + U1/2))
            X1 = np.clip(X1, self.X_min, self.X_max)
            E_Y_I += prob * value_fixed_k.get((1, X1, U1), 0)
        
        return E_Y_I
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    def print_results(self, output_file='RESULTS.txt'):
        """
        Print results - saves analysis to file
        
        Shows:
        1. Standard Optimal Stopping (all states)
        2. Fixed Intervention at Time k (comparison table)
        """
        
        # Solve algorithms
        V_optimal = self.solve_standard_optimal_stopping()
        
        # Solve fixed intervention policies
        fixed_results = {}
        fixed_results['never'] = self.solve_fixed_intervention_at_k(k=None)
        for k in range(1, 6):
            fixed_results[f't={k}'] = self.solve_fixed_intervention_at_k(k=k)
        
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
                        
                        if t == 6:
                            # Terminal time - no policy, just show value
                            f.write(f"  U_{t}={U:2d}, I=0: E[Y]={value:.4f}\n")
                        else:
                            # Non-terminal - show policy
                            policy = self.policy.get((t, X, U, 0), '?')
                            if policy == 'INTERVENE':
                                X_intervened = self.apply_intervention(X)
                                f.write(f"  U_{t}={U:2d}, I=0: 🔴 INTERVENE → X becomes {X_intervened}, E[Y]={value:.4f}\n")
                            elif policy == 'WAIT':
                                f.write(f"  U_{t}={U:2d}, I=0: ⚪ WAIT, E[Y]={value:.4f}\n")
                            elif policy == 'no_action':
                                f.write(f"  U_{t}={U:2d}, I=0: ☠️  DEATH (boundary state), E[Y]={value:.4f}\n")
                            else:
                                f.write(f"  U_{t}={U:2d}, I=0: ?, E[Y]={value:.4f}\n")
                        
                        # State (X, U, I=1) - already intervened
                        value_used = self.value_function.get((t, X, U, 1), 0)
                        policy_used = self.policy.get((t, X, U, 1), '?')
                        if policy_used == 'no_action' and X in self.boundary_states:
                            f.write(f"  U_{t}={U:2d}, I=1: ☠️  DEATH (boundary state), E[Y]={value_used:.4f}\n")
                        else:
                            f.write(f"  U_{t}={U:2d}, I=1: no_action, E[Y]={value_used:.4f}\n")
            
            # ================================================================
            # FIXED INTERVENTION AT TIME k
            # ================================================================
            f.write(f"\n\n{'='*80}\n")
            f.write("FIXED INTERVENTION AT TIME k\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Starting from X₀={self.X0}\n")
            f.write(f"Evaluates E[Y] under fixed policies: 'always intervene at time k'\n\n")
            
            f.write(f"{'─'*80}\n")
            f.write("COMPARISON OF FIXED INTERVENTION POLICIES\n")
            f.write(f"{'─'*80}\n\n")
            
            # Find best fixed policy
            best_k = max(fixed_results, key=fixed_results.get)
            best_val = fixed_results[best_k]
            
            f.write(f"  Never intervene:            E[Y] = {fixed_results['never']:.4f}\n")
            for k in range(1, 6):
                val = fixed_results[f't={k}']
                marker = " ← Best fixed policy" if f't={k}' == best_k else ""
                f.write(f"  Always intervene at t={k}:      E[Y] = {val:.4f}{marker}\n")
            
            f.write(f"\n  Optimal adaptive policy:    E[Y] = {V_optimal:.4f} (from Standard Optimal Stopping)\n\n")
            
            # Analysis
            f.write(f"{'─'*80}\n")
            f.write("ANALYSIS\n")
            f.write(f"{'─'*80}\n\n")
            
            improvement = V_optimal - best_val
            pct_improvement = (improvement / best_val * 100) if best_val > 0 else 0
            
            f.write(f"Best fixed policy: {best_k} with E[Y] = {best_val:.4f}\n")
            f.write(f"Optimal adaptive policy: E[Y] = {V_optimal:.4f}\n\n")
            f.write(f"Value of adaptive decision-making:\n")
            f.write(f"  Absolute improvement: {improvement:.4f}\n")
            f.write(f"  Relative improvement: {pct_improvement:.2f}%\n\n")
            
            if improvement > 0.01:
                f.write(f"✓ Adaptive policy significantly outperforms any fixed-time intervention\n")
                f.write(f"  State-dependent decisions add substantial value\n")
            elif improvement > 0.001:
                f.write(f"✓ Adaptive policy slightly outperforms fixed-time interventions\n")
                f.write(f"  State-dependent decisions add modest value\n")
            else:
                f.write(f"✓ Fixed and adaptive policies perform similarly\n")
                f.write(f"  Simple fixed-time rule may be sufficient\n")
            
            f.write(f"\n{'='*80}\n")


# Run algorithms
if __name__ == "__main__":
    model = CausalOptimalStopping(X0=10)
    
    # Solve optimal policies and generate output file
    output_file = 'RESULTS.txt'
    model.print_results(output_file)