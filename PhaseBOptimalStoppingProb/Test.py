import numpy as np

class CausalOptimalStopping:
    """
    Causal Optimal Stopping with 3 Algorithms:
    1. Standard Optimal Stopping
    2. Bounds under Incomplete Information
    3. Robust Optimal Stopping
    
    UPDATES:
    - U now includes 0: U ∈ {-3, -2, -1, 0, 1, 2, 3} (7 states)
    - Asymmetric U distribution (negative bias)
    - Intervention pulls toward center (no specific "healthy range")
    - Binary outcome Y: Y=0 for extreme X₆, Y=1 otherwise
    - Output shows ALL states (280 per time period)
    """
    
    def __init__(self, X0=10, prob_uncertainty=0.15, intervention_uncertainty=0.25):
        self.X0 = X0
        self.X_min, self.X_max = 1, 20
        
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
        
        # Boundary behavior
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
        """Standard Optimal Stopping: Find optimal intervention time"""
        
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
        """Bounds under Incomplete Information: Compute upper and lower bounds on E[Y]"""
        
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
        """Robust Optimal Stopping: Robust policy under uncertainty"""
        
        # First solve Bounds if not already done
        if not self.value_lower:
            self.solve_bounds_incomplete_information()
        
        # Derive robust policy
        for t in range(1, self.T + 1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    
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
    # COUNTERFACTUAL ANALYSIS (Pearl's Rung 3)
    # ========================================================================
    
    def simulate_path_with_intervention(self, U_sequence, intervene_at=None):
        """
        Simulate a path given a specific shock sequence
        
        Parameters:
        -----------
        U_sequence : list
            [U1, U2, U3, U4, U5, U6] - the realized shocks
        intervene_at : int or None
            Time to intervene (1-5), or None for no intervention
        
        Returns:
        --------
        result : dict
            'X_path': List of X values [X0, X1, ..., X6]
            'intervened': Whether intervention was applied
            'intervention_time': When intervention occurred
            'X_final': Final X6 value
            'outcome': Y ∈ {0, 1}
        """
        X_path = [self.X0]
        X_current = self.X0
        intervened = False
        
        for t in range(1, self.T + 1):
            U_t = U_sequence[t-1]
            
            # For t=1: X1 = floor(X0 + U1/2)
            # For t>1: X_t = floor(X_{t-1} + U_{t-1}/3 + U_t/2)
            if t == 1:
                # First transition: only U1 affects
                if intervene_at == t and not intervened:
                    X_current_intervened = self.apply_intervention(X_current)
                    X_next = int(np.floor(X_current_intervened + U_t/2))
                    X_next = int(np.clip(X_next, self.X_min, self.X_max))
                    intervened = True
                else:
                    X_next = int(np.floor(X_current + U_t/2))
                    X_next = int(np.clip(X_next, self.X_min, self.X_max))
            else:
                # Later transitions: both U_{t-1} and U_t affect
                U_prev = U_sequence[t-2]
                
                # Check if we should intervene at this time
                if intervene_at == t and not intervened:
                    X_next = self.transition(X_current, U_prev, U_t, intervene=True)
                    intervened = True
                else:
                    X_next = self.transition(X_current, U_prev, U_t, intervene=False)
            
            X_path.append(X_next)
            X_current = X_next
        
        X_final = X_path[-1]
        outcome = self.compute_Y(X_final)
        
        return {
            'X_path': X_path,
            'intervened': intervened,
            'intervention_time': intervene_at if intervened else None,
            'X_final': X_final,
            'outcome': outcome
        }
    
    def analyze_counterfactuals(self, U_sequence, verbose=True):
        """
        Counterfactual analysis: "What if we had intervened at different times?"
        
        Given a specific shock sequence, compute the outcome for:
        - No intervention (natural path)
        - Intervention at t=1, 2, 3, 4, 5
        
        Find the optimal intervention time for THIS specific path.
        
        Parameters:
        -----------
        U_sequence : list
            [U1, U2, U3, U4, U5, U6] - the realized shocks
        verbose : bool
            Print detailed results
        
        Returns:
        --------
        results : dict
            Results for each intervention time
        optimal_tau : int or None
            Best intervention time for this sequence
        """
        if len(U_sequence) != 6:
            raise ValueError("U_sequence must have exactly 6 elements [U1, ..., U6]")
        
        results = {}
        
        # Natural path (no intervention)
        results['natural'] = self.simulate_path_with_intervention(U_sequence, intervene_at=None)
        
        # Try intervening at each time
        for tau in range(1, 6):  # t = 1, 2, 3, 4, 5
            results[tau] = self.simulate_path_with_intervention(U_sequence, intervene_at=tau)
        
        # Find optimal intervention time
        intervention_outcomes = {tau: results[tau]['outcome'] for tau in range(1, 6)}
        optimal_tau = max(intervention_outcomes, key=intervention_outcomes.get)
        
        if verbose:
            print(f"\n{'='*80}")
            print("COUNTERFACTUAL ANALYSIS")
            print(f"{'='*80}")
            print(f"Shock sequence: U = {U_sequence}\n")
            
            # Natural path
            nat = results['natural']
            print(f"Natural (no intervention):")
            print(f"  X path: {nat['X_path']}")
            print(f"  Final X₆ = {nat['X_final']}")
            print(f"  Y = {nat['outcome']} ({'SUCCESS' if nat['outcome']==1 else 'FAILURE'})\n")
            
            # All intervention scenarios
            for tau in range(1, 6):
                res = results[tau]
                is_optimal = '← OPTIMAL' if tau == optimal_tau else ''
                print(f"Intervene at t={tau}:")
                print(f"  X path: {res['X_path']}")
                print(f"  Final X₆ = {res['X_final']}")
                print(f"  Y = {res['outcome']} ({'SUCCESS' if res['outcome']==1 else 'FAILURE'}) {is_optimal}\n")
            
            # Summary
            print(f"{'─'*80}")
            print(f"Optimal intervention time for this path: t={optimal_tau}")
            print(f"Expected outcome at optimal time: Y = {results[optimal_tau]['outcome']}")
            print(f"Improvement over no intervention: {results[optimal_tau]['outcome'] - results['natural']['outcome']}")
            print(f"{'='*80}\n")
        
        return results, optimal_tau
    
    def compare_multiple_counterfactuals(self, num_paths=10):
        """
        Analyze counterfactuals for multiple random shock sequences
        
        Shows distribution of optimal intervention times across different scenarios
        """
        print(f"\n{'='*80}")
        print(f"COUNTERFACTUAL ANALYSIS: {num_paths} RANDOM SHOCK SEQUENCES")
        print(f"{'='*80}\n")
        
        optimal_times = []
        improvements = []
        
        for path_num in range(num_paths):
            # Generate random shock sequence
            np.random.seed(path_num)
            U_sequence = np.random.choice(self.U_values, size=6, p=self.U_probs).tolist()
            
            # Analyze counterfactuals
            results, optimal_tau = self.analyze_counterfactuals(U_sequence, verbose=False)
            
            optimal_times.append(optimal_tau)
            improvement = results[optimal_tau]['outcome'] - results['natural']['outcome']
            improvements.append(improvement)
            
            # Print summary for this path
            print(f"Path #{path_num + 1}: U = {U_sequence}")
            print(f"  Natural: Y = {results['natural']['outcome']}")
            print(f"  Optimal τ*={optimal_tau}: Y = {results[optimal_tau]['outcome']}")
            print(f"  Improvement: {improvement:+d}\n")
        
        # Summary statistics
        print(f"{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        from collections import Counter
        time_distribution = Counter(optimal_times)
        
        print(f"Distribution of optimal intervention times:")
        for t in sorted(time_distribution.keys()):
            count = time_distribution[t]
            pct = count / num_paths * 100
            print(f"  t={t}: {count}/{num_paths} paths ({pct:.1f}%)")
        
        print(f"\nAverage improvement from optimal intervention: {np.mean(improvements):.3f}")
        print(f"Max improvement: {np.max(improvements):.0f}")
        print(f"Min improvement: {np.min(improvements):.0f}")
        print(f"{'='*80}\n")
        
        return optimal_times, improvements
    
    # ========================================================================
    # OUTPUT - NOW SHOWS ALL STATES
    # ========================================================================
    
    def print_results(self, output_file='detailed_results.txt'):
        """
        Print results - saves ALL analysis to file
        
        NOW SHOWS ALL 280 STATES PER TIME PERIOD:
        - 20 X values × 7 U values × 2 I values = 280 states
        - Includes "unreachable" states (important for backward induction analysis)
        
        Parameters:
        -----------
        output_file : str
            Path to save detailed results
        """
        
        # Solve all algorithms
        V2 = self.solve_standard_optimal_stopping()
        V3_lower, V3_upper = self.solve_bounds_incomplete_information()
        self.solve_robust_optimal_stopping()
        
        # Open file for ALL output
        with open(output_file, 'w') as f:
            # Header
            f.write(f"{'='*80}\n")
            f.write("CAUSAL OPTIMAL STOPPING - COMPLETE STATE ANALYSIS\n")
            f.write(f"{'='*80}\n")
            f.write(f"State space: {self.X_max - self.X_min + 1} X values × {len(self.U_values)} U values × 2 I values\n")
            f.write(f"            = {(self.X_max - self.X_min + 1) * len(self.U_values) * 2} states per time period\n")
            f.write(f"Initial condition: X₀ = {self.X0}\n")
            f.write(f"U distribution (asymmetric, negative bias):\n")
            for u, prob in zip(self.U_values, self.U_probs):
                f.write(f"  U={u:2d}: {prob:.3f} ({prob*100:.1f}%)\n")
            f.write(f"\n")
            
            # ================================================================
            # STANDARD OPTIMAL STOPPING
            # ================================================================
            f.write(f"{'='*80}\n")
            f.write("STANDARD OPTIMAL STOPPING - ALL STATES\n")
            f.write(f"{'='*80}\n")
            f.write(f"E[Y | X₀={self.X0}, optimal intervention]: {V2:.4f}\n\n")
            
            # Show ALL states for each time period
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*80}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*80}\n")
                
                # Loop through ALL X values (no filtering)
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX_{t}={X:2d}:\n")
                    
                    # Loop through ALL U values
                    for U in self.U_values:
                        # State (X, U, I=0) - haven't intervened yet
                        policy = self.policy.get((t, X, U, 0), '?')
                        value = self.value_function.get((t, X, U, 0), 0)
                        
                        if policy == 'INTERVENE':
                            X_intervened = self.apply_intervention(X)
                            f.write(f"  U_{t}={U:2d}, I=0: 🔴 INTERVENE → X becomes {X_intervened}, E[Y]={value:.4f}\n")
                        elif policy == 'WAIT':
                            f.write(f"  U_{t}={U:2d}, I=0: ⚪ WAIT, E[Y]={value:.4f}\n")
                        else:
                            f.write(f"  U_{t}={U:2d}, I=0: ?, E[Y]={value:.4f}\n")
                        
                        # State (X, U, I=1) - already intervened
                        value_used = self.value_function.get((t, X, U, 1), 0)
                        f.write(f"  U_{t}={U:2d}, I=1: no_action, E[Y]={value_used:.4f}\n")
            
            # Terminal states
            f.write(f"\n{'─'*80}\n")
            f.write(f"TIME t=6 (TERMINAL)\n")
            f.write(f"{'─'*80}\n")
            for X in range(self.X_min, self.X_max + 1):
                Y = self.compute_Y(X)
                outcome_str = "SUCCESS" if Y == 1 else "FAILURE"
                f.write(f"X₆={X:2d}: Y={Y} ({outcome_str})\n")
            
            # ================================================================
            # BOUNDS UNDER INCOMPLETE INFORMATION
            # ================================================================
            f.write(f"\n{'='*80}\n")
            f.write("BOUNDS UNDER INCOMPLETE INFORMATION - ALL STATES\n")
            f.write(f"{'='*80}\n")
            f.write(f"E[Y | X₀={self.X0}] ∈ [{V3_lower:.4f}, {V3_upper:.4f}]\n")
            f.write(f"Uncertainty width: {V3_upper - V3_lower:.4f}\n\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*80}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*80}\n")
                
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
            f.write(f"\n{'='*80}\n")
            f.write("ROBUST OPTIMAL STOPPING - ALL STATES\n")
            f.write(f"{'='*80}\n\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*80}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*80}\n")
                
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX_{t}={X:2d}:\n")
                    
                    for U in self.U_values:
                        policy = self.robust_policy.get((t, X, U, 0), '?')
                        if policy == 'INTERVENE':
                            symbol = '🔴'
                        elif policy == 'WAIT':
                            symbol = '⚪'
                        elif policy == 'AMBIGUOUS':
                            symbol = '🟡'
                        else:
                            symbol = '?'
                        f.write(f"  U_{t}={U:2d}, I=0: {symbol} {policy}\n")
                        f.write(f"  U_{t}={U:2d}, I=1: no_action\n")
            
            # ================================================================
            # SUMMARY STATISTICS
            # ================================================================
            f.write(f"\n{'='*80}\n")
            f.write("CAUSAL OPTIMAL STOPPING - SUMMARY STATISTICS\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("STATE SPACE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"X values: [{self.X_min}, {self.X_max}] ({self.X_max - self.X_min + 1} values)\n")
            f.write(f"U values: {self.U_values} ({len(self.U_values)} values)\n")
            f.write(f"I values: {{0, 1}} (2 values)\n")
            f.write(f"Total states per time: {(self.X_max - self.X_min + 1) * len(self.U_values) * 2}\n")
            f.write(f"\nU distribution (negative bias):\n")
            for u, prob in zip(self.U_values, self.U_probs):
                f.write(f"  U={u:2d}: {prob:.3f}\n")
            expected_U = sum(u * p for u, p in zip(self.U_values, self.U_probs))
            f.write(f"Expected U: {expected_U:.3f}\n\n")
            
            f.write("STANDARD OPTIMAL STOPPING\n")
            f.write("-" * 80 + "\n")
            f.write(f"E[Y | X₀={self.X0}]: {V2:.4f}\n\n")
            
            # Intervention summary by time
            f.write("Intervention Pattern (% of I=0 states where INTERVENE is optimal):\n")
            for t in range(1, self.T + 1):
                intervene_count = sum(1 for X in range(self.X_min, self.X_max + 1)
                                    for U in self.U_values
                                    if self.policy.get((t, X, U, 0)) == 'INTERVENE')
                total = (self.X_max - self.X_min + 1) * len(self.U_values)
                pct = intervene_count / total * 100
                f.write(f"  t={t}: {intervene_count:3d}/{total} ({pct:5.1f}%)\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("BOUNDS UNDER INCOMPLETE INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"E[Y | X₀={self.X0}] ∈ [{V3_lower:.4f}, {V3_upper:.4f}]\n")
            f.write(f"Uncertainty width: {V3_upper - V3_lower:.4f}\n\n")
            
            f.write(f"{'='*80}\n")
            f.write("ROBUST OPTIMAL STOPPING\n")
            f.write("-" * 80 + "\n")
            
            # Count robust decisions across all times
            total_stats = {'INTERVENE': 0, 'WAIT': 0, 'AMBIGUOUS': 0, 'total': 0}
            
            for t in range(1, self.T + 1):
                intervene_count = 0
                wait_count = 0
                ambiguous_count = 0
                total = 0
                
                for X in range(self.X_min, self.X_max + 1):
                    for U in self.U_values:
                        policy = self.robust_policy.get((t, X, U, 0), None)
                        if policy:
                            total += 1
                            if policy == 'INTERVENE':
                                intervene_count += 1
                            elif policy == 'WAIT':
                                wait_count += 1
                            elif policy == 'AMBIGUOUS':
                                ambiguous_count += 1
                
                if total > 0:
                    f.write(f"t={t}: INTERVENE {intervene_count:3d} ({intervene_count/total*100:5.1f}%), " + 
                          f"WAIT {wait_count:3d} ({wait_count/total*100:5.1f}%), " +
                          f"AMBIGUOUS {ambiguous_count:3d} ({ambiguous_count/total*100:5.1f}%)\n")
                    
                    total_stats['INTERVENE'] += intervene_count
                    total_stats['WAIT'] += wait_count
                    total_stats['AMBIGUOUS'] += ambiguous_count
                    total_stats['total'] += total
            
            f.write(f"\nOVERALL:\n")
            t = total_stats['total']
            f.write(f"  INTERVENE:  {total_stats['INTERVENE']:3d}/{t} ({total_stats['INTERVENE']/t*100:5.1f}%)\n")
            f.write(f"  WAIT:       {total_stats['WAIT']:3d}/{t} ({total_stats['WAIT']/t*100:5.1f}%)\n")
            f.write(f"  AMBIGUOUS:  {total_stats['AMBIGUOUS']:3d}/{t} ({total_stats['AMBIGUOUS']/t*100:5.1f}%)\n")
            
            f.write(f"\n{'='*80}\n")
        
        # Print simple confirmation to console
        print(f"\n{'='*80}")
        print(f"✓ All results saved to: {output_file}")
        print(f"{'='*80}\n")


# Run all algorithms and counterfactual analysis
if __name__ == "__main__":
    model = CausalOptimalStopping(X0=10, prob_uncertainty=0.15, intervention_uncertainty=0.25)
    
    # Open output file for ALL results
    output_file = 'detailed_results.txt'
    
    # Solve optimal policies (this writes to file)
    model.print_results(output_file)
    
    # Now append counterfactual analysis to the same file
    with open(output_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("COUNTERFACTUAL ANALYSIS EXAMPLES\n")
        f.write("="*80 + "\n")
        f.write("\nDemonstrating Pearl's Rung 3: 'What if we had intervened differently?'\n\n")
        
        # Example 1: Specific shock sequence
        f.write("Example 1: Analyzing a specific shock sequence\n")
        U_example = [-2, 1, -1, 0, -3, 1]
        
        # Capture counterfactual results
        results, optimal_tau = model.analyze_counterfactuals(U_example, verbose=False)
        
        f.write(f"\n{'='*80}\n")
        f.write("COUNTERFACTUAL ANALYSIS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Shock sequence: U = {U_example}\n\n")
        
        # Natural path
        nat = results['natural']
        f.write(f"Natural (no intervention):\n")
        f.write(f"  X path: {nat['X_path']}\n")
        f.write(f"  Final X₆ = {nat['X_final']}\n")
        f.write(f"  Y = {nat['outcome']} ({'SUCCESS' if nat['outcome']==1 else 'FAILURE'})\n\n")
        
        # All intervention scenarios
        for tau in range(1, 6):
            res = results[tau]
            is_optimal = '← OPTIMAL' if tau == optimal_tau else ''
            f.write(f"Intervene at t={tau}:\n")
            f.write(f"  X path: {res['X_path']}\n")
            f.write(f"  Final X₆ = {res['X_final']}\n")
            f.write(f"  Y = {res['outcome']} ({'SUCCESS' if res['outcome']==1 else 'FAILURE'}) {is_optimal}\n\n")
        
        # Summary
        f.write(f"{'─'*80}\n")
        f.write(f"Optimal intervention time for this path: t={optimal_tau}\n")
        f.write(f"Expected outcome at optimal time: Y = {results[optimal_tau]['outcome']}\n")
        f.write(f"Improvement over no intervention: {results[optimal_tau]['outcome'] - results['natural']['outcome']}\n")
        f.write(f"{'='*80}\n\n")
        
        # Example 2: Another shock sequence
        f.write("\nExample 2: Another shock sequence\n")
        U_example2 = [3, -2, 2, -1, 1, -3]
        results2, optimal_tau2 = model.analyze_counterfactuals(U_example2, verbose=False)
        
        f.write(f"\n{'='*80}\n")
        f.write("COUNTERFACTUAL ANALYSIS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Shock sequence: U = {U_example2}\n\n")
        
        nat2 = results2['natural']
        f.write(f"Natural (no intervention):\n")
        f.write(f"  X path: {nat2['X_path']}\n")
        f.write(f"  Final X₆ = {nat2['X_final']}\n")
        f.write(f"  Y = {nat2['outcome']} ({'SUCCESS' if nat2['outcome']==1 else 'FAILURE'})\n\n")
        
        for tau in range(1, 6):
            res2 = results2[tau]
            is_optimal2 = '← OPTIMAL' if tau == optimal_tau2 else ''
            f.write(f"Intervene at t={tau}:\n")
            f.write(f"  X path: {res2['X_path']}\n")
            f.write(f"  Final X₆ = {res2['X_final']}\n")
            f.write(f"  Y = {res2['outcome']} ({'SUCCESS' if res2['outcome']==1 else 'FAILURE'}) {is_optimal2}\n\n")
        
        f.write(f"{'─'*80}\n")
        f.write(f"Optimal intervention time for this path: t={optimal_tau2}\n")
        f.write(f"Expected outcome at optimal time: Y = {results2[optimal_tau2]['outcome']}\n")
        f.write(f"Improvement over no intervention: {results2[optimal_tau2]['outcome'] - results2['natural']['outcome']}\n")
        f.write(f"{'='*80}\n\n")
        
        # Example 3: Multiple random paths
        f.write("\n" + "="*80 + "\n")
        f.write("ANALYZING MULTIPLE RANDOM PATHS\n")
        f.write("="*80 + "\n\n")
        
        num_paths = 15
        f.write(f"{'='*80}\n")
        f.write(f"COUNTERFACTUAL ANALYSIS: {num_paths} RANDOM SHOCK SEQUENCES\n")
        f.write(f"{'='*80}\n\n")
        
        optimal_times = []
        improvements = []
        
        for path_num in range(num_paths):
            # Generate random shock sequence
            np.random.seed(path_num)
            U_sequence = np.random.choice(model.U_values, size=6, p=model.U_probs).tolist()
            
            # Analyze counterfactuals
            results_i, optimal_tau_i = model.analyze_counterfactuals(U_sequence, verbose=False)
            
            optimal_times.append(optimal_tau_i)
            improvement = results_i[optimal_tau_i]['outcome'] - results_i['natural']['outcome']
            improvements.append(improvement)
            
            # Write summary for this path
            f.write(f"Path #{path_num + 1}: U = {U_sequence}\n")
            f.write(f"  Natural: Y = {results_i['natural']['outcome']}\n")
            f.write(f"  Optimal τ*={optimal_tau_i}: Y = {results_i[optimal_tau_i]['outcome']}\n")
            f.write(f"  Improvement: {improvement:+d}\n\n")
        
        # Summary statistics
        f.write(f"{'='*80}\n")
        f.write("SUMMARY STATISTICS\n")
        f.write(f"{'='*80}\n")
        
        from collections import Counter
        time_distribution = Counter(optimal_times)
        
        f.write(f"Distribution of optimal intervention times:\n")
        for t in sorted(time_distribution.keys()):
            count = time_distribution[t]
            pct = count / num_paths * 100
            f.write(f"  t={t}: {count}/{num_paths} paths ({pct:.1f}%)\n")
        
        f.write(f"\nAverage improvement from optimal intervention: {np.mean(improvements):.3f}\n")
        f.write(f"Max improvement: {np.max(improvements):.0f}\n")
        f.write(f"Min improvement: {np.min(improvements):.0f}\n")
        f.write(f"{'='*80}\n\n")
    
    print(f"✓ All results and counterfactual analysis saved to: {output_file}")
    print(f"✓ Total file contains: Optimal policies + Counterfactual examples + Summary statistics")