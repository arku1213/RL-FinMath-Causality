import numpy as np

class CausalOptimalStopping:
    """
    Causal Optimal Stopping with 3 Algorithms:
    1. Standard Optimal Stopping
    2. Bounds under Incomplete Information
    3. Robust Optimal Stopping
    """
    
    def __init__(self, X0=10, prob_uncertainty=0.15, intervention_uncertainty=0.25):
        self.X0 = X0
        self.X_min, self.X_max = 1, 20
        self.healthy_low, self.healthy_high = 7, 14
        self.intervention_low, self.intervention_high = 10, 12
        
        # U values (no zero!)
        self.U_values = [-3, -2, -1, 1, 2, 3]
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
        probs = np.array([np.exp(-0.5 * (u ** 2)) for u in self.U_values])
        return probs / probs.sum()
    
    def _compute_probability_bounds(self):
        eps = self.prob_uncertainty
        probs_lower = np.maximum(0, self.U_probs - eps)
        probs_lower = probs_lower / probs_lower.sum()
        probs_upper = np.minimum(1, self.U_probs + eps)
        probs_upper = probs_upper / probs_upper.sum()
        return probs_lower, probs_upper
    
    def transition(self, X, U_current, U_next, intervene=False, intervention_range=None):
        if intervene:
            if intervention_range is None:
                X = np.clip(X, self.intervention_low, self.intervention_high)
            else:
                X = np.clip(X, intervention_range[0], intervention_range[1])
        
        if X < 3 or X > 17:
            return X
        
        X_next = np.floor(X + U_current/3 + U_next/2)
        return int(np.clip(X_next, self.X_min, self.X_max))
    
    def compute_Y(self, X6):
        """
        Probabilistic outcome based on final health marker X6
        
        Returns P(Y=1 | X6) - probability of good outcome
        
        Zones (symmetric around optimal [10,12]):
        - [10, 12]: 1.00 (certain success - optimal range)
        - [8, 10) or (12, 13]: 0.90 (very good)
        - [7, 8) or [13, 14]: 0.70 (good)
        - [5, 7) or (14, 16]: 0.30 (poor)
        - < 5 or > 16: 0.00 (certain failure)
        """
        if 10 <= X6 <= 12:
            return 1.0  # Optimal range - certain success
        elif (8 <= X6 < 10) or (12 < X6 <= 13):
            return 0.9  # Very close to optimal
        elif (7 <= X6 < 8) or (13 < X6 <= 14):
            return 0.7  # Acceptable but not great
        elif (5 <= X6 < 7) or (14 < X6 <= 16):
            return 0.3  # Poor outcome likely
        else:
            return 0.0  # Too extreme - certain failure
    

    
    # ========================================================================
    # STANDARD OPTIMAL STOPPING
    # ========================================================================
    
    def solve_standard_optimal_stopping(self):
        """Standard Optimal Stopping: Find optimal intervention time"""
        
        # Terminal
        for X6 in range(self.X_min, self.X_max + 1):
            Y = self.compute_Y(X6)
            for U6 in self.U_values:
                self.value_function[(6, X6, U6, 0)] = Y
                self.value_function[(6, X6, U6, 1)] = Y
        
        # Backward induction
        for t in range(5, 0, -1):
            for Xt in range(self.X_min, self.X_max + 1):
                for Ut in self.U_values:
                    
                    # Already intervened (I=1)
                    cont_used = self._continuation_value(t, Xt, Ut, True, 'standard')
                    self.value_function[(t, Xt, Ut, 1)] = cont_used
                    self.policy[(t, Xt, Ut, 1)] = 'no_action'
                    
                    # Haven't intervened (I=0) - OPTIMAL STOPPING
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
        Xt_int = int(np.clip(Xt, self.intervention_low, self.intervention_high))
        total = 0.0
        
        for U_next in self.U_values:
            prob = self.U_probs[self.U_values.index(U_next)]
            X_next = self.transition(Xt_int, Ut, U_next, intervene=False)
            
            if mode == 'standard':
                future = self.value_function.get((t+1, X_next, U_next, 1), 0)
            total += prob * future
        
        return total
    
    def _continuation_value(self, t, Xt, Ut, already_intervened, mode='standard'):
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
        
        # Terminal
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
        scenarios = []
        
        # Try different intervention ranges
        int_ranges = [
            (self.intervention_low, self.intervention_high),
            (int(self.intervention_low * (1 - self.intervention_uncertainty)),
             int(self.intervention_high * (1 + self.intervention_uncertainty)))
        ]
        
        for int_range in int_ranges:
            for use_lower_probs in [True, False]:
                probs = self.U_probs_lower if use_lower_probs else self.U_probs_upper
                
                Xt_int = int(np.clip(Xt, int_range[0], int_range[1]))
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
            'outcome': P(Y=1 | X6)
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
                    X_current_intervened = int(np.clip(X_current, self.intervention_low, self.intervention_high))
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
            print(f"  Final X6 = {nat['X_final']}")
            print(f"  P(Y=1) = {nat['outcome']:.2f} ({nat['outcome']*100:.0f}% success)\n")
            
            # All intervention scenarios
            for tau in range(1, 6):
                res = results[tau]
                is_optimal = '← OPTIMAL' if tau == optimal_tau else ''
                print(f"Intervene at t={tau}:")
                print(f"  X path: {res['X_path']}")
                print(f"  Final X6 = {res['X_final']}")
                print(f"  P(Y=1) = {res['outcome']:.2f} ({res['outcome']*100:.0f}% success) {is_optimal}\n")
            
            # Summary
            print(f"{'─'*80}")
            print(f"Optimal intervention time for this path: t={optimal_tau}")
            print(f"Expected outcome at optimal time: P(Y=1) = {results[optimal_tau]['outcome']:.2f}")
            print(f"Improvement over no intervention: {(results[optimal_tau]['outcome'] - results['natural']['outcome']):.2f}")
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
            print(f"  Natural: P(Y=1) = {results['natural']['outcome']:.2f}")
            print(f"  Optimal τ*={optimal_tau}: P(Y=1) = {results[optimal_tau]['outcome']:.2f}")
            print(f"  Improvement: +{improvement:.2f}\n")
        
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
        print(f"Max improvement: {np.max(improvements):.3f}")
        print(f"Min improvement: {np.min(improvements):.3f}")
        print(f"{'='*80}\n")
        
        return optimal_times, improvements
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    def print_results(self, output_file='detailed_results.txt'):
        """
        Print results - saves detailed analysis to file, shows summary to console
        
        NOTE: E[Y] now represents expected probability of good outcome (probabilistic Y)
        - Y ∈ [0, 1] where Y = P(success | X6)
        - Y = 1.0 if X6 ∈ [10,12] (optimal)
        - Y decreases as X6 moves away from optimal range
        
        Parameters:
        -----------
        output_file : str
            Path to save detailed results
        """
        
        # Solve all algorithms
        V2 = self.solve_standard_optimal_stopping()
        V3_lower, V3_upper = self.solve_bounds_incomplete_information()
        self.solve_robust_optimal_stopping()
        
        # Open file for detailed output
        with open(output_file, 'w') as f:
            # Redirect all detailed output to file
            f.write(f"{'='*80}\n")
            f.write("STANDARD OPTIMAL STOPPING - DETAILED STATE ANALYSIS\n")
            f.write(f"{'='*80}\n")
            f.write(f"E[Y | X_0={self.X0}, optimal intervention]: {V2:.4f}  ({V2*100:.1f}% success)\n\n")
            
            # Detailed state-by-state analysis
            for t in range(1, self.T + 1):
                f.write(f"\n{'─'*80}\n")
                f.write(f"TIME t={t}\n")
                f.write(f"{'─'*80}\n")
                
                # Show ALL X values
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"\nX={X}:\n")
                    
                    for U in self.U_values:
                        policy = self.policy.get((t, X, U, 0), '?')
                        value_wait = self.value_function.get((t, X, U, 0), 0)
                        
                        if policy == 'INTERVENE':
                            # Show what happens if we intervene
                            X_intervened = int(np.clip(X, self.intervention_low, self.intervention_high))
                            value_intervene = self._intervention_value(t, X, U, 'standard')
                            
                            # Show all possible next states
                            next_states = []
                            for U_next in self.U_values:
                                X_next = self.transition(X_intervened, U, U_next, intervene=False)
                                V_next = self.value_function.get((t+1, X_next, U_next, 1), 0)
                                next_states.append((X_next, U_next, V_next))
                            
                            f.write(f"  U={U:2d}: 🔴 INTERVENE\n")
                            f.write(f"         State (X={X}, U={U}, I=0) → E[Y]={value_wait:.4f}\n")
                            f.write(f"         Intervene: X→{X_intervened} → E[Y]={value_intervene:.4f}\n")
                            f.write(f"         Possible next states after intervention:\n")
                            
                            for X_next, U_next, V_next in next_states:
                                prob = self.U_probs[self.U_values.index(U_next)]
                                f.write(f"           → (X={X_next}, U={U_next}, I=1): E[Y]={V_next:.4f} [p={prob:.3f}]\n")
                        
                        elif policy == 'WAIT':
                            f.write(f"  U={U:2d}: ⚪ WAIT → E[Y]={value_wait:.4f}\n")
            
            # Bounds analysis
            f.write(f"\n{'='*80}\n")
            f.write("BOUNDS UNDER INCOMPLETE INFORMATION\n")
            f.write(f"{'='*80}\n")
            f.write(f"E[Y | X_0={self.X0}] ∈ [{V3_lower:.4f}, {V3_upper:.4f}]\n")
            f.write(f"Uncertainty width: {V3_upper - V3_lower:.4f}\n\n")
            
            # Show bounds for all states
            f.write("VALUE BOUNDS FOR ALL STATES:\n")
            f.write("-" * 80 + "\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\nTIME t={t}:\n")
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"  X={X}:\n")
                    for U in self.U_values:
                        V_lower = self.value_lower.get((t, X, U, 0), 0)
                        V_upper = self.value_upper.get((t, X, U, 0), 0)
                        width = V_upper - V_lower
                        f.write(f"    U={U:2d}: E[Y] ∈ [{V_lower:.4f}, {V_upper:.4f}]  (width={width:.4f})\n")
            
            # Robust policy
            f.write(f"\n{'='*80}\n")
            f.write("ROBUST OPTIMAL STOPPING - DETAILED\n")
            f.write(f"{'='*80}\n\n")
            
            for t in range(1, self.T + 1):
                f.write(f"\nTIME t={t}:\n")
                for X in range(self.X_min, self.X_max + 1):
                    f.write(f"  X={X}:\n")
                    for U in self.U_values:
                        policy = self.robust_policy.get((t, X, U, 0), '?')
                        if policy == 'INTERVENE':
                            symbol = '🔴'
                        elif policy == 'WAIT':
                            symbol = '⚪'
                        else:
                            symbol = '🟡'
                        f.write(f"    U={U:2d}: {symbol} {policy}\n")
        
        # Print summary to console
        print(f"\n{'='*80}")
        print("CAUSAL OPTIMAL STOPPING - RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        print("STANDARD OPTIMAL STOPPING")
        print("-" * 80)
        print(f"E[Y | X_0={self.X0}]: {V2:.4f}  ({V2*100:.1f}% success)\n")
        
        # Intervention summary by time
        print("Intervention Pattern (% of states where INTERVENE is optimal):")
        for t in range(1, self.T + 1):
            intervene_count = sum(1 for X in range(self.X_min, self.X_max + 1)
                                for U in self.U_values
                                if self.policy.get((t, X, U, 0)) == 'INTERVENE')
            total = 20 * 6  # 120 states per time
            pct = intervene_count / total * 100
            print(f"  t={t}: {intervene_count:3d}/{total} ({pct:5.1f}%)")
        
        print(f"\n{'='*80}")
        print("BOUNDS UNDER INCOMPLETE INFORMATION")
        print("-" * 80)
        print(f"E[Y | X_0={self.X0}] ∈ [{V3_lower:.4f}, {V3_upper:.4f}]")
        print(f"Uncertainty width: {V3_upper - V3_lower:.4f}\n")
        
        print(f"{'='*80}")
        print("ROBUST OPTIMAL STOPPING")
        print("-" * 80)
        
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
                print(f"t={t}: INTERVENE {intervene_count:3d} ({intervene_count/total*100:5.1f}%), " + 
                      f"WAIT {wait_count:3d} ({wait_count/total*100:5.1f}%), " +
                      f"AMBIGUOUS {ambiguous_count:3d} ({ambiguous_count/total*100:5.1f}%)")
                
                total_stats['INTERVENE'] += intervene_count
                total_stats['WAIT'] += wait_count
                total_stats['AMBIGUOUS'] += ambiguous_count
                total_stats['total'] += total
        
        print(f"\nOVERALL:")
        t = total_stats['total']
        print(f"  INTERVENE:  {total_stats['INTERVENE']:3d}/{t} ({total_stats['INTERVENE']/t*100:5.1f}%)")
        print(f"  WAIT:       {total_stats['WAIT']:3d}/{t} ({total_stats['WAIT']/t*100:5.1f}%)")
        print(f"  AMBIGUOUS:  {total_stats['AMBIGUOUS']:3d}/{t} ({total_stats['AMBIGUOUS']/t*100:5.1f}%)")
        
        print(f"\n{'='*80}")
        print(f"✓ Detailed results saved to: {output_file}")
        print(f"{'='*80}\n")


# Run all algorithms and counterfactual analysis
if __name__ == "__main__":
    model = CausalOptimalStopping(X0=10, prob_uncertainty=0.15, intervention_uncertainty=0.25)
    
    # Solve optimal policies
    model.print_results()
    
    print("\n" + "="*80)
    print("COUNTERFACTUAL ANALYSIS EXAMPLES")
    print("="*80)
    print("\nDemonstrating Pearl's Rung 3: 'What if we had intervened differently?'\n")
    
    # Example 1: Specific shock sequence
    print("Example 1: Analyzing a specific shock sequence")
    U_example = [-2, 1, -1, 2, -3, 1]
    results, optimal_tau = model.analyze_counterfactuals(U_example)
    
    # Example 2: Another shock sequence
    print("\nExample 2: Another shock sequence")
    U_example2 = [3, -2, 2, -1, 1, -3]
    results2, optimal_tau2 = model.analyze_counterfactuals(U_example2)
    
    # Example 3: Multiple random paths
    print("\n" + "="*80)
    print("ANALYZING MULTIPLE RANDOM PATHS")
    print("="*80)
    optimal_times, improvements = model.compare_multiple_counterfactuals(num_paths=15)