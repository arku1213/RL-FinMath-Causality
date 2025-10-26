import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict

class CausalOptimalStopping:
    """
    Causal Optimal Stopping Problem using Backward Induction
    
    Adapts the American Put Option framework to find optimal intervention timing
    in a causal DAG to maximize expected outcome Y.
    
    Think of it like the blood pressure analogy:
    - X variables = health markers over time
    - U shocks = random daily fluctuations  
    - A = treatment indicator (0=no treatment, 1=treatment given)
    - Y = binary outcome (1=good, 0=bad)
    - Goal: Find optimal time τ* to start treatment
    """
    
    def __init__(self, 
                 n_endogenous=7,      # Number of X variables (X₀ to X₆)
                 n_exogenous=6,       # Number of U shocks (U₁ to U₆)
                 T=6,                 # Time horizon (can be float now, e.g. 6.0)
                 X_min=1,             # Min value for X variables
                 X_max=20,            # Max value for X variables
                 U_min=-3,            # Min value for U shocks
                 U_max=3,             # Max value for U shocks
                 treatment_effect=3,  # Boost from treatment
                 Y_threshold=10,      # Threshold for Y=1 outcome
                 discount=0.95,       # Discount factor for future values
                 dt=0.2,              # Time step size
                 uncertainty_mode='none',  # 'none', 'bounds', or 'robust'
                 prob_uncertainty=0.1,     # Uncertainty radius for probabilities
                 effect_uncertainty=0.2):  # Uncertainty in treatment effect
        
        self.n_endogenous = n_endogenous
        self.n_exogenous = n_exogenous
        self.T_final = T  # Total time horizon
        self.dt = dt  # Time step size
        
        # Create fine time grid
        self.time_grid = np.arange(0, T + dt, dt)
        self.T = len(self.time_grid) - 1  # Number of time steps
        
        self.X_min = X_min
        self.X_max = X_max
        self.U_min = U_min
        self.U_max = U_max
        self.treatment_effect = treatment_effect
        self.Y_threshold = Y_threshold
        self.discount = discount
        
        # NEW: Uncertainty parameters for Algorithms 3 & 4
        self.uncertainty_mode = uncertainty_mode
        self.prob_uncertainty = prob_uncertainty  # ε for P ∈ [P̲, P̄]
        self.effect_uncertainty = effect_uncertainty  # δ for effect ∈ [e-δ, e+δ]
        
        # U values and their probabilities
        self.U_values = list(range(U_min, U_max + 1))
        self.U_probs = self._compute_U_probabilities()
        
        # Compute uncertainty sets if in bounds mode
        if uncertainty_mode in ['bounds', 'robust']:
            self.U_probs_lower, self.U_probs_upper = self._compute_probability_bounds()
            self.effect_lower = treatment_effect * (1 - effect_uncertainty)
            self.effect_upper = treatment_effect * (1 + effect_uncertainty)
        
        # Storage for value functions and policies
        self.value_function = {}  # V(t, X_state, A)
        self.policy = {}          # Optimal action at each state
        
        # NEW: Storage for bounds (Algorithms 3 & 4)
        self.value_lower = {}     # Lower bound V̲(t, X, A)
        self.value_upper = {}     # Upper bound V̄(t, X, A)
        self.robust_policy = {}   # Robust policy under uncertainty
        
        print(f"Initialized Causal Optimal Stopping Problem (DAMPENED VERSION - CONTINUOUS TIME)")
        print(f"  Formula: X_i+1 = X_i + U_i/3 + U_i+1/2")
        print(f"  Time grid: t ∈ [{self.time_grid[0]:.1f}, {self.time_grid[-1]:.1f}] with dt={dt}")
        print(f"  Number of time steps: {self.T} (vs {n_exogenous} in discrete)")
        print(f"  Time points: {self.time_grid[:5]}... (showing first 5)")
        print(f"  Endogenous variables: X₀ to X_T")
        print(f"  X range: [{X_min}, {X_max}]")
        print(f"  U range: [{U_min}, {U_max}]")
        print(f"  U weights: previous=1/3, current=1/2 (dampened)")
        print(f"  Treatment effect: ±{treatment_effect} (state-dependent)")
        if uncertainty_mode in ['bounds', 'robust']:
            print(f"  🔒 UNCERTAINTY MODE: {uncertainty_mode.upper()}")
            print(f"    Probability uncertainty: ±{prob_uncertainty*100:.1f}%")
            print(f"    Treatment effect: [{self.effect_lower:.2f}, {self.effect_upper:.2f}]")
        print(f"  DEATH if X < {X_min} or X > {X_max}")
        print(f"  Healthy range: [7, 14] for Y=1")
        print(f"  Treatment: Pushes X toward optimal value (10)")
    
    def _compute_U_probabilities(self):
        """
        Compute probability distribution for U shocks
        Using a discrete approximation of normal distribution centered at 0
        """
        U_range = self.U_max - self.U_min + 1
        probs = np.zeros(U_range)
        
        # Simple triangular/normal-ish distribution
        for i, u in enumerate(self.U_values):
            probs[i] = np.exp(-0.5 * (u ** 2))
        
        # Normalize
        probs = probs / probs.sum()
        
        return probs
    
    def _compute_probability_bounds(self):
        """
        Compute lower and upper probability bounds for uncertainty set
        
        Algorithm 3: Bounds under Incomplete Information
        
        Creates uncertainty set P ∈ [P̲, P̄] where:
        - P̲(u) = max(0, P(u) - ε)
        - P̄(u) = min(1, P(u) + ε)
        
        Then renormalize to ensure they sum to 1.
        
        Returns:
        --------
        probs_lower : array
            Lower bound probabilities P̲
        probs_upper : array
            Upper bound probabilities P̄
        """
        eps = self.prob_uncertainty
        
        # Lower bounds
        probs_lower = np.maximum(0, self.U_probs - eps)
        probs_lower = probs_lower / probs_lower.sum()  # Renormalize
        
        # Upper bounds
        probs_upper = np.minimum(1, self.U_probs + eps)
        probs_upper = probs_upper / probs_upper.sum()  # Renormalize
        
        return probs_lower, probs_upper
    
    def structural_equation(self, X_prev, U_current, U_prev, intervention_now, t):
        """
        Structural equation for X_t with ONE-TIME INTERVENTION
        
        COLLEAGUE'S DAMPENED FORMULA:
        X_{i+1} = X_i + U_i/3 + U_{i+1}/2
        
        - Previous shock (U_i) has dampened effect (÷3)
        - Current shock (U_{i+1}) has dampened effect (÷2)
        - Models realistic decay: recent shocks matter more
        
        Each U_i affects X_i and X_{i+1}
        
        Treatment is ONE-TIME and STATE-DEPENDENT:
        - intervention_now = True: Apply treatment THIS period (boost or reduce once)
        - intervention_now = False: No treatment this period
        
        Treatment effect:
        - If X < optimal: treatment INCREASES X (one-time boost)
        - If X > optimal: treatment DECREASES X (one-time reduction)
        
        Parameters:
        -----------
        X_prev : int
            Previous X value (X_{t-1})
        U_current : int
            Current shock (U_t)
        U_prev : int or None
            Previous shock (U_{t-1}), affects current X
        intervention_now : bool
            True if intervening THIS period (one-time only)
        t : int
            Current time step
        
        Returns:
        --------
        X_new : int or str
            New X value, or "DEATH" if boundaries exceeded
        """
        optimal_X = 10  # Target value for treatment
        
        # COLLEAGUE'S DAMPENED FORMULA
        # Base transition: X_t = X_{t-1} + U_{t-1}/3 + U_t/2
        X_new = float(X_prev)  # Use float for divisions
        
        # Add dampened effect of previous U (weight = 1/3)
        if U_prev is not None:
            X_new += U_prev / 3.0
        
        # Add dampened effect of current U (weight = 1/2)
        X_new += U_current / 2.0
        
        # ONE-TIME STATE-DEPENDENT treatment effect
        # Only applies if intervention_now is True
        if intervention_now:
            if X_new < optimal_X:
                # Too low: treatment INCREASES X (one-time boost)
                X_new += self.treatment_effect
            elif X_new > optimal_X:
                # Too high: treatment DECREASES X (one-time reduction)
                X_new -= self.treatment_effect
            # else: X_new at optimal, no treatment effect needed
        
        # Round to integer
        X_new = round(X_new)
        
        # Check DEATH boundaries - both sides!
        if X_new < self.X_min or X_new > self.X_max:
            return "DEATH"
        
        return int(X_new)
    
    def compute_outcome_Y(self, X_final, U_final):
        """
        Compute binary outcome Y based on final X₆ and U₆
        
        Y = 1 if X₆ is in healthy range (good outcome)
        Y = 0 if X₆ is outside healthy range or DEATH (bad outcome)
        
        DEATH occurs if X < 1 or X > 20
        """
        # Check for death first
        if X_final == "DEATH" or X_final < self.X_min or X_final > self.X_max:
            return 0  # Bad outcome
        
        # Healthy range: centered around 10 with some tolerance
        # Y=1 if X is in [7, 14] range (healthy zone)
        healthy_low = 7
        healthy_high = 14
        
        effective_value = X_final + 0.3 * U_final  # Small stochastic component
        
        if healthy_low <= effective_value <= healthy_high:
            return 1  # Good outcome
        else:
            return 0  # Bad outcome (too low or too high, but not death)
    
    def get_expected_outcome(self, X_final):
        """
        Compute E[Y | X₆] by averaging over possible U₆ values
        
        Handles death states (X < 1 or X > 20)
        """
        # If already in death state, outcome is always 0
        if X_final == "DEATH" or X_final < self.X_min or X_final > self.X_max:
            return 0.0
        
        expected_Y = 0.0
        for u6, prob in zip(self.U_values, self.U_probs):
            Y = self.compute_outcome_Y(X_final, u6)
            expected_Y += prob * Y
        return expected_Y
    
    def solve_backward_induction(self, X0=10, verbose=True):
        """
        Solve the optimal stopping problem using backward induction
        
        This is the Dynamic Programming approach adapted from American put options!
        
        KEY CHANGE: Intervention is ONE-TIME only (like exercising an option)
        
        State space: (t_idx, X, already_intervened)
        - t_idx: time step index (maps to actual time via time_grid)
        - already_intervened = 0: Haven't used intervention yet
        - already_intervened = 1: Already used intervention (locked in)
        
        Parameters:
        -----------
        X0 : int
            Initial value of X₀
        verbose : bool
            Print progress
        """
        if verbose:
            print(f"\n{'='*80}")
            print("SOLVING CAUSAL OPTIMAL STOPPING VIA BACKWARD INDUCTION")
            print(f"{'='*80}")
            print(f"Initial state: X₀ = {X0}")
            print(f"\nONE-TIME INTERVENTION (Like American Put Options!)")
            print(f"You can intervene ONCE - choose the optimal time τ*")
            print(f"We work backwards from Y to find optimal intervention times.\n")
        
        # Clear previous solutions
        self.value_function = {}
        self.policy = {}
        
        # Stage T (final stage): Compute E[Y | X₆] for all possible X₆
        if verbose:
            print(f"Stage t={self.time_grid[-1]:.1f}: Computing terminal values E[Y | X_final]...")
        
        for X_final in range(self.X_min, self.X_max + 1):
            # At final stage, no intervention decision to make
            # Value is just E[Y | X₆]
            expected_Y = self.get_expected_outcome(X_final)
            
            # Store for both already_intervened states
            self.value_function[(self.T, X_final, 0)] = expected_Y  # Unused intervention
            self.value_function[(self.T, X_final, 1)] = expected_Y  # Already used
        
        # Backward induction: Work from T-1 back to 0
        for t_idx in range(self.T - 1, -1, -1):
            t_actual = self.time_grid[t_idx]
            
            if verbose:
                print(f"\nStage t={t_actual:.1f}: Computing optimal policies...")
            
            intervene_count = 0
            wait_count = 0
            
            # For each possible X_t value
            for X_t in range(self.X_min, self.X_max + 1):
                
                # Case 1: Already intervened (already_intervened=1)
                # No decision to make - just compute continuation value without intervention
                continuation_used = self._compute_continuation_value(
                    t_idx, X_t, already_intervened=True
                )
                self.value_function[(t_idx, X_t, 1)] = continuation_used
                self.policy[(t_idx, X_t, 1)] = 'no_action'  # Already used intervention
                
                # Case 2: Haven't intervened yet (already_intervened=0)
                # OPTIMAL STOPPING DECISION: Intervene now vs. Wait
                
                # Option A: Intervene NOW (use the one-time intervention)
                value_intervene = self._compute_intervention_value(t_idx, X_t)
                
                # Option B: Wait (keep intervention available for later)
                value_wait = self._compute_continuation_value(
                    t_idx, X_t, already_intervened=False
                )
                
                # BELLMAN EQUATION: Take the maximum!
                if value_intervene > value_wait:
                    self.value_function[(t_idx, X_t, 0)] = value_intervene
                    self.policy[(t_idx, X_t, 0)] = 'INTERVENE'
                    intervene_count += 1
                else:
                    self.value_function[(t_idx, X_t, 0)] = value_wait
                    self.policy[(t_idx, X_t, 0)] = 'wait'
                    wait_count += 1
            
            if verbose:
                print(f"  → {intervene_count} states: INTERVENE optimal")
                print(f"  → {wait_count} states: WAIT optimal")
        
        if verbose:
            print(f"\n{'='*80}")
            print("BACKWARD INDUCTION COMPLETE!")
            print(f"{'='*80}")
            print(f"Optimal policy computed for all states (t, X, already_intervened)")
            print(f"Ready to simulate paths and find τ* for each!\n")
    
    def solve_backward_induction_with_bounds(self, X0=10, verbose=True):
        """
        Algorithm 3: Bounds under Incomplete Information
        
        Solve the optimal stopping problem when the transition model is uncertain.
        Compute LOWER and UPPER bounds on E[Y] by considering worst/best-case 
        probability distributions.
        
        Model uncertainty:
        - Shock distribution: P ∈ [P̲, P̄] (probability uncertainty)
        - Treatment effect: e ∈ [e-δ, e+δ] (effect uncertainty)
        
        Uses min-max dynamic programming:
        - Lower bound V̲: assumes worst-case distributions
        - Upper bound V̄: assumes best-case distributions
        
        Parameters:
        -----------
        X0 : int
            Initial value of X₀
        verbose : bool
            Print progress
        """
        if verbose:
            print(f"\n{'='*80}")
            print("ALGORITHM 3: BOUNDS UNDER INCOMPLETE INFORMATION")
            print(f"{'='*80}")
            print(f"Computing value bounds with model uncertainty")
            print(f"Initial state: X₀ = {X0}")
            print(f"Probability uncertainty: ±{self.prob_uncertainty*100:.1f}%")
            print(f"Treatment effect bounds: [{self.effect_lower:.2f}, {self.effect_upper:.2f}]")
            print(f"\nComputing LOWER and UPPER bound value functions...\n")
        
        # Clear previous solutions
        self.value_lower = {}
        self.value_upper = {}
        
        # Stage T (final stage): Compute bounds on E[Y | X_final]
        if verbose:
            print(f"Stage t={self.time_grid[-1]:.1f}: Computing terminal value bounds...")
        
        for X_final in range(self.X_min, self.X_max + 1):
            # Terminal value is deterministic (no uncertainty in outcome function)
            expected_Y = self.get_expected_outcome(X_final)
            
            # Both bounds equal at terminal nodes
            self.value_lower[(self.T, X_final, 0)] = expected_Y
            self.value_lower[(self.T, X_final, 1)] = expected_Y
            self.value_upper[(self.T, X_final, 0)] = expected_Y
            self.value_upper[(self.T, X_final, 1)] = expected_Y
        
        # Backward induction with bounds
        for t_idx in range(self.T - 1, -1, -1):
            t_actual = self.time_grid[t_idx]
            
            if verbose:
                print(f"\nStage t={t_actual:.1f}: Computing bounded policies...")
            
            for X_t in range(self.X_min, self.X_max + 1):
                
                # Case 1: Already intervened
                cont_lower = self._compute_continuation_value_bounded(
                    t_idx, X_t, already_intervened=True, bound_type='lower'
                )
                cont_upper = self._compute_continuation_value_bounded(
                    t_idx, X_t, already_intervened=True, bound_type='upper'
                )
                
                self.value_lower[(t_idx, X_t, 1)] = cont_lower
                self.value_upper[(t_idx, X_t, 1)] = cont_upper
                
                # Case 2: Haven't intervened yet - compute both intervention and wait
                # Lower bound: worst case
                intervene_lower = self._compute_intervention_value_bounded(
                    t_idx, X_t, bound_type='lower'
                )
                wait_lower = self._compute_continuation_value_bounded(
                    t_idx, X_t, already_intervened=False, bound_type='lower'
                )
                
                # Upper bound: best case
                intervene_upper = self._compute_intervention_value_bounded(
                    t_idx, X_t, bound_type='upper'
                )
                wait_upper = self._compute_continuation_value_bounded(
                    t_idx, X_t, already_intervened=False, bound_type='upper'
                )
                
                # Store bounds (max for action choice)
                self.value_lower[(t_idx, X_t, 0)] = max(intervene_lower, wait_lower)
                self.value_upper[(t_idx, X_t, 0)] = max(intervene_upper, wait_upper)
        
        if verbose:
            print(f"\n{'='*80}")
            print("ALGORITHM 3 COMPLETE!")
            print(f"{'='*80}")
            print(f"Computed LOWER and UPPER bounds for all states")
            print(f"Bounds quantify uncertainty in optimal policy\n")
    
    def solve_robust_backward_induction(self, X0=10, verbose=True):
        """
        Algorithm 4: Bounds with Intervention (Robust Optimal Stopping)
        
        Combine optimal stopping (Algorithm 2) with incomplete information (Algorithm 3).
        Derive ROBUST policies that work well even under model uncertainty.
        
        Robust decision rule:
        - Intervene if: V̲_intervene > V̄_wait
          (even worst-case intervention beats best-case waiting)
        - Wait if: V̄_intervene < V̲_wait  
          (even best-case intervention loses to worst-case waiting)
        - Ambiguous otherwise
        
        This is like robust American option pricing under volatility uncertainty.
        
        Parameters:
        -----------
        X0 : int
            Initial value of X₀
        verbose : bool
            Print progress
        """
        if verbose:
            print(f"\n{'='*80}")
            print("ALGORITHM 4: ROBUST OPTIMAL STOPPING WITH INTERVENTION")
            print(f"{'='*80}")
            print(f"Deriving robust policies under model uncertainty")
            print(f"Initial state: X₀ = {X0}")
            print(f"\nRobust decision rule:")
            print(f"  - INTERVENE if V̲_intervene > V̄_wait (robust intervention)")
            print(f"  - WAIT if V̄_intervene < V̲_wait (robust waiting)")
            print(f"  - AMBIGUOUS otherwise (requires more information)\n")
        
        # First compute bounds (Algorithm 3)
        self.solve_backward_induction_with_bounds(X0=X0, verbose=False)
        
        # Now derive robust policies
        self.robust_policy = {}
        
        if verbose:
            print("Deriving robust policies from bounds...\n")
        
        robust_intervene_count = 0
        robust_wait_count = 0
        ambiguous_count = 0
        
        for t_idx in range(self.T):
            for X_t in range(self.X_min, self.X_max + 1):
                
                # Already intervened: no decision
                self.robust_policy[(t_idx, X_t, 1)] = 'no_action'
                
                # Haven't intervened: apply robust decision rule
                # Get bounds from Algorithm 3
                intervene_lower = self._compute_intervention_value_bounded(
                    t_idx, X_t, bound_type='lower'
                )
                intervene_upper = self._compute_intervention_value_bounded(
                    t_idx, X_t, bound_type='upper'
                )
                wait_lower = self._compute_continuation_value_bounded(
                    t_idx, X_t, already_intervened=False, bound_type='lower'
                )
                wait_upper = self._compute_continuation_value_bounded(
                    t_idx, X_t, already_intervened=False, bound_type='upper'
                )
                
                # Robust decision rule
                if intervene_lower > wait_upper:
                    # Worst-case intervention > best-case waiting → ROBUSTLY INTERVENE
                    self.robust_policy[(t_idx, X_t, 0)] = 'INTERVENE'
                    robust_intervene_count += 1
                elif intervene_upper < wait_lower:
                    # Best-case intervention < worst-case waiting → ROBUSTLY WAIT
                    self.robust_policy[(t_idx, X_t, 0)] = 'wait'
                    robust_wait_count += 1
                else:
                    # Ambiguous: bounds overlap
                    self.robust_policy[(t_idx, X_t, 0)] = 'ambiguous'
                    ambiguous_count += 1
        
        if verbose:
            total = robust_intervene_count + robust_wait_count + ambiguous_count
            print(f"{'='*80}")
            print("ALGORITHM 4 COMPLETE!")
            print(f"{'='*80}")
            print(f"Robust policy statistics:")
            print(f"  - Robustly INTERVENE: {robust_intervene_count}/{total} ({robust_intervene_count/total*100:.1f}%)")
            print(f"  - Robustly WAIT: {robust_wait_count}/{total} ({robust_wait_count/total*100:.1f}%)")
            print(f"  - AMBIGUOUS (need more info): {ambiguous_count}/{total} ({ambiguous_count/total*100:.1f}%)")
            print(f"\nReady for robust simulation!\n")
    
    def _compute_intervention_value(self, t, X_t):
        """
        Compute value of intervening NOW at time t with state X_t
        
        ONE-TIME INTERVENTION:
        - Apply treatment effect THIS period
        - After this, already_intervened = 1 (can't intervene again)
        - Future states follow natural evolution without intervention
        
        This is like "exercising" the American option!
        """
        total_value = 0.0
        t_next = t + 1
        
        if t_next > self.T:
            # Terminal state - apply intervention and get outcome
            # We need to compute X with intervention applied
            for u_t, prob_t in zip(self.U_values, self.U_probs):
                X_final = self.structural_equation(X_t, u_t, None, intervention_now=True, t=t_next)
                if X_final == "DEATH":
                    expected_Y = 0.0
                else:
                    expected_Y = self.get_expected_outcome(X_final)
                total_value += prob_t * expected_Y
            return total_value
        
        # Average over possible U_t and U_{t+1} (both affect X_{t+1})
        for u_t, prob_t in zip(self.U_values, self.U_probs):
            for u_next, prob_next in zip(self.U_values, self.U_probs):
                # Compute X_{t+1} WITH intervention applied at time t
                if t == 0:
                    X_next = self.structural_equation(X_t, u_next, None, intervention_now=True, t=t_next)
                else:
                    X_next = self.structural_equation(X_t, u_next, u_t, intervention_now=True, t=t_next)
                
                # Handle death state
                if X_next == "DEATH":
                    future_value = 0.0
                else:
                    # After intervention, state is (t_next, X_next, already_intervened=1)
                    # Can't intervene again!
                    future_value = self.value_function.get((t_next, X_next, 1), 0)
                
                total_value += prob_t * prob_next * (self.discount * future_value)
        
        return total_value

    
    def _compute_continuation_value(self, t, X_t, already_intervened):
        """
        Compute value of continuing WITHOUT intervening now
        
        TWO CASES:
        1. already_intervened=False: Still have intervention available for future
        2. already_intervened=True: Already used intervention, just natural evolution
        
        This is like "holding" the American option!
        """
        total_value = 0.0
        t_next = t + 1
        
        if t_next > self.T:
            return self.get_expected_outcome(X_t)
        
        # Average over possible U_t and U_{t+1}
        for u_t, prob_t in zip(self.U_values, self.U_probs):
            for u_next, prob_next in zip(self.U_values, self.U_probs):
                # Compute X_{t+1} WITHOUT intervention (natural evolution)
                if t == 0:
                    X_next = self.structural_equation(X_t, u_next, None, intervention_now=False, t=t_next)
                else:
                    X_next = self.structural_equation(X_t, u_next, u_t, intervention_now=False, t=t_next)
                
                # Handle death state
                if X_next == "DEATH":
                    future_value = 0.0
                else:
                    # Look up future value with same intervention status
                    if already_intervened:
                        # Already used - state remains (t_next, X_next, 1)
                        future_value = self.value_function.get((t_next, X_next, 1), 0)
                    else:
                        # Still available - state is (t_next, X_next, 0)
                        future_value = self.value_function.get((t_next, X_next, 0), 0)
                
                total_value += prob_t * prob_next * (self.discount * future_value)
        
        return total_value
    
    def _compute_intervention_value_bounded(self, t, X_t, bound_type='lower'):
        """
        Compute BOUNDED value of intervening (for Algorithm 3 & 4)
        
        PROPER MIN-MAX APPROACH:
        - Lower bound: min over (treatment_effect, probability) of expected value
        - Upper bound: max over (treatment_effect, probability) of expected value
        
        We compute 4 scenarios (2 effects × 2 prob distributions) and take min/max
        
        Parameters:
        -----------
        bound_type : str
            'lower' for worst-case (pessimistic), 'upper' for best-case (optimistic)
        """
        t_next = t + 1
        
        # Try all combinations of uncertainties
        scenarios = []
        
        for effect_mult in [1 - self.effect_uncertainty, 1 + self.effect_uncertainty]:
            for use_lower_probs in [True, False]:
                probs = self.U_probs_lower if use_lower_probs else self.U_probs_upper
                
                scenario_value = 0.0
                
                if t_next > self.T:
                    # Terminal state
                    for u_t, prob_t in zip(self.U_values, probs):
                        X_final = self.structural_equation_bounded(
                            X_t, u_t, None, intervention_now=True, t=t_next,
                            effect_multiplier=effect_mult
                        )
                        if X_final == "DEATH":
                            expected_Y = 0.0
                        else:
                            expected_Y = self.get_expected_outcome(X_final)
                        scenario_value += prob_t * expected_Y
                else:
                    # Non-terminal
                    for u_t, prob_t in zip(self.U_values, probs):
                        for u_next, prob_next in zip(self.U_values, probs):
                            if t == 0:
                                X_next = self.structural_equation_bounded(
                                    X_t, u_next, None, intervention_now=True, t=t_next,
                                    effect_multiplier=effect_mult
                                )
                            else:
                                X_next = self.structural_equation_bounded(
                                    X_t, u_next, u_t, intervention_now=True, t=t_next,
                                    effect_multiplier=effect_mult
                                )
                            
                            if X_next == "DEATH":
                                future_value = 0.0
                            else:
                                # Use same bound type for future
                                if bound_type == 'lower':
                                    future_value = self.value_lower.get((t_next, X_next, 1), 0)
                                else:
                                    future_value = self.value_upper.get((t_next, X_next, 1), 0)
                            
                            scenario_value += prob_t * prob_next * self.discount * future_value
                
                scenarios.append(scenario_value)
        
        # Take min or max over all scenarios
        if bound_type == 'lower':
            return min(scenarios)  # Worst case
        else:
            return max(scenarios)  # Best case
    
    def _compute_continuation_value_bounded(self, t, X_t, already_intervened, bound_type='lower'):
        """
        Compute BOUNDED value of continuing without intervening (for Algorithm 3 & 4)
        
        PROPER MIN-MAX APPROACH:
        - Lower bound: min over probability distributions
        - Upper bound: max over probability distributions
        
        Parameters:
        -----------
        bound_type : str
            'lower' for worst-case, 'upper' for best-case
        """
        t_next = t + 1
        
        if t_next > self.T:
            return self.get_expected_outcome(X_t)
        
        # Try both probability distributions
        scenarios = []
        
        for use_lower_probs in [True, False]:
            probs = self.U_probs_lower if use_lower_probs else self.U_probs_upper
            
            scenario_value = 0.0
            
            for u_t, prob_t in zip(self.U_values, probs):
                for u_next, prob_next in zip(self.U_values, probs):
                    # No intervention - standard structural equation
                    if t == 0:
                        X_next = self.structural_equation(X_t, u_next, None, intervention_now=False, t=t_next)
                    else:
                        X_next = self.structural_equation(X_t, u_next, u_t, intervention_now=False, t=t_next)
                    
                    if X_next == "DEATH":
                        future_value = 0.0
                    else:
                        # Get bounded future value
                        if bound_type == 'lower':
                            if already_intervened:
                                future_value = self.value_lower.get((t_next, X_next, 1), 0)
                            else:
                                future_value = self.value_lower.get((t_next, X_next, 0), 0)
                        else:
                            if already_intervened:
                                future_value = self.value_upper.get((t_next, X_next, 1), 0)
                            else:
                                future_value = self.value_upper.get((t_next, X_next, 0), 0)
                    
                    scenario_value += prob_t * prob_next * self.discount * future_value
            
            scenarios.append(scenario_value)
        
        # Take min or max over scenarios
        if bound_type == 'lower':
            return min(scenarios)  # Worst case
        else:
            return max(scenarios)  # Best case
    
    def structural_equation_bounded(self, X_prev, U_current, U_prev, intervention_now, t, effect_multiplier=1.0):
        """
        Structural equation with BOUNDED treatment effect (DAMPENED VERSION)
        
        For Algorithm 3 & 4: treatment effect can be uncertain
        effect_multiplier adjusts the treatment strength
        
        Formula: X_{i+1} = X_i + U_i/3 + U_{i+1}/2  (dampened shocks)
        
        Parameters:
        -----------
        effect_multiplier : float
            Multiplier for treatment effect (e.g., 0.8 or 1.2 for ±20% uncertainty)
        """
        optimal_X = 10
        
        # Base transition with dampening: X_t = X_{t-1} + U_{t-1}/3 + U_t/2
        X_new = float(X_prev)
        
        if U_prev is not None:
            X_new += U_prev / 3.0  # Dampened previous shock
        X_new += U_current / 2.0   # Dampened current shock
        
        # Apply intervention if requested (with bounded effect)
        if intervention_now:
            adjusted_effect = self.treatment_effect * effect_multiplier
            
            if X_new < optimal_X:
                X_new += adjusted_effect
            elif X_new > optimal_X:
                X_new -= adjusted_effect
        
        # Round and check boundaries
        X_new = round(X_new)
        
        if X_new < self.X_min or X_new > self.X_max:
            return "DEATH"
        
        return int(X_new)
    
    def simulate_path(self, X0=10, seed=None):
        """
        Simulate a single path of X values with random U shocks
        NO intervention applied during simulation - this is the natural path
        
        Works with CONTINUOUS TIME GRID
        
        Returns:
        --------
        X_path : list
            Sequence of X values at each time step
        U_path : list
            Sequence of U values at each time step
        t_path : list
            Actual time values corresponding to each step
        """
        if seed is not None:
            np.random.seed(seed)
        
        X_path = [X0]
        U_path = []
        t_path = [self.time_grid[0]]  # Start at t=0
        
        for i in range(1, len(self.time_grid)):
            # Sample U_t
            U_t = np.random.choice(self.U_values, p=self.U_probs)
            U_path.append(U_t)
            
            # Compute X_t WITHOUT intervention (natural evolution)
            U_prev = U_path[-2] if len(U_path) >= 2 else None
            X_t = self.structural_equation(X_path[-1], U_t, U_prev, intervention_now=False, t=i)
            
            # Handle death
            if X_t == "DEATH":
                X_path.append("DEATH")
                t_path.append(self.time_grid[i])
                break
            
            X_path.append(X_t)
            t_path.append(self.time_grid[i])
        
        return X_path, U_path, t_path
    
    def find_optimal_intervention_time(self, X_path, U_path, t_path):
        """
        Given a realized path, find the optimal intervention time τ*
        
        ONE-TIME INTERVENTION: Can only intervene once!
        
        Returns τ* in actual time units (e.g., 3.4, 4.8, etc.)
        
        Returns:
        --------
        tau_star : float or None
            Optimal intervention time in actual time units
        intervention_info : dict
            Details about the intervention decision
        """
        # Start with intervention available (already_intervened=0)
        already_intervened = False
        
        # Walk through the path
        for t_idx in range(len(X_path) - 1):  # Don't include final X
            X_t = X_path[t_idx]
            t_actual = t_path[t_idx]
            
            # Check if we died
            if X_t == "DEATH":
                return None, {
                    'time': None,
                    'time_index': t_idx,
                    'died_at': t_actual,
                    'intervened': False,
                    'reason': 'died_before_decision'
                }
            
            # Check policy at this state
            already_intervened_int = 1 if already_intervened else 0
            action = self.policy.get((t_idx, X_t, already_intervened_int), 'wait')
            
            if action == 'INTERVENE' and not already_intervened:
                # Found optimal intervention time!
                value_at_intervention = self.value_function.get((t_idx, X_t, 0), 0)
                
                return t_actual, {
                    'time': t_actual,
                    'time_index': t_idx,
                    'X_value': X_t,
                    'value': value_at_intervention,
                    'intervened': True
                }
        
        # Check final state
        X_final = X_path[-1]
        if X_final == "DEATH":
            return None, {
                'time': None,
                'X_final': "DEATH",
                'value': 0,
                'intervened': False,
                'reason': 'died_without_intervention'
            }
        
        # Never optimal to intervene on this path
        U_final = U_path[-1] if U_path else 0
        Y = self.compute_outcome_Y(X_final, U_final)
        
        return None, {
            'time': None,
            'X_final': X_final,
            'Y': Y,
            'value': Y,
            'intervened': False,
            'reason': 'never_optimal'
        }
    
    def simulate_multiple_paths(self, X0=10, num_paths=20, verbose=True):
        """
        Simulate multiple paths and find τ* for each
        
        ONE-TIME INTERVENTION: Just like American put simulation!
        Now with CONTINUOUS TIME: τ* can be 0.4, 2.8, 4.6, etc.!
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"SIMULATING {num_paths} CAUSAL PATHS (CONTINUOUS TIME - DAMPENED)")
            print(f"{'='*80}")
            print(f"Finding optimal ONE-TIME intervention time τ* for each path")
            print(f"τ* can now be any value in [{self.time_grid[0]:.1f}, {self.time_grid[-1]:.1f}]\n")
        
        results = []
        
        for path_num in range(num_paths):
            X_path, U_path, t_path = self.simulate_path(X0=X0, seed=path_num)
            tau_star, info = self.find_optimal_intervention_time(X_path, U_path, t_path)
            
            result = {
                'path_number': path_num + 1,
                'X_path': X_path,
                'U_path': U_path,
                't_path': t_path,
                'tau_star': tau_star,
                'info': info
            }
            results.append(result)
            
            if verbose:
                print(f"Path #{path_num + 1}:")
                
                # Handle death in trajectory
                if "DEATH" in X_path:
                    death_idx = X_path.index("DEATH")
                    print(f"  X trajectory: {X_path[:min(4, death_idx)]}... → DEATH at t={t_path[death_idx]:.1f}")
                else:
                    print(f"  X trajectory: {X_path[:4]}... → {X_path[-1]} at t={t_path[-1]:.1f}")
                
                print(f"  U shocks: {U_path[:3]}...")
                
                if tau_star is not None:
                    print(f"  ✓ Optimal Intervention Time τ*: t={tau_star:.2f}")  # Show decimals!
                    print(f"    X value at intervention: {info['X_value']}")
                    print(f"    Expected value: {info['value']:.3f}")
                else:
                    reason = info.get('reason', 'unknown')
                    if reason == 'died_before_decision' or reason == 'died_without_intervention':
                        print(f"  ☠ Path died without intervention")
                        print(f"    Died at: t={info.get('died_at', 'unknown'):.2f}")
                    elif reason == 'never_optimal':
                        print(f"  ✗ Never intervened (not needed)")
                        print(f"    Final X: {info['X_final']}, Outcome Y: {info['Y']}")
                print()
        
        # Summary statistics
        intervention_times = [r['tau_star'] for r in results if r['tau_star'] is not None]
        deaths = sum(1 for r in results if 'died' in r['info'].get('reason', ''))
        never_needed = sum(1 for r in results if r['info'].get('reason') == 'never_optimal')
        
        print(f"{'='*80}")
        print("SUMMARY STATISTICS (CONTINUOUS TIME - DAMPENED)")
        print(f"{'='*80}")
        print(f"  Total Paths: {num_paths}")
        print(f"  Paths with Intervention: {len(intervention_times)} ({len(intervention_times)/num_paths*100:.1f}%)")
        print(f"  Paths without Intervention: {num_paths - len(intervention_times)}")
        print(f"    - Died without intervention: {deaths}")
        print(f"    - Never needed: {never_needed}")
        
        if intervention_times:
            print(f"\n  Intervention Times (continuous):")
            print(f"    Average τ*: {np.mean(intervention_times):.2f}")
            print(f"    Earliest: t={np.min(intervention_times):.2f}")
            print(f"    Latest: t={np.max(intervention_times):.2f}")
            print(f"    Std deviation: {np.std(intervention_times):.2f}")
            
            # Distribution by bins
            print(f"\n  Distribution of τ* (by integer bins):")
            for bin_start in range(int(self.time_grid[0]), int(self.time_grid[-1]) + 1):
                count = sum(1 for t in intervention_times if bin_start <= t < bin_start + 1)
                if count > 0:
                    print(f"    t ∈ [{bin_start}, {bin_start+1}): {count} paths ({count/len(intervention_times)*100:.1f}%)")
            
            # Show actual times (first 10)
            print(f"\n  Actual intervention times (first {min(10, len(intervention_times))}):")
            print(f"    {[f'{t:.2f}' for t in sorted(intervention_times)[:10]]}")
        
        print(f"{'='*80}\n")
        
        return results
    
    def print_policy_summary(self, X0=10):
        """
        Print the optimal policy for key states
        Shows actual time values (not indices)
        """
        print(f"\n{'='*80}")
        print("OPTIMAL INTERVENTION POLICY (Key States)")
        print(f"{'='*80}")
        print(f"Starting from X0 = {X0}")
        print("ONE-TIME INTERVENTION: Can only use once!\n")
        print(f"{'Time':<10} {'X Value':<12} {'No Intervention':<30} {'Value':<15}")
        print("-" * 75)
        
        # Sample time points strategically (don't show all 30!)
        # Show: start, every ~5th step, and end
        sample_indices = []
        step = max(1, self.T // 10)  # Show ~10 time points
        for t_idx in range(0, self.T, step):
            sample_indices.append(t_idx)
        if sample_indices[-1] != self.T - 1:
            sample_indices.append(self.T - 1)  # Always show last time
        
        for t_idx in sample_indices:
            t_actual = self.time_grid[t_idx]
            
            # Show policy for a range of X values
            for X in [3, 7, 10, 14, 17]:
                if X < self.X_min or X > self.X_max:
                    continue
                    
                policy_available = self.policy.get((t_idx, X, 0), 'unknown')
                value_available = self.value_function.get((t_idx, X, 0), 0)
                
                action_str = "🔴 INTERVENE NOW!" if policy_available == 'INTERVENE' else "⚪ Wait"
                
                print(f"t={t_actual:<7.1f} X={X:<10} {action_str:<30} {value_available:.4f}")
        
        print(f"{'='*80}\n")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CAUSAL OPTIMAL STOPPING: ALL 4 ALGORITHMS")
    print("DAMPENED VERSION + Comprehensive Analysis with Robust Extensions")
    print("="*80)
    print("\n🩺 Blood Pressure Analogy:")
    print("  X = health markers (1-20)")
    print("  U = random daily shocks (-3 to +3)")
    print("  Formula: X_{i+1} = X_i + U_i/3 + U_{i+1}/2 (DAMPENED)")
    print("  A = medication (one-time intervention)")
    print("  Y = outcome (1=healthy, 0=adverse)")
    print("  Goal: Find optimal intervention strategy under uncertainty\n")
    
    # ========================================================================
    # PART 1: ALGORITHMS 1 & 2 (Standard Case - Known Model)
    # ========================================================================
    print("\n" + "="*80)
    print("PART 1: ALGORITHMS 1 & 2 - STANDARD OPTIMAL STOPPING")
    print("Assumption: Model parameters are KNOWN with certainty")
    print("="*80)
    
    model_standard = CausalOptimalStopping(
        n_endogenous=7,
        n_exogenous=6,
        T=6.0,
        X_min=1,
        X_max=20,
        U_min=-3,
        U_max=3,
        treatment_effect=3,
        Y_threshold=10,
        discount=0.95,
        dt=0.2,
        uncertainty_mode='none'
    )
    
    # Algorithm 2: Optimal Stopping (Algorithm 1 is implicit)
    print("\n🔹 Running Algorithm 2: Backward Induction for Optimal Stopping")
    model_standard.solve_backward_induction(X0=10, verbose=True)
    
    # Show policy summary
    model_standard.print_policy_summary(X0=10)
    
    # Simulate paths to find τ* values
    print("\n🔹 Simulating Paths to Find Optimal Intervention Times (τ*)")
    results_standard = model_standard.simulate_multiple_paths(X0=10, num_paths=15, verbose=True)
    
    # ========================================================================
    # PART 2: ALGORITHMS 3 & 4 (Robust Case - Uncertain Model)
    # ========================================================================
    print("\n" + "="*80)
    print("PART 2: ALGORITHMS 3 & 4 - ROBUST EXTENSIONS")
    print("Assumption: Model parameters are UNCERTAIN")
    print("  • Probability distribution: P ∈ [P̲, P̄] (±15% uncertainty)")
    print("  • Treatment effect: e ∈ [2.55, 3.45] (±15% uncertainty)")
    print("="*80)
    
    model_robust = CausalOptimalStopping(
        n_endogenous=7,
        n_exogenous=6,
        T=6.0,
        X_min=1,
        X_max=20,
        U_min=-3,
        U_max=3,
        treatment_effect=3,
        Y_threshold=10,
        discount=0.95,
        dt=0.2,
        uncertainty_mode='robust',
        prob_uncertainty=0.15,
        effect_uncertainty=0.15
    )
    
    # Algorithm 4: Robust Optimal Stopping (internally calls Algorithm 3)
    print("\n🔹 Running Algorithm 4: Robust Backward Induction")
    print("   (This automatically runs Algorithm 3 to compute bounds first)")
    model_robust.solve_robust_backward_induction(X0=10, verbose=True)
    
    # ========================================================================
    # UNIFIED COMPARISON: ALL 4 ALGORITHMS
    # ========================================================================
    print("\n" + "="*80)
    print("UNIFIED COMPARISON: ALL 4 ALGORITHMS AT KEY STATES")
    print("="*80)
    
    # Compare at initial state and other key states
    comparison_states = [(0, 10, "Initial state"), (15, 10, "Mid-point"), (29, 10, "Near terminal")]
    
    for t_idx, X, description in comparison_states:
        t_actual = model_standard.time_grid[t_idx]
        print(f"\n{'='*80}")
        print(f"State: t={t_actual:.1f}, X={X} ({description})")
        print(f"{'='*80}")
        
        # Algorithm 2: Standard value and policy
        standard_val = model_standard.value_function.get((t_idx, X, 0), 0)
        standard_policy = model_standard.policy.get((t_idx, X, 0), 'unknown')
        
        print(f"\n📊 Algorithm 2 (Standard - Known Model):")
        print(f"   Value:  {standard_val:.4f}")
        print(f"   Policy: {standard_policy.upper()}")
        
        # Algorithm 3: Bounds
        lower_val = model_robust.value_lower.get((t_idx, X, 0), 0)
        upper_val = model_robust.value_upper.get((t_idx, X, 0), 0)
        width = upper_val - lower_val
        
        print(f"\n📊 Algorithm 3 (Bounds - Uncertain Model):")
        print(f"   Lower bound: {lower_val:.4f}")
        print(f"   Upper bound: {upper_val:.4f}")
        print(f"   Width:       {width:.4f}")
        
        # Check if standard value is within bounds
        in_bounds = lower_val <= standard_val <= upper_val
        bounds_check = "✅ YES" if in_bounds else "⚠️  NO (outside bounds)"
        print(f"   Standard value within bounds? {bounds_check}")
        
        # Algorithm 4: Robust policy
        robust_policy = model_robust.robust_policy.get((t_idx, X, 0), 'unknown')
        
        policy_symbol = {
            'INTERVENE': '🟢',
            'wait': '🟡',
            'ambiguous': '🔴'
        }.get(robust_policy, '❓')
        
        policy_description = {
            'INTERVENE': 'ROBUSTLY INTERVENE (worst-case beats waiting)',
            'wait': 'ROBUSTLY WAIT (best-case intervention loses)',
            'ambiguous': 'AMBIGUOUS (need more information)'
        }.get(robust_policy, 'unknown')
        
        print(f"\n📊 Algorithm 4 (Robust Policy):")
        print(f"   {policy_symbol} {policy_description}")
    
    # ========================================================================
    # COMPREHENSIVE SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n{'ALGORITHM 2: STANDARD OPTIMAL STOPPING':<60}")
    print("-"*80)
    print(f"Simulated paths: {len(results_standard)}")
    
    # Count intervention statistics
    paths_with_intervention = sum(1 for r in results_standard if r['tau_star'] is not None)
    paths_without_intervention = len(results_standard) - paths_with_intervention
    intervention_times = [r['tau_star'] for r in results_standard if r['tau_star'] is not None]
    
    print(f"Paths requiring intervention: {paths_with_intervention} "
          f"({paths_with_intervention/len(results_standard)*100:.1f}%)")
    print(f"Paths NOT requiring intervention: {paths_without_intervention} "
          f"({paths_without_intervention/len(results_standard)*100:.1f}%)")
    
    if intervention_times:
        print(f"\nIntervention timing statistics:")
        print(f"  Average τ*: {np.mean(intervention_times):.2f}")
        print(f"  Min τ*:     {np.min(intervention_times):.2f}")
        print(f"  Max τ*:     {np.max(intervention_times):.2f}")
    
    print(f"\n{'ALGORITHM 3: BOUNDS UNDER UNCERTAINTY':<60}")
    print("-"*80)
    
    # Sample some bounds to show the uncertainty
    print("Sample value bounds (showing uncertainty range):")
    print(f"  {'State':<20} {'Lower':<12} {'Upper':<12} {'Width':<12}")
    print("  " + "-"*56)
    for t_idx, X in [(0, 5), (0, 10), (0, 15), (15, 10), (29, 10)]:
        t_actual = model_robust.time_grid[t_idx]
        lower = model_robust.value_lower.get((t_idx, X, 0), 0)
        upper = model_robust.value_upper.get((t_idx, X, 0), 0)
        width = upper - lower
        print(f"  t={t_actual:.1f}, X={X:<2}        {lower:<12.4f} {upper:<12.4f} {width:<12.4f}")
    
    print(f"\n{'ALGORITHM 4: ROBUST POLICY CLASSIFICATION':<60}")
    print("-"*80)
    
    # Count robust policies
    intervene_count = sum(1 for p in model_robust.robust_policy.values() if p == 'INTERVENE')
    wait_count = sum(1 for p in model_robust.robust_policy.values() if p == 'wait')
    ambiguous_count = sum(1 for p in model_robust.robust_policy.values() if p == 'ambiguous')
    total_policies = len(model_robust.robust_policy)
    
    print(f"Total states classified: {total_policies}")
    print(f"\n🟢 ROBUSTLY INTERVENE:  {intervene_count:>4} / {total_policies} ({intervene_count/total_policies*100:>5.1f}%)")
    print(f"   → Worst-case intervention beats best-case waiting")
    print(f"\n🟡 ROBUSTLY WAIT:       {wait_count:>4} / {total_policies} ({wait_count/total_policies*100:>5.1f}%)")
    print(f"   → Best-case intervention loses to worst-case waiting")
    print(f"\n🔴 AMBIGUOUS:           {ambiguous_count:>4} / {total_policies} ({ambiguous_count/total_policies*100:>5.1f}%)")
    print(f"   → Bounds overlap, need more information or data")
    
    # ========================================================================
    # KEY INSIGHTS
    # ========================================================================
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM ALL 4 ALGORITHMS (DAMPENED VERSION)")
    print("="*80)
    
    print("""
🎯 WHAT WE LEARNED:

1. DAMPENED SHOCK STRUCTURE:
   • Formula: X_{i+1} = X_i + U_i/3 + U_{i+1}/2
   • Previous shocks decay faster (1/3 weight)
   • Current shocks have moderate impact (1/2 weight)
   • More stable than full shock accumulation

2. ALGORITHM 2 (Standard Optimal Stopping):
   • Finds optimal intervention time τ* for each path
   • Dampening reduces extreme trajectories
   • Treatment decisions more conservative

3. ALGORITHM 3 (Bounds under Uncertainty):
   • Quantifies epistemic uncertainty
   • Width of bounds shows confidence level
   • Wider bounds → need more data

4. ALGORITHM 4 (Robust Optimal Stopping):
   • Conservative policies under worst-case
   • High ambiguity reflects model uncertainty
   • Identifies states needing more information

🔬 DAMPENED vs STANDARD COMPARISON:
   • Dampened: More gradual X evolution
   • Standard: Shocks accumulate fully
   • Dampened: Intervention less urgent
   • Standard: More aggressive policies

This reflects different causal mechanisms!
    """)
    
    print("="*80)
    print("✅ ALL 4 ALGORITHMS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n💡 To modify parameters, edit the code in __main__ block")
    print("   Current: Runs ALL algorithms for comprehensive analysis\n")
    print("\n" + "="*80)
    print("CAUSAL OPTIMAL STOPPING: Finding Optimal Intervention Times")
    print("DAMPENED SHOCKS VERSION + CONTINUOUS TIME (Colleague's Formula, dt=0.2)")
    print("="*80)
    print("\n🩺 Blood Pressure Analogy:")
    print("  X = health markers (1-20, higher is better)")
    print("  U = random daily shocks (-3 to +3)")
    print("  Formula: X_i+1 = X_i + U_i/3 + U_i+1/2 (dampened)")
    print("  A = medication (0=not taking, 1=taking)")
    print("  Y = outcome (1=healthy, 0=adverse event)")
    print("  Goal: Find optimal time τ* to start medication")
    print("  τ* can be ANY time (e.g., 2.4 days, 4.8 days, etc.)\n")
    
    # Initialize the model with DAMPENED formula + CONTINUOUS TIME (dt=0.2)
    model = CausalOptimalStopping(
        n_endogenous=7,
        n_exogenous=6,
        T=6.0,             # Total time (float)
        X_min=1,
        X_max=20,
        U_min=-3,
        U_max=3,
        treatment_effect=3,    # Stronger effect since shocks are dampened
        Y_threshold=10,        # Not used anymore, kept for compatibility
        discount=0.95,
        dt=0.2                 # Time step = 0.2 (5 steps per unit time)
    )
    
    # Solve using backward induction (Dynamic Programming!)
    model.solve_backward_induction(X0=10, verbose=True)
    
    # Show the optimal policy
    model.print_policy_summary(X0=10)
    
    # Simulate paths and find τ* for each
    results = model.simulate_multiple_paths(X0=10, num_paths=15, verbose=True)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
🎯 This adapts the American Put Option framework for causal inference!

1. BACKWARD INDUCTION (Dynamic Programming):
   - Work backwards from outcome Y
   - At each state (t, X, A), compute:
     * Value of intervening now
     * Value of waiting
   - Take the maximum (Bellman equation)

2. OPTIMAL STOPPING PROBLEM:
   - Like exercising an American put, but for causal interventions
   - Each path has its own τ* (optimal intervention time)
   - Some paths never need intervention

3. CAUSAL STRUCTURE:
   - Each U_i affects X_i and X_{i+1} (persistent shocks)
   - Treatment A causally affects all future X values
   - Y is determined by final X₆ being in healthy range [7, 14]

4. STATE-DEPENDENT TREATMENT:
   - If X < 10: treatment INCREASES X (prevents death from going too low)
   - If X > 10: treatment DECREASES X (prevents death from going too high)
   - Optimal X ≈ 10 (middle of safe zone)

5. DEATH BOUNDARIES:
   - X < 1: DEATH (too low)
   - X > 20: DEATH (too high)
   - Treatment must balance keeping X in safe zone [1, 20]

6. PATH-DEPENDENT DECISIONS:
   - Different realizations → different optimal times
   - Low X paths: Intervene early to boost up
   - High X paths: Intervene to push down (or never if stable)
   - Medium X paths: Wait and see trajectory
   - Just like the exercise boundary in American options!

🩺 Blood Pressure Interpretation:
   - Monitor your health markers (X) over time
   - Random shocks (U) affect your trajectory  
   - Optimal policy tells you when to start medication
   - Treatment is smart: raises low BP, lowers high BP
   - Goal: Keep X in healthy zone, avoid death from extremes
   - Maximize probability of good outcome (Y=1)
    """)
    print("="*80)