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
        
        X_t = f(X_{t-1}, U_{t-1}, U_t, intervention_now)
        
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
        
        # Base transition: X_t = X_{t-1} + U_{t-1} + U_t
        X_new = X_prev
        
        # Add effect of previous U (persistent effect)
        if U_prev is not None:
            X_new += U_prev
        
        # Add effect of current U
        X_new += U_current
        
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
        Structural equation with BOUNDED treatment effect
        
        For Algorithm 3 & 4: treatment effect can be uncertain
        effect_multiplier adjusts the treatment strength
        
        Parameters:
        -----------
        effect_multiplier : float
            Multiplier for treatment effect (e.g., 0.8 or 1.2 for ±20% uncertainty)
        """
        optimal_X = 10
        
        # Base transition: X_t = X_{t-1} + U_{t-1} + U_t
        X_new = float(X_prev)
        
        if U_prev is not None:
            X_new += U_prev
        X_new += U_current
        
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
            print(f"SIMULATING {num_paths} CAUSAL PATHS (CONTINUOUS TIME)")
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

    # ========================================================================
    # NEW VISUALIZATION METHODS (Features 1, 2, 3)
    # ========================================================================
    
    def plot_intervention_boundary(self, model_type='standard', figsize=(12, 8), save_path=None):
        """
        FEATURE 1: Intervention Boundary Visualization
        
        Creates a heatmap showing optimal stopping boundary over time and state space.
        Red = INTERVENE, Blue = WAIT, Yellow = AMBIGUOUS (for robust only)
        
        This is THE MOST IMPACTFUL visualization - shows "act now" vs "wait and see" trade-off!
        
        Parameters:
        -----------
        model_type : str
            'standard' for Algorithm 2, 'robust' for Algorithm 4
        figsize : tuple
            Figure size
        save_path : str or None
            If provided, saves figure to this path
        """
        # Sample time points for visualization (don't plot all if dt is small)
        if len(self.time_grid) > 30:
            step = max(1, len(self.time_grid) // 20)
            time_indices = list(range(0, len(self.time_grid), step))
        else:
            time_indices = list(range(len(self.time_grid)))
        
        X_values = list(range(self.X_min, self.X_max + 1))
        
        # Create policy matrix
        policy_matrix = np.zeros((len(X_values), len(time_indices)))
        
        if model_type == 'standard':
            # Use standard policy
            for i, X in enumerate(X_values):
                for j, t_idx in enumerate(time_indices):
                    policy = self.policy.get((t_idx, X, 0), 'wait')
                    if policy == 'INTERVENE':
                        policy_matrix[i, j] = 1  # Red
                    else:
                        policy_matrix[i, j] = 0  # Blue
            
            cmap = plt.cm.colors.ListedColormap(['#3498db', '#e74c3c'])  # Blue, Red
            labels = ['WAIT', 'INTERVENE']
            
        else:  # robust
            # Use robust policy with 3 colors
            for i, X in enumerate(X_values):
                for j, t_idx in enumerate(time_indices):
                    policy = self.robust_policy.get((t_idx, X, 0), 'wait')
                    if policy == 'INTERVENE':
                        policy_matrix[i, j] = 2  # Red
                    elif policy == 'ambiguous':
                        policy_matrix[i, j] = 1  # Yellow
                    else:
                        policy_matrix[i, j] = 0  # Blue
            
            cmap = plt.cm.colors.ListedColormap(['#3498db', '#f39c12', '#e74c3c'])  # Blue, Yellow, Red
            labels = ['WAIT', 'AMBIGUOUS', 'INTERVENE']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(policy_matrix, aspect='auto', origin='lower', cmap=cmap, 
                      extent=[0, len(time_indices)-1, self.X_min, self.X_max],
                      interpolation='nearest')
        
        # Set ticks
        time_labels = [f"{self.time_grid[t_idx]:.1f}" for t_idx in time_indices]
        ax.set_xticks(range(len(time_indices)))
        ax.set_xticklabels(time_labels, rotation=45)
        ax.set_xlabel('Time t', fontsize=14, fontweight='bold')
        ax.set_ylabel('Health Marker X', fontsize=14, fontweight='bold')
        
        # Title
        if model_type == 'standard':
            title = 'Optimal Intervention Boundary (Algorithm 2: Standard)'
        else:
            title = 'Robust Intervention Boundary (Algorithm 4: Robust)'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2] if model_type == 'robust' else [0, 1])
        cbar.set_ticklabels(labels)
        cbar.ax.tick_params(labelsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add text annotation
        textstr = f'Red region: Intervene immediately\n'
        textstr += f'Blue region: Wait (intervention not optimal yet)\n'
        if model_type == 'robust':
            textstr += f'Yellow region: Ambiguous (need more information)'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved intervention boundary plot to {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_causal_dag(self, figsize=(14, 6), save_path=None):
        """
        FEATURE 2: Causal DAG Visualization
        
        Draws the causal structure showing:
        - U shocks affecting X states over time
        - Intervention A cutting edges (do-operator)
        - Markov chain structure
        - Connection to final outcome Y
        
        This is PURE PEARL - emphasizes graphical models and do-calculus!
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str or None
            If provided, saves figure to this path
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 4)
        ax.axis('off')
        
        # Title
        ax.text(4, 3.7, 'Causal DAG: Optimal Intervention Timing', 
                ha='center', fontsize=18, fontweight='bold')
        
        # Node positions
        # U nodes (top row)
        u_y = 3.0
        u_positions = [(i * 1.0 + 0.5, u_y) for i in range(7)]
        
        # X nodes (middle row)
        x_y = 1.8
        x_positions = [(i * 1.0 + 0.5, x_y) for i in range(7)]
        
        # A node (intervention)
        a_pos = (3.5, 0.8)
        
        # Y node (outcome)
        y_pos = (6.5, 1.8)
        
        # Draw U nodes
        for i, (x, y) in enumerate(u_positions):
            circle = plt.Circle((x, y), 0.15, color='#95a5a6', ec='black', linewidth=2, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, f'U{i}', ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw X nodes
        for i, (x, y) in enumerate(x_positions):
            circle = plt.Circle((x, y), 0.18, color='#3498db', ec='black', linewidth=2, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, f'X{i}', ha='center', va='center', fontsize=10, 
                   fontweight='bold', color='white')
        
        # Draw A node (intervention)
        rect = plt.Rectangle((a_pos[0]-0.2, a_pos[1]-0.15), 0.4, 0.3, 
                            color='#e74c3c', ec='black', linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(a_pos[0], a_pos[1], 'A', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        
        # Draw Y node (outcome)
        circle = plt.Circle(y_pos, 0.2, color='#27ae60', ec='black', linewidth=2.5, zorder=3)
        ax.add_patch(circle)
        ax.text(y_pos[0], y_pos[1], 'Y', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        
        # Draw arrows
        arrow_props = dict(arrowstyle='->', lw=1.5, color='black', zorder=1)
        
        # U → U (horizontal persistence)
        for i in range(6):
            ax.annotate('', xy=u_positions[i+1], xytext=u_positions[i],
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='#7f8c8d', alpha=0.6, zorder=1))
        
        # U → X (vertical shocks)
        for i in range(7):
            ax.annotate('', xy=x_positions[i], xytext=u_positions[i],
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='#95a5a6', zorder=1))
        
        # X → X (state evolution)
        for i in range(6):
            ax.annotate('', xy=x_positions[i+1], xytext=x_positions[i],
                       arrowprops=dict(arrowstyle='->', lw=2.5, color='#2980b9', zorder=2))
        
        # A → X (intervention effects - shown as cutting edge with special style)
        # Draw dashed lines from A to X3, X4, X5 to show potential intervention
        for i in [3, 4, 5]:
            ax.plot([a_pos[0], x_positions[i][0]], [a_pos[1], x_positions[i][1]], 
                   'r--', lw=2, alpha=0.7, zorder=1)
            # Add arrow at end
            ax.annotate('', xy=x_positions[i], xytext=(a_pos[0], a_pos[1]+0.3),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c', 
                                     linestyle='--', alpha=0.7, zorder=2))
        
        # X6 → Y
        ax.annotate('', xy=y_pos, xytext=x_positions[6],
                   arrowprops=dict(arrowstyle='->', lw=3, color='#27ae60', zorder=2))
        
        # Add legend/explanation
        legend_x = 0.3
        legend_y = 0.5
        
        ax.text(legend_x, legend_y, 'Causal Structure:', fontsize=11, fontweight='bold')
        ax.text(legend_x, legend_y - 0.15, '• U: Exogenous shocks (random)', fontsize=9)
        ax.text(legend_x, legend_y - 0.3, '• X: Health markers (state)', fontsize=9)
        ax.text(legend_x, legend_y - 0.45, '• A: Intervention (one-time)', fontsize=9)
        ax.text(legend_x, legend_y - 0.6, '• Y: Binary outcome', fontsize=9)
        
        # Add do-calculus note
        ax.text(5.5, 0.3, 'do(A=1 at τ*) cuts incoming edges\nand forces X toward target', 
               fontsize=9, ha='center', style='italic',
               bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved causal DAG to {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_standard_vs_robust_comparison(self, X0=10, figsize=(15, 5), save_path=None):
        """
        FEATURE 3: Comparison Plot - Standard vs Robust Policies
        
        Side-by-side comparison showing:
        - Left: Standard policy (Algorithm 2)
        - Right: Robust policy (Algorithm 4)
        - Highlights differences in intervention decisions under uncertainty
        
        Parameters:
        -----------
        X0 : int
            Initial state for comparison
        figsize : tuple
            Figure size
        save_path : str or None
            If provided, saves figure to this path
        """
        # Sample time points
        if len(self.time_grid) > 30:
            step = max(1, len(self.time_grid) // 20)
            time_indices = list(range(0, len(self.time_grid), step))
        else:
            time_indices = list(range(len(self.time_grid)))
        
        X_values = list(range(self.X_min, self.X_max + 1))
        
        # Create matrices for both policies
        standard_matrix = np.zeros((len(X_values), len(time_indices)))
        robust_matrix = np.zeros((len(X_values), len(time_indices)))
        
        for i, X in enumerate(X_values):
            for j, t_idx in enumerate(time_indices):
                # Standard policy
                std_policy = self.policy.get((t_idx, X, 0), 'wait')
                if std_policy == 'INTERVENE':
                    standard_matrix[i, j] = 1
                else:
                    standard_matrix[i, j] = 0
                
                # Robust policy
                rob_policy = self.robust_policy.get((t_idx, X, 0), 'wait')
                if rob_policy == 'INTERVENE':
                    robust_matrix[i, j] = 2
                elif rob_policy == 'ambiguous':
                    robust_matrix[i, j] = 1
                else:
                    robust_matrix[i, j] = 0
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Standard
        cmap_std = plt.cm.colors.ListedColormap(['#3498db', '#e74c3c'])
        im1 = ax1.imshow(standard_matrix, aspect='auto', origin='lower', cmap=cmap_std,
                        extent=[0, len(time_indices)-1, self.X_min, self.X_max],
                        interpolation='nearest')
        
        time_labels = [f"{self.time_grid[t_idx]:.1f}" for t_idx in time_indices]
        ax1.set_xticks(range(len(time_indices)))
        ax1.set_xticklabels(time_labels, rotation=45)
        ax1.set_xlabel('Time t', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Health Marker X', fontsize=12, fontweight='bold')
        ax1.set_title('Standard Policy (Algorithm 2)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1])
        cbar1.set_ticklabels(['WAIT', 'INTERVENE'])
        
        # Plot 2: Robust
        cmap_rob = plt.cm.colors.ListedColormap(['#3498db', '#f39c12', '#e74c3c'])
        im2 = ax2.imshow(robust_matrix, aspect='auto', origin='lower', cmap=cmap_rob,
                        extent=[0, len(time_indices)-1, self.X_min, self.X_max],
                        interpolation='nearest')
        
        ax2.set_xticks(range(len(time_indices)))
        ax2.set_xticklabels(time_labels, rotation=45)
        ax2.set_xlabel('Time t', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Health Marker X', fontsize=12, fontweight='bold')
        ax2.set_title('Robust Policy (Algorithm 4)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
        cbar2.set_ticklabels(['WAIT', 'AMBIGUOUS', 'INTERVENE'])
        
        # Main title
        fig.suptitle(f'Policy Comparison: Standard vs Robust (X₀={X0})', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Add summary statistics
        std_intervene = np.sum(standard_matrix == 1)
        rob_intervene = np.sum(robust_matrix == 2)
        rob_ambiguous = np.sum(robust_matrix == 1)
        total_states = standard_matrix.size
        
        summary_text = f'Standard: {std_intervene}/{total_states} intervene ({std_intervene/total_states*100:.1f}%)\n'
        summary_text += f'Robust: {rob_intervene}/{total_states} intervene ({rob_intervene/total_states*100:.1f}%), '
        summary_text += f'{rob_ambiguous}/{total_states} ambiguous ({rob_ambiguous/total_states*100:.1f}%)'
        
        fig.text(0.5, -0.05, summary_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        plt.show()
        
        return fig


# ============================================================================
# DEMONSTRATION: Create all 3 visualizations
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CREATING 3 KEY VISUALIZATIONS")
    print("="*80)
    
    # Build both models
    print("\n[1/3] Building Standard Model (Algorithm 2)...")
    model_standard = CausalOptimalStopping(
        T=6, dt=1.0, treatment_effect=4,
        uncertainty_mode='none'
    )
    model_standard.solve_backward_induction(X0=10, verbose=False)
    
    print("[2/3] Building Robust Model (Algorithm 4)...")
    model_robust = CausalOptimalStopping(
        T=6, dt=1.0, treatment_effect=4,
        uncertainty_mode='robust',
        prob_uncertainty=0.15,
        effect_uncertainty=0.25
    )
    model_robust.solve_robust_backward_induction(X0=10, verbose=False)
    
    print("[3/3] Generating visualizations...\n")
    
    # Visualization 1: Intervention Boundary (Standard)
    print("Visualization 1: Intervention Boundary (Standard Policy)")
    model_standard.plot_intervention_boundary(model_type='standard')
    
    # Visualization 2: Causal DAG
    print("\nVisualization 2: Causal DAG")
    model_standard.plot_causal_dag()
    
    # Visualization 3: Standard vs Robust Comparison
    print("\nVisualization 3: Standard vs Robust Comparison")
    model_robust.plot_standard_vs_robust_comparison(X0=10)
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)