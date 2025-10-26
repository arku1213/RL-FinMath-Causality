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
                 T=6,                 # Time periods (matches n_exogenous)
                 X_min=1,             # Min value for X variables
                 X_max=20,            # Max value for X variables
                 U_min=-3,            # Min value for U shocks
                 U_max=3,             # Max value for U shocks
                 treatment_effect=4,  # Boost from treatment
                 Y_threshold=10,      # Threshold for Y=1 outcome
                 discount=0.95):      # Discount factor for future values
        
        self.n_endogenous = n_endogenous
        self.n_exogenous = n_exogenous
        self.T = T
        self.X_min = X_min
        self.X_max = X_max
        self.U_min = U_min
        self.U_max = U_max
        self.treatment_effect = treatment_effect
        self.Y_threshold = Y_threshold
        self.discount = discount
        
        # U values and their probabilities
        self.U_values = list(range(U_min, U_max + 1))
        self.U_probs = self._compute_U_probabilities()
        
        # Storage for value functions and policies
        self.value_function = {}  # V(t, X_state, A)
        self.policy = {}          # Optimal action at each state
        
        print(f"Initialized Causal Optimal Stopping Problem")
        print(f"  Endogenous variables: X₀ to X₆ ({n_endogenous} total)")
        print(f"  Exogenous shocks: U₁ to U₆ ({n_exogenous} total)")
        print(f"  Time periods: T = {T}")
        print(f"  X range: [{X_min}, {X_max}]")
        print(f"  U range: [{U_min}, {U_max}]")
        print(f"  Treatment effect: ±{treatment_effect} (state-dependent)")
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
        
        State space: (t, X, already_intervened)
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
            print(f"Stage {self.T}: Computing terminal values E[Y | X₆]...")
        
        for X_final in range(self.X_min, self.X_max + 1):
            # At final stage, no intervention decision to make
            # Value is just E[Y | X₆]
            expected_Y = self.get_expected_outcome(X_final)
            
            # Store for both already_intervened states
            self.value_function[(self.T, X_final, 0)] = expected_Y  # Unused intervention
            self.value_function[(self.T, X_final, 1)] = expected_Y  # Already used
        
        # Backward induction: Work from T-1 back to 0
        for t in range(self.T - 1, -1, -1):
            if verbose:
                print(f"\nStage {t}: Computing optimal policies...")
            
            intervene_count = 0
            wait_count = 0
            
            # For each possible X_t value
            for X_t in range(self.X_min, self.X_max + 1):
                
                # Case 1: Already intervened (already_intervened=1)
                # No decision to make - just compute continuation value without intervention
                continuation_used = self._compute_continuation_value(
                    t, X_t, already_intervened=True
                )
                self.value_function[(t, X_t, 1)] = continuation_used
                self.policy[(t, X_t, 1)] = 'no_action'  # Already used intervention
                
                # Case 2: Haven't intervened yet (already_intervened=0)
                # OPTIMAL STOPPING DECISION: Intervene now vs. Wait
                
                # Option A: Intervene NOW (use the one-time intervention)
                value_intervene = self._compute_intervention_value(t, X_t)
                
                # Option B: Wait (keep intervention available for later)
                value_wait = self._compute_continuation_value(
                    t, X_t, already_intervened=False
                )
                
                # BELLMAN EQUATION: Take the maximum!
                if value_intervene > value_wait:
                    self.value_function[(t, X_t, 0)] = value_intervene
                    self.policy[(t, X_t, 0)] = 'INTERVENE'
                    intervene_count += 1
                else:
                    self.value_function[(t, X_t, 0)] = value_wait
                    self.policy[(t, X_t, 0)] = 'wait'
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
    
    def simulate_path(self, X0=10, seed=None):
        """
        Simulate a single path of X values with random U shocks
        NO intervention applied during simulation - this is the natural path
        
        Returns:
        --------
        X_path : list
            Sequence of X values [X₀, X₁, ..., X₆]
        U_path : list
            Sequence of U values [U₁, U₂, ..., U₆]
        """
        if seed is not None:
            np.random.seed(seed)
        
        X_path = [X0]
        U_path = []
        
        for t in range(1, self.n_endogenous):
            # Sample U_t
            U_t = np.random.choice(self.U_values, p=self.U_probs)
            U_path.append(U_t)
            
            # Compute X_t WITHOUT intervention (natural evolution)
            U_prev = U_path[-2] if len(U_path) >= 2 else None
            X_t = self.structural_equation(X_path[-1], U_t, U_prev, intervention_now=False, t=t)
            
            # Handle death
            if X_t == "DEATH":
                X_path.append("DEATH")
                break
            
            X_path.append(X_t)
        
        return X_path, U_path
    
    def find_optimal_intervention_time(self, X_path, U_path):
        """
        Given a realized path, find the optimal intervention time τ*
        
        ONE-TIME INTERVENTION: Can only intervene once!
        
        This is analogous to finding when to exercise the American put!
        
        Returns:
        --------
        tau_star : int or None
            Optimal intervention time (None if never optimal or path dies)
        intervention_info : dict
            Details about the intervention decision
        """
        # Start with intervention available (already_intervened=0)
        already_intervened = False
        
        # Walk through the path
        for t in range(len(X_path) - 1):  # Don't include final X₆
            X_t = X_path[t]
            
            # Check if we died
            if X_t == "DEATH":
                return None, {
                    'time': None,
                    'died_at': t,
                    'intervened': False,
                    'reason': 'died_before_decision'
                }
            
            # Check policy at this state
            already_intervened_int = 1 if already_intervened else 0
            action = self.policy.get((t, X_t, already_intervened_int), 'wait')
            
            if action == 'INTERVENE' and not already_intervened:
                # Found optimal intervention time!
                value_at_intervention = self.value_function.get((t, X_t, 0), 0)
                
                return t, {
                    'time': t,
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
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"SIMULATING {num_paths} CAUSAL PATHS")
            print(f"{'='*80}")
            print(f"Finding optimal ONE-TIME intervention time τ* for each path\n")
        
        results = []
        
        for path_num in range(num_paths):
            X_path, U_path = self.simulate_path(X0=X0, seed=path_num)
            tau_star, info = self.find_optimal_intervention_time(X_path, U_path)
            
            result = {
                'path_number': path_num + 1,
                'X_path': X_path,
                'U_path': U_path,
                'tau_star': tau_star,
                'info': info
            }
            results.append(result)
            
            if verbose:
                print(f"Path #{path_num + 1}:")
                
                # Handle death in trajectory
                if "DEATH" in X_path:
                    death_idx = X_path.index("DEATH")
                    print(f"  X trajectory: {X_path[:death_idx]} → DEATH at t={death_idx}")
                else:
                    print(f"  X trajectory: {X_path[:4]}... → {X_path[-1]}")
                
                print(f"  U shocks: {U_path[:3]}...")
                
                if tau_star is not None:
                    print(f"  ✓ Optimal Intervention Time τ*: t={tau_star}")
                    print(f"    X value at intervention: {info['X_value']}")
                    print(f"    Expected value: {info['value']:.3f}")
                else:
                    reason = info.get('reason', 'unknown')
                    if reason == 'died_before_decision' or reason == 'died_without_intervention':
                        print(f"  ☠ Path died without intervention")
                        print(f"    Died at: {info.get('died_at', 'unknown')}")
                    elif reason == 'never_optimal':
                        print(f"  ✗ Never intervened (not needed)")
                        print(f"    Final X: {info['X_final']}, Outcome Y: {info['Y']}")
                print()
        
        # Summary statistics
        intervention_times = [r['tau_star'] for r in results if r['tau_star'] is not None]
        deaths = sum(1 for r in results if 'died' in r['info'].get('reason', ''))
        never_needed = sum(1 for r in results if r['info'].get('reason') == 'never_optimal')
        
        print(f"{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"  Total Paths: {num_paths}")
        print(f"  Paths with Intervention: {len(intervention_times)} ({len(intervention_times)/num_paths*100:.1f}%)")
        print(f"  Paths without Intervention: {num_paths - len(intervention_times)}")
        print(f"    - Died without intervention: {deaths}")
        print(f"    - Never needed: {never_needed}")
        
        if intervention_times:
            print(f"\n  Intervention Times:")
            print(f"    Average τ*: {np.mean(intervention_times):.2f}")
            print(f"    Earliest: t={np.min(intervention_times)}")
            print(f"    Latest: t={np.max(intervention_times)}")
            
            # Distribution
            from collections import Counter
            time_dist = Counter(intervention_times)
            print(f"\n  Distribution of τ*:")
            for t in sorted(time_dist.keys()):
                count = time_dist[t]
                print(f"    t={t}: {count} paths ({count/len(intervention_times)*100:.1f}%)")
        
        print(f"{'='*80}\n")
        
        return results
    
    def print_policy_summary(self, X0=10):
        """
        Print the optimal policy for key states
        """
        print(f"\n{'='*80}")
        print("OPTIMAL INTERVENTION POLICY (Key States)")
        print(f"{'='*80}")
        print(f"Starting from X0 = {X0}")
        print("ONE-TIME INTERVENTION: Can only use once!\n")
        print(f"{'Time':<8} {'X Value':<12} {'No Intervention':<30} {'Value':<15}")
        print("-" * 75)
        
        for t in range(self.T):
            # Show policy for a range of X values
            for X in [3, 7, 10, 14, 17]:
                if X < self.X_min or X > self.X_max:
                    continue
                    
                policy_available = self.policy.get((t, X, 0), 'unknown')
                value_available = self.value_function.get((t, X, 0), 0)
                
                action_str = "🔴 INTERVENE NOW!" if policy_available == 'INTERVENE' else "⚪ Wait"
                
                print(f"t={t:<6} X={X:<10} {action_str:<30} {value_available:.4f}")
        
        print(f"{'='*80}\n")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CAUSAL OPTIMAL STOPPING: Finding Optimal Intervention Times")
    print("="*80)
    print("\n🩺 Blood Pressure Analogy:")
    print("  X = health markers (1-20, higher is better)")
    print("  U = random daily shocks (-3 to +3)")
    print("  A = medication (0=not taking, 1=taking)")
    print("  Y = outcome (1=healthy, 0=adverse event)")
    print("  Goal: Find optimal time τ* to start medication\n")
    
    # Initialize the model
    model = CausalOptimalStopping(
        n_endogenous=7,
        n_exogenous=6,
        T=6,
        X_min=1,
        X_max=20,
        U_min=-3,
        U_max=3,
        treatment_effect=4,    # Increased from 3 for more nuanced decisions
        Y_threshold=10,        # Not used anymore, kept for compatibility
        discount=0.95
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