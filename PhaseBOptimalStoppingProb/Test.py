import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    
    # ========================================================================
    # 3D SCATTER VISUALIZATION (ORIGINAL)
    # ========================================================================
    
    def plot_intervention_boundaries_3d(self, filename='intervention_boundaries_3d.png'):
        """
        Create 3D scatter plot of intervention boundaries in (t, X, U) space
        PLUS 2D slices showing cross-sections
        
        Shows which states lead to INTERVENE vs WAIT vs DEATH decisions
        """
        # ========================================================================
        # ORIGINAL 3D PLOT
        # ========================================================================
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Collect points for each policy type
        intervene_points = {'t': [], 'X': [], 'U': []}
        wait_points = {'t': [], 'X': [], 'U': []}
        death_points = {'t': [], 'X': [], 'U': []}
        
        # Loop through all states (only I=0, where decisions happen)
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
        
        # Plot each category
        if len(intervene_points['t']) > 0:
            ax.scatter(intervene_points['t'], intervene_points['X'], intervene_points['U'],
                    c='red', marker='o', s=50, alpha=0.6, label='INTERVENE')
        
        if len(wait_points['t']) > 0:
            ax.scatter(wait_points['t'], wait_points['X'], wait_points['U'],
                    c='blue', marker='o', s=30, alpha=0.3, label='WAIT')
        
        if len(death_points['t']) > 0:
            ax.scatter(death_points['t'], death_points['X'], death_points['U'],
                    c='black', marker='x', s=40, alpha=0.8, label='DEATH')
        
        # Labels and title
        ax.set_xlabel('Time (t)', fontsize=14, labelpad=10)
        ax.set_ylabel('Health State (X)', fontsize=14, labelpad=10)
        ax.set_zlabel('Shock (U)', fontsize=14, labelpad=10)
        ax.set_title('Intervention Boundaries in (t, X, U) Space', fontsize=16, pad=20)
        
        # Set axis limits
        ax.set_xlim(0.5, self.T - 0.5)
        ax.set_ylim(self.X_min - 0.5, self.X_max + 0.5)
        ax.set_zlim(min(self.U_values) - 0.5, max(self.U_values) + 0.5)
        ax.set_yticks(range(0, self.X_max + 1, 2))
        
        # Legend
        ax.legend(loc='upper left', fontsize=12)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Save 3D plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================================================
        # NEW: 2D SLICES
        # ========================================================================
        
        # Create figure with subplots for each time slice
        n_times = self.T - 1  # Number of decision times
        fig, axes = plt.subplots(1, n_times, figsize=(5*n_times, 5))
        
        # Handle case where T=2 (only one subplot)
        if n_times == 1:
            axes = [axes]
        
        for idx, t in enumerate(range(1, self.T)):
            ax = axes[idx]
            
            # Collect data for this time slice
            for X in range(self.X_min, self.X_max + 1):
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), '?')
                    
                    if policy == 'INTERVENE':
                        ax.scatter(X, U, c='red', marker='o', s=100, alpha=0.7)
                    elif policy == 'WAIT':
                        ax.scatter(X, U, c='blue', marker='o', s=60, alpha=0.4)
                    elif policy == 'no_action':
                        ax.scatter(X, U, c='black', marker='x', s=80, alpha=0.9)
            
            # Format subplot
            ax.set_xlabel('Health State (X)', fontsize=12)
            ax.set_ylabel('Shock (U)', fontsize=12)
            ax.set_title(f't = {t}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.X_min - 0.5, self.X_max + 0.5)
            ax.set_ylim(min(self.U_values) - 0.5, max(self.U_values) + 0.5)
        
        # Add legend to the figure
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='INTERVENE'),
            Patch(facecolor='blue', alpha=0.4, label='WAIT'),
            Patch(facecolor='black', alpha=0.9, label='DEATH')
        ]
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig.legend(handles=legend_elements, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=12)
        
        slice_filename = filename.replace('.png', '_slices.png')
        plt.savefig(slice_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename, slice_filename
    
    def plot_intervention_boundaries_3d_with_slices(self, filename='intervention_boundaries_3d_contour.png'):

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # For each time slice, create a contour/filled plot
        for t in range(1, self.T):
            # Create grid for this time slice
            X_grid = np.arange(self.X_min, self.X_max + 1)
            U_grid = np.array(self.U_values)
            
            # Create meshgrid
            X_mesh, U_mesh = np.meshgrid(X_grid, U_grid)
            
            # Create policy values: 2=INTERVENE, 1=WAIT, 0=DEATH
            policy_values = np.zeros_like(X_mesh, dtype=float)
            
            for i, U in enumerate(U_grid):
                for j, X in enumerate(X_grid):
                    policy = self.policy.get((t, X, U, 0), '?')
                    if policy == 'INTERVENE':
                        policy_values[i, j] = 2
                    elif policy == 'WAIT':
                        policy_values[i, j] = 1
                    elif policy == 'no_action':
                        policy_values[i, j] = 0
            
            # Plot the contour slice at position t - DISCRETE SQUARES VERSION
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            for i in range(len(U_grid)):
                for j in range(len(X_grid)):
                    val = policy_values[i, j]
                    if val == 2:  # INTERVENE
                        color = 'red'
                        alpha = 0.7
                    elif val == 1:  # WAIT
                        color = 'blue'
                        alpha = 0.3
                    else:  # DEATH
                        color = 'black'
                        alpha = 0.8
                    
                    # Draw a small square at this position
                    x = [X_grid[j] - 0.4, X_grid[j] + 0.4, X_grid[j] + 0.4, X_grid[j] - 0.4]
                    y = [t, t, t, t]
                    z = [U_grid[i] - 0.4, U_grid[i] - 0.4, U_grid[i] + 0.4, U_grid[i] + 0.4]
                    
                    verts = [list(zip(y, x, z))]
                    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none')
                    ax.add_collection3d(poly)
            
            # Draw outline of slice
            outline_x = [self.X_min, self.X_max, self.X_max, self.X_min, self.X_min]
            outline_t = [t, t, t, t, t]
            outline_u = [min(self.U_values), min(self.U_values), max(self.U_values), 
                        max(self.U_values), min(self.U_values)]
            ax.plot(outline_t, outline_x, outline_u, 'k-', linewidth=1.5, alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('Time (t)', fontsize=14, labelpad=10)
        ax.set_ylabel('Health State (X)', fontsize=14, labelpad=10)
        ax.set_zlabel('Shock (U)', fontsize=14, labelpad=10)
        ax.set_title('Intervention Policy Slices in (t, X, U, I) Space', fontsize=16, pad=20)
        
        # Set limits
        ax.set_xlim(0.5, self.T - 0.5)
        ax.set_ylim(self.X_min - 0.5, self.X_max + 0.5)
        ax.set_zlim(min(self.U_values) - 0.5, max(self.U_values) + 0.5)
        ax.set_yticks(range(2, self.X_max + 1, 2))
        
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='INTERVENE'),
            Patch(facecolor='blue', alpha=0.3, label='WAIT'),
            Patch(facecolor='black', alpha=0.8, label='DEATH')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
        
        # Set initial viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Save
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_intervention_boundaries_plotly(self, filename='intervention_boundaries_3d_interactive.html'):
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return None
        
        fig = go.Figure()
        
        # Collect all points by policy type
        policy_data = {
            'INTERVENE': {'t': [], 'X': [], 'U': []},
            'WAIT': {'t': [], 'X': [], 'U': []},
            'DEATH': {'t': [], 'X': [], 'U': []}
        }
        
        # Loop through all states and collect points
        for t in range(1, self.T):
            for X in range(self.X_min, self.X_max + 1):
                for U in self.U_values:
                    policy = self.policy.get((t, X, U, 0), '?')
                    
                    if policy == 'INTERVENE':
                        policy_data['INTERVENE']['t'].append(t)
                        policy_data['INTERVENE']['X'].append(X)
                        policy_data['INTERVENE']['U'].append(U)
                    elif policy == 'WAIT':
                        policy_data['WAIT']['t'].append(t)
                        policy_data['WAIT']['X'].append(X)
                        policy_data['WAIT']['U'].append(U)
                    elif policy == 'no_action':
                        policy_data['DEATH']['t'].append(t)
                        policy_data['DEATH']['X'].append(X)
                        policy_data['DEATH']['U'].append(U)
        
        # Add traces for each policy type
        colors = {'INTERVENE': 'red', 'WAIT': 'blue', 'DEATH': 'black'}
        sizes = {'INTERVENE': 8, 'WAIT': 6, 'DEATH': 8}
        symbols = {'INTERVENE': 'circle', 'WAIT': 'circle', 'DEATH': 'x'}
        opacities = {'INTERVENE': 0.8, 'WAIT': 0.5, 'DEATH': 0.9}
        
        for policy_name in ['INTERVENE', 'WAIT', 'DEATH']:
            data = policy_data[policy_name]
            
            if len(data['t']) > 0:
                fig.add_trace(go.Scatter3d(
                    x=data['t'],
                    y=data['X'],
                    z=data['U'],
                    mode='markers',
                    marker=dict(
                        size=sizes[policy_name],
                        color=colors[policy_name],
                        symbol=symbols[policy_name],
                        opacity=opacities[policy_name],
                        line=dict(width=0)
                    ),
                    name=policy_name,
                    hovertemplate=f'<b>{policy_name}</b><br>Time: %{{x}}<br>Health: %{{y}}<br>Shock: %{{z}}<extra></extra>'
                ))
        
        # Add frames for each time slice
        for t in range(1, self.T):
            outline_x = [self.X_min, self.X_max, self.X_max, self.X_min, self.X_min]
            outline_t = [t, t, t, t, t]
            outline_u = [min(self.U_values), min(self.U_values), max(self.U_values), 
                        max(self.U_values), min(self.U_values)]
            
            fig.add_trace(go.Scatter3d(
                x=outline_t,
                y=outline_x,
                z=outline_u,
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interactive Intervention Policy Slices in (t, X, U, I) Space',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            scene=dict(
                xaxis_title='Time (t)',
                yaxis_title='Health State (X)',
                zaxis_title='Shock (U)',
                xaxis=dict(range=[0.5, self.T - 0.5]),
                yaxis=dict(range=[self.X_min - 0.5, self.X_max + 0.5]),
                zaxis=dict(range=[min(self.U_values) - 0.5, max(self.U_values) + 0.5]),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1400,
            height=1000,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=14)
            )
        )
        
        # Save as HTML
        fig.write_html(filename)
        print(f"Interactive Plotly visualization saved to {filename}")
        print("Open this file in your web browser to interact with the 3D plot!")
        
        return filename
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    def print_results(self, output_file='RESULTS.txt'):
        """
        Print results - saves analysis to file
        
        Shows:
        1. Standard Optimal Stopping (all states) with optimal intervention targets
        2. Threshold Policy extracted from optimal policy (detailed by t and U)
        3. Algorithm 3: Optimal Intervention at Fixed Time k
        """
        
        # Solve Algorithm 1
        V_optimal = self.solve_standard_optimal_stopping()
        
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
        viz_filename, slice_filename = self.plot_intervention_boundaries_3d()
        contour_filename = self.plot_intervention_boundaries_3d_with_slices()
        plotly_filename = self.plot_intervention_boundaries_plotly()
        
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
                f.write(f"\nAt time t={t}:\n")
                for U in self.U_values:
                    intervene_states = thresholds[(t, U)]
                    if intervene_states:
                        states_str = '{' + ', '.join(map(str, intervene_states)) + '}'
                    else:
                        states_str = '∅'
                    f.write(f"  U={U:2d}: Intervene if X ∈ {states_str}\n")
            
            # ================================================================
            # OPTIMAL INTERVENTION AT FIXED TIME k
            # ================================================================
            f.write(f"\n\n{'='*80}\n")
            f.write("OPTIMAL INTERVENTION AT FIXED TIME k\n")
            f.write(f"{'='*80}\n\n")
            f.write("Everyone must intervene at time k, choosing optimal target based on state.\n\n")
            
            # Summary table
            f.write("COMPARISON:\n")
            f.write(f"{'─'*80}\n")
            for k in range(1, self.T):
                E_Y = alg3_results[k]
                f.write(f"  Optimal intervention at k={k}:  E[Y] = {E_Y:.4f}\n")
            
            # ================================================================
            # 3D VISUALIZATION
            # ================================================================
            f.write(f"\n\n{'='*80}\n")
            f.write("VISUALIZATIONS\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"3D scatter plot saved to: {viz_filename}\n")
            f.write(f"2D time slices saved to: {slice_filename}\n")
            f.write(f"3D contour slices saved to: {contour_filename}\n")
            if plotly_filename:
                f.write(f"Interactive 3D plot (Plotly) saved to: {plotly_filename}\n")
                f.write(f"  -> Open {plotly_filename} in your web browser for full 3D interaction!\n")
            
            f.write(f"\n{'='*80}\n")
        
        # Console output
        print(f"Results saved to {output_file}")
        print(f"3D visualization saved to {viz_filename}")
        print(f"2D time slices saved to {slice_filename}")
        print(f"3D contour slices saved to {contour_filename}")
        if plotly_filename:
            print(f"Interactive Plotly visualization saved to {plotly_filename}")


# Run algorithms
if __name__ == "__main__":
    model = CausalOptimalStopping(X0=10)
    
    # Solve optimal policies and generate output files
    output_file = 'RESULTS.txt'
    model.print_results(output_file)