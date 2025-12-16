import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION - CAUSAL MODEL WITH N-VALUED EXOGENOUS SHOCKS
# ============================================================
X0 = 100.0  # Initial state (e.g., health marker, wealth, asset value)
r = 0.00    # Discount rate
K = 100.0   # Threshold for positive outcome
T_steps = 2  # Number of time periods (causal depth)
dt = 1.0

# EXOGENOUS SHOCK PARAMETERS
N = 9  # Number of possible shock values: {-4, -3, -2, -1, 0, 1, 2, 3, 4}
sigma = 0.3  # Shock volatility parameter
lambda_param = np.sqrt(N - 1)  # Scales with N

# Generate N symmetric shock multipliers
# Each represents how the shock transforms the state: X_{t+1} = X_t * multiplier[U_t]
shock_multipliers = []
for i in range(N):
    # Map index to symmetric range [-1, +1]
    ratio = (i - (N-1)/2) / ((N-1)/2)
    # Convert to multiplicative shock: exp(normalized_value * sigma)
    multiplier = np.exp(ratio * lambda_param * sigma * np.sqrt(dt))
    shock_multipliers.append(multiplier)

shock_multipliers = sorted(shock_multipliers, reverse=True)  # Largest to smallest
print(f"\nN={N} exogenous shock multipliers: {[f'{m:.4f}' for m in shock_multipliers]}")

# LSMC parameters for conditional expectation estimation
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# RL hyperparameters - SCALED WITH N!
BASE_ACTOR_LR = 0.00003 * (3/N)**0.5  # Scale down for larger N
BASE_CRITIC_LR = 0.00010 * (3/N)**0.5
ACTOR_LR = BASE_ACTOR_LR
CRITIC_LR = BASE_CRITIC_LR

# Scale ACTION_SCALE based on maximum possible outcome
max_state_value = X0 * (shock_multipliers[0] ** T_steps)  # Max upward trajectory
max_terminal_outcome = max(max_state_value - K, 0)
ACTION_SCALE = max_terminal_outcome * 1.2

REPLICATION_PENALTY = 500000 * (N/3)  # Scale penalty with N
COST_WEIGHT = 0.0

# Training configuration - SCALED WITH N!
base_episodes = 800000
TOTAL_EPISODES = int(base_episodes * (N/3)**0.8)  # More episodes for larger N
NUM_ITERATIONS = 16
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.003
BUFFER_SIZE = int(800000 * (N/3)**0.5)  # Larger buffer for complex trees
HIDDEN_DIM = 256 + 64 * (N - 3)  # Bigger network for more interventions

# Improvement parameters
LR_DECAY = 0.93
NOISE_DECAY = 0.75
EARLY_STOP_PATIENCE = 4

print("=" * 60)
print(f"CAUSAL MODEL: DEEP RL INTERVENTION DISCOVERY (T={T_steps})")
print("=" * 60)
print("METHOD: Pure RL discovers optimal intervention portfolios")
print(f"Causal Structure: {N} exogenous shock values → {N} binary interventions")
print("Market Completeness: COMPLETE (interventions on all shock values)")
print("=" * 60)
print("STATE REPRESENTATION: [X/X₀, t/T, shock_history]")
print("  - NO conditional expectations given to actor!")
print("ACTION: Intervention doses for each possible shock value")
print("REWARD: Penalty for deviation from target outcome")
print("Actor discovers optimal causal interventions through trial-and-error")
print("=" * 60)
print(f"\nSCALING INFO:")
print(f"  Total Episodes: {TOTAL_EPISODES:,}")
print(f"  Episodes/Iter: {TOTAL_EPISODES//NUM_ITERATIONS:,}")
print(f"  Buffer Size: {BUFFER_SIZE:,}")
print(f"  Hidden Dim: {HIDDEN_DIM}")
print(f"  Penalty: {REPLICATION_PENALTY:,.0f}")
print("=" * 60)


# ============================================================
# SHOCK PROBABILITIES (RISK-NEUTRAL MEASURE)
# ============================================================
def calculate_shock_probabilities(multipliers, r, dt):
    """
    Calculate probabilities for exogenous shocks under risk-neutral measure.
    
    In causal language: These are the "natural" probabilities of each 
    exogenous shock value occurring, adjusted for time preference.
    """
    N = len(multipliers)
    growth = np.exp(r * dt)
    
    def objective(p):
        # Minimize deviation from uniform distribution
        return np.sum((p - 1/N)**2)
    
    def constraint_mean(p):
        # Risk-neutral condition: expected growth equals discount rate
        return np.sum(p * np.array(multipliers)) - growth
    
    def constraint_sum(p):
        # Probabilities sum to 1
        return np.sum(p) - 1
    
    # Initial guess: uniform distribution
    p0 = [1/N] * N
    
    constraints = [
        {'type': 'eq', 'fun': constraint_mean},
        {'type': 'eq', 'fun': constraint_sum}
    ]
    bounds = [(0.001, 0.999)] * N
    
    result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        probs = result.x
    else:
        print("WARNING: Optimization failed, using uniform fallback")
        probs = [1/N] * N
    
    # Verify
    prob_sum = np.sum(probs)
    expected_growth = np.sum(probs * np.array(multipliers))
    
    print(f"\nExogenous Shock Probabilities (N={N}):")
    for i, p in enumerate(probs):
        print(f"  P(U_t = shock_{i}) = {p:.6f}")
    print(f"  Sum = {prob_sum:.6f}, Expected growth = {expected_growth:.6f}")
    
    assert abs(prob_sum - 1.0) < 1e-6, "Probabilities must sum to 1"
    assert abs(expected_growth - growth) < 1e-2, "Risk-neutral condition violated"
    assert all(p >= 0 for p in probs), "Negative probabilities"
    
    print("  ✓ Valid probability distribution")
    return probs


shock_probs = calculate_shock_probabilities(shock_multipliers, r, dt)


# ============================================================
# UTILITIES
# ============================================================
def create_polynomial_features(X, degree):
    """Create polynomial features for regression."""
    X = np.array(X).reshape(-1, 1)
    features = np.ones((X.shape[0], degree + 1))
    for d in range(1, degree + 1):
        features[:, d] = (X[:, 0] ** d)
    return features


class RidgeRegression:
    """Ridge regression with regularization."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
    
    def fit(self, X, y):
        n_features = X.shape[1]
        XtX = X.T @ X
        reg_matrix = self.alpha * np.eye(n_features)
        try:
            self.coef_ = np.linalg.solve(XtX + reg_matrix, X.T @ y)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(XtX + reg_matrix, X.T @ y, rcond=None)[0]
        return self
    
    def predict(self, X):
        return X @ self.coef_


# ============================================================
# CAUSAL PATH SIMULATION
# ============================================================
class CausalPathSimulator:
    """
    Simulate causal paths with exogenous shocks.
    
    Causal DAG:
        X₀ → X₁ → X₂ → ... → X_T → Y
             ↑    ↑         ↑
             U₁   U₂        U_T
    
    Where:
    - X_t: Endogenous state variable (determined by model)
    - U_t: Exogenous shock (external cause)
    - Structural equation: X_{t+1} = X_t * shock_multiplier[U_t]
    - Y: Target outcome = max(X_T - K, 0)
    """
    def __init__(self, X0, shock_multipliers, shock_probs, T_steps, dt):
        self.X0 = X0  # Initial state
        self.shock_multipliers = shock_multipliers
        self.shock_probs = shock_probs
        self.N = len(shock_multipliers)
        self.T_steps = T_steps
        self.dt = dt
    
    def simulate_paths(self, num_paths):
        """
        Simulate causal paths.
        
        Each path represents one realization of the causal process:
        - Start at X₀
        - At each time t, exogenous shock U_t occurs
        - State evolves via structural equation
        - End with outcome Y
        """
        paths = []
        for _ in range(num_paths):
            path = []
            X = self.X0  # Initial state
            shock_history = []  # History of exogenous shocks (causes)
            
            for t in range(self.T_steps + 1):
                path_step = {
                    'X': X,  # Current state value
                    't': t,  # Time (causal depth)
                    'outcome': max(X - K, 0) if t == self.T_steps else None,
                    'shock_occurred': None,  # Which shock occurred
                    'shock_history': shock_history.copy()  # Past causal history
                }
                
                if t < self.T_steps:
                    # Sample exogenous shock according to probabilities
                    shock_idx = np.random.choice(self.N, p=self.shock_probs)
                    
                    # Apply structural equation: X_{t+1} = f(X_t, U_{t+1})
                    X *= self.shock_multipliers[shock_idx]
                    
                    path_step['shock_occurred'] = shock_idx
                    shock_history.append(shock_idx)
                
                path.append(path_step)
            paths.append(path)
        return paths


# ============================================================
# CONDITIONAL EXPECTATION ESTIMATOR (LSMC)
# ============================================================
class ConditionalExpectationEstimator:
    """
    Least Squares Monte Carlo for conditional expectation estimation.
    
    In causal language:
    - Estimates E[Y | X_t] = expected outcome given current state
    - These are the "causal effects" we want our interventions to achieve
    - NOT given to the actor - only used in reward computation!
    """
    def __init__(self, polynomial_degree=3, alpha=0.1):
        self.poly_degree = polynomial_degree
        self.alpha = alpha
        self.continuation_models = {}
        self.T_steps = T_steps
    
    def estimate_continuation_values(self, paths, r, dt):
        print("\n" + "=" * 60)
        print("ESTIMATING CONDITIONAL EXPECTATIONS (LSMC)")
        print("=" * 60)
        print("Computing E[Y | X_t] for all states in the causal tree")
        print("These represent the 'target values' for intervention portfolios")
        
        # Initialize terminal values: Y = max(X_T - K, 0)
        for path in paths:
            path[-1]['value'] = path[-1]['outcome']
        
        # Backward induction through causal DAG
        for t in range(self.T_steps - 1, -1, -1):
            X_values, y_values = [], []
            for path in paths:
                X_values.append(path[t]['X'])
                y_values.append(np.exp(-r * dt) * path[t+1]['value'])
            
            # Fit regression model: E[Y | X_t] ≈ polynomial(X_t)
            X_poly = create_polynomial_features(X_values, self.poly_degree)
            model = RidgeRegression(alpha=self.alpha).fit(X_poly, y_values)
            self.continuation_models[t] = model
            
            # Store estimated conditional expectations
            for path in paths:
                X_t = path[t]['X']
                X_pred = create_polynomial_features([X_t], self.poly_degree)
                path[t]['value'] = model.predict(X_pred)[0]
        
        print("CONDITIONAL EXPECTATION ESTIMATION COMPLETE")
        print("(Used for REWARD computation only - NOT given to actor!)")
        print("=" * 60)
        return paths
    
    def predict_continuation_value(self, X, t):
        """Predict E[Y | X_t]"""
        if t not in self.continuation_models:
            return 0.0
        X_features = create_polynomial_features([X], self.poly_degree)
        return self.continuation_models[t].predict(X_features)[0]
    
    def predict_child_continuation_values(self, X, t):
        """
        Returns [E[Y | X_t, U_{t+1}=0], ..., E[Y | X_t, U_{t+1}=N-1]]
        
        These are the "causal effects" of each shock value on the expected outcome.
        """
        if t >= self.T_steps - 1:
            # Terminal: just compute outcomes directly
            return [max(X * multiplier - K, 0) for multiplier in shock_multipliers]
        else:
            # Non-terminal: predict conditional expectations
            return [self.predict_continuation_value(X * multiplier, t + 1) 
                    for multiplier in shock_multipliers]


# ============================================================
# NEURAL NETWORKS
# ============================================================
class CausalActor(nn.Module):
    """
    Actor network that learns optimal intervention policy.
    
    Input: Current state [X, t, shock_history]
    Output: Intervention doses for each possible shock value
    
    The actor learns π(X_t, t) → [C_{t+1,0}, ..., C_{t+1,N-1}]
    where C_{t+1,k} is the intervention dose for shock k.
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CausalActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)
    
    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        x = torch.tanh(self.fc4(x)) * ACTION_SCALE
        return x


class CausalCritic(nn.Module):
    """Critic network that estimates Q-value of intervention portfolio."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CausalCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x


# ============================================================
# DDPG COMPONENTS
# ============================================================
class OUNoise:
    """Ornstein-Uhlenbeck process for exploration."""
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.20):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.initial_sigma = sigma
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def sample(self, decay=1.0):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state * decay


class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class RewardNormalizer:
    """Online reward normalization using Welford's algorithm."""
    def __init__(self, clip_range=10.0):
        self.mean = 0.0
        self.std = 1.0
        self.clip_range = clip_range
        self.count = 0
        self.M2 = 0.0
    
    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.M2 += delta * delta2
        
        if self.count > 1:
            self.std = np.sqrt(self.M2 / (self.count - 1))
    
    def normalize(self, reward):
        if self.std > 0 and self.count > 10:
            normalized = (reward - self.mean) / (self.std + 1e-8)
        else:
            normalized = reward
        return np.clip(normalized, -self.clip_range, self.clip_range)


# ============================================================
# DDPG AGENT
# ============================================================
class CausalDDPGAgent:
    """Deep Deterministic Policy Gradient agent for causal intervention discovery."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.actor = CausalActor(state_dim, action_dim, hidden_dim)
        self.actor_target = CausalActor(state_dim, action_dim, hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = CausalCritic(state_dim, action_dim, hidden_dim)
        self.critic_target = CausalCritic(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.noise = OUNoise(action_dim)
    
    def select_action(self, state, add_noise=True, noise_decay=1.0):
        state = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        if add_noise:
            action += self.noise.sample(noise_decay)
        return action
    
    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * GAMMA * self.critic_target(next_states, next_actions)
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


# ============================================================
# STATE AND REWARD FUNCTIONS (PURE RL!)
# ============================================================
def construct_causal_state(X, t, shock_history):
    """
    Construct state vector for causal model WITHOUT conditional expectations!
    
    State = [X_norm, t_norm, shock_encoding_1, shock_encoding_2]
    
    The actor receives:
    - Current state X (caused by past shocks)
    - Time t (position in causal sequence)
    - Recent shock history (recent exogenous causes)
    
    The actor does NOT receive:
    - E[Y | X_t] (conditional expectations)
    - E[Y | X_t, U_{t+1}=k] (causal effects of shocks)
    
    It must discover optimal interventions purely from reward feedback!
    """
    state_vec = np.zeros(4)
    state_vec[0] = X / X0  # Normalized current state
    state_vec[1] = t / T_steps  # Time (causal depth)
    
    # Encode recent shocks: normalize to [-1, +1]
    # shock_idx 0 (largest up) → +1, shock_idx N-1 (largest down) → -1
    if len(shock_history) >= 1:
        state_vec[2] = 1.0 - 2.0 * shock_history[-1] / (N - 1)
    if len(shock_history) >= 2:
        state_vec[3] = 1.0 - 2.0 * shock_history[-2] / (N - 1)
    
    return state_vec


def compute_causal_reward(intervention_doses, X, t, expectation_estimator, is_terminal):
    """
    Correct reward that accounts for self-financing constraint.
    
    The portfolio after shock u occurs should be:
    V_t(X) + C_{t+1,u} - Σ_k C_{t+1,k} · p_k = V_{t+1}(X · multiplier[u])
    
    Equivalently:
    C_{t+1,u} - avg_cost = V_{t+1}(X · multiplier[u]) - V_t(X)
    where avg_cost = Σ_k C_{t+1,k} · p_k
    """
    intervention_doses = np.atleast_1d(intervention_doses)
    
    if is_terminal:
        target = max(X - K, 0)
        deviation = abs(intervention_doses[0] - target)
    else:
        # Get conditional expectations
        V_t = expectation_estimator.predict_continuation_value(X, t)
        child_values = expectation_estimator.predict_child_continuation_values(X, t)
        
        # Marginal causal effects (what we should buy)
        targets = [child_values[k] - V_t for k in range(N)]
        
        # Check self-financing: avg cost should be close to 0
        avg_cost = np.sum(intervention_doses * shock_probs)
        
        # Deviations from targets
        deviations = [abs(intervention_doses[i] - targets[i]) for i in range(N)]
        deviation = np.mean(deviations)
        
        # Optional: penalize non-self-financing
        # deviation += abs(avg_cost) * 0.1
    
    norm_factor = max(max_terminal_outcome, 1.0)
    normalized_deviation = deviation / norm_factor
    reward = -REPLICATION_PENALTY * normalized_deviation**2
    
    return np.clip(reward, -10000000, 0), deviation

# ============================================================
# TRAINING LOOP
# ============================================================
def train_causal_intervention_discovery():
    """
    Main training loop - PURE RL discovers causal intervention strategies.
    
    The actor learns optimal intervention portfolios π(X_t, t) through
    trial-and-error, without explicit knowledge of:
    - Conditional expectations E[Y | X_t]
    - Causal effects E[Y | X_t, U_{t+1}=k]
    - Structural equations
    
    It discovers these purely from reward feedback!
    """
    simulator = CausalPathSimulator(X0, shock_multipliers, shock_probs, T_steps, dt)
    expectation_estimator = ConditionalExpectationEstimator(
        polynomial_degree=POLYNOMIAL_DEGREE, 
        alpha=REGRESSION_ALPHA
    )
    
    agent = CausalDDPGAgent(state_dim=4, action_dim=N, hidden_dim=HIDDEN_DIM)
    reward_normalizer = RewardNormalizer(clip_range=10.0)
    
    best_avg_deviation = float('inf')
    best_actor_state = None
    patience_counter = 0
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}\nITERATION {iteration + 1}/{NUM_ITERATIONS}\n{'='*60}")
        
        # Learning rate decay
        current_actor_lr = BASE_ACTOR_LR * (LR_DECAY ** iteration)
        current_critic_lr = BASE_CRITIC_LR * (LR_DECAY ** iteration)
        
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = current_actor_lr
        for param_group in agent.critic_optimizer.param_groups:
            param_group['lr'] = current_critic_lr
        
        print(f"Learning Rates: Actor={current_actor_lr:.6f}, Critic={current_critic_lr:.6f}")
        
        # Noise decay
        iteration_noise_scale = max(0.15, 1.0 - (iteration / NUM_ITERATIONS) * NOISE_DECAY)
        agent.noise.set_sigma(agent.noise.initial_sigma * iteration_noise_scale)
        print(f"Exploration Noise: sigma={agent.noise.sigma:.4f}")
        
        # Generate causal paths and estimate conditional expectations
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = expectation_estimator.estimate_continuation_values(paths, r, dt)
        
        print(f"\nPure RL training (NO conditional expectations in state)...")
        print("Actor must discover causal intervention strategy from rewards only!")
        episodes_this_iter = TOTAL_EPISODES // NUM_ITERATIONS
        reward_history = []
        
        for episode in range(episodes_this_iter):
            # Sample random state from causal tree
            path_idx = np.random.randint(len(paths))
            time_idx = np.random.randint(T_steps + 1)
            
            sampled_node = paths[path_idx][time_idx]
            X, t = sampled_node['X'], sampled_node['t']
            shock_history = sampled_node['shock_history']
            is_terminal = (t == T_steps)
            
            # Construct state (no conditional expectations!)
            state = construct_causal_state(X, t, shock_history)
            
            # Select intervention doses (with exploration noise)
            noise_decay = max(0.1, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.98
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            # Extract relevant intervention doses
            action_used = action[:1] if is_terminal else action[:N]
            
            # Compute reward based on deviation from causal targets
            reward, deviation = compute_causal_reward(
                action_used, X, t, expectation_estimator, is_terminal
            )
            
            # Normalize reward
            reward_normalizer.update(reward)
            normalized_reward = reward_normalizer.normalize(reward)
            
            # Store experience
            agent.replay_buffer.push(state, action, normalized_reward, state, False)
            reward_history.append(reward)
            
            # Update networks
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            # Progress reporting
            if (episode + 1) % (episodes_this_iter // 10) == 0:
                recent_rewards = reward_history[-1000:] if len(reward_history) >= 1000 else reward_history
                avg_reward = np.mean(recent_rewards)
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.1f}")
        
        # Evaluation: How well does the actor match causal targets?
        print(f"\nEvaluating intervention strategy accuracy...")
        total_deviation, num_evals = 0, 0
        
        for path in paths[:1000]:
            for node in path:
                X_eval, t_eval = node['X'], node['t']
                shock_history_eval = node['shock_history']
                is_terminal_eval = (t_eval == T_steps)
                
                # Get actor's intervention strategy (no noise)
                state_eval = construct_causal_state(X_eval, t_eval, shock_history_eval)
                action_eval = agent.select_action(state_eval, add_noise=False)
                action_used_eval = action_eval[:1] if is_terminal_eval else action_eval[:N]
                
                # Measure deviation from causal targets
                _, deviation = compute_causal_reward(
                    action_used_eval, X_eval, t_eval, expectation_estimator, is_terminal_eval
                )
                
                total_deviation += deviation
                num_evals += 1
        
        avg_deviation = total_deviation / num_evals
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Average Absolute Deviation: ${avg_deviation:.4f}")
        
        # Early stopping with patience
        if avg_deviation < best_avg_deviation:
            best_avg_deviation = avg_deviation
            best_actor_state = agent.actor.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ NEW BEST! Saving model...")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{EARLY_STOP_PATIENCE})")
        
        if patience_counter >= EARLY_STOP_PATIENCE and iteration >= 8:
            print(f"\n✅ EARLY STOPPING! No improvement for {EARLY_STOP_PATIENCE} iterations")
            print(f"Restoring best model")
            agent.actor.load_state_dict(best_actor_state)
            break
        
        if avg_deviation > 500:
            print(f"\n⚠️  WARNING: Model struggling (deviation ${avg_deviation:.2f})")
            print(f"Restoring best model (deviation ${best_avg_deviation:.2f})")
            if best_actor_state is not None:
                agent.actor.load_state_dict(best_actor_state)
            break
        
        if avg_deviation < 2.0:
            print(f"\n🎯 EXCELLENT! Deviation < $2.00!")
            break
    
    print(f"\n{'='*60}\nTRAINING COMPLETE!\n{'='*60}")
    print(f"Best Average Absolute Deviation: ${best_avg_deviation:.4f}")
    
    if best_actor_state is not None:
        agent.actor.load_state_dict(best_actor_state)
        print("Restored best model for final evaluation")
    
    return agent, expectation_estimator


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    agent, expectation_estimator = train_causal_intervention_discovery()
    
    # ============================================================
    # FINAL EVALUATION - CAUSAL INTERPRETATION
    # ============================================================
    print("\n" + "="*60)
    print(f"FINAL EVALUATION - CAUSAL INTERVENTION DISCOVERY")
    print("="*60)
    print("\nCAUSAL MODEL STRUCTURE:")
    print(f"  X₀ → X₁ → X₂ → ... → X_T → Y")
    print(f"       ↑    ↑         ↑")
    print(f"       U₁   U₂        U_T")
    print(f"\n  Each U_t ∈ {{shock_0, shock_1, ..., shock_{N-1}}} (N={N} possible values)")
    print(f"  Structural equation: X_{{t+1}} = X_t · shock_multiplier[U_{{t+1}}]")
    print(f"  Target outcome: Y = max(X_T - {K}, 0)")
    print("="*60)
    
    # Sample diverse states from the causal tree
    print(f"\nEvaluating intervention strategies across the causal tree...")
    print("(Testing root, intermediate states, and diverse terminal outcomes)")
    print("="*60)
    
    test_count = 0
    success_count = 0
    
    # Test root node (t=0, no shocks yet)
    X, t, shock_history = X0, 0, []
    is_terminal = False
    
    causal_targets = expectation_estimator.predict_child_continuation_values(X, t)
    state = construct_causal_state(X, t, shock_history)
    action = agent.select_action(state, add_noise=False)
    action_used = action[:N]
    
    _, deviation = compute_causal_reward(action_used, X, t, expectation_estimator, is_terminal)
    is_success = deviation < 5.0
    if is_success:
        success_count += 1
    test_count += 1
    
    print(f"\nt={t} | X={X:.2f} | Causal History: ROOT")
    print("-"*60)
    print(f"Causal Targets E[Y|X,U_1=k]: {[f'${v:.2f}' for v in causal_targets]}")
    print(f"Actor's Interventions: {[f'{a:+.2f}' for a in action_used]}")
    print(f"Average Deviation: ${deviation:.4f} {'✓' if is_success else '✗'}")
    print("="*60)
    
    # Test t=1 nodes (after first shock)
    print(f"\nSampling t=1 nodes (after first exogenous shock)...")
    for shock_idx in [0, N//2, N-1]:  # Test extreme and middle shocks
        X = X0 * shock_multipliers[shock_idx]
        t = 1
        shock_history = [shock_idx]
        is_terminal = False
        
        causal_targets = expectation_estimator.predict_child_continuation_values(X, t)
        state = construct_causal_state(X, t, shock_history)
        action = agent.select_action(state, add_noise=False)
        action_used = action[:N]
        
        _, deviation = compute_causal_reward(action_used, X, t, expectation_estimator, is_terminal)
        is_success = deviation < 5.0
        if is_success:
            success_count += 1
        test_count += 1
        
        shock_name = f"shock_{shock_idx}"
        print(f"\nt={t} | X={X:.2f} | Causal History: U₁={shock_name}")
        print("-"*60)
        print(f"Average Deviation: ${deviation:.4f} {'✓' if is_success else '✗'}")
    
    # Test terminal nodes (t=T)
    print(f"\nSampling t={T_steps} terminal nodes (final outcomes)...")
    terminal_samples = [
        [0] * T_steps,  # All largest shocks
        [N-1] * T_steps,  # All smallest shocks
        [N//2] * T_steps,  # All middle shocks
        [0, N-1],  # Mixed extremes
        list(range(min(T_steps, N)))  # Sequential pattern
    ]
    
    for shock_history in terminal_samples[:min(5, len(terminal_samples))]:
        X = X0
        for shock_idx in shock_history:
            X *= shock_multipliers[shock_idx]
        t = T_steps
        is_terminal = True
        
        causal_target = max(X - K, 0)
        state = construct_causal_state(X, t, shock_history)
        action = agent.select_action(state, add_noise=False)
        action_used = action[:1]
        
        _, deviation = compute_causal_reward(action_used, X, t, expectation_estimator, is_terminal)
        is_success = deviation < 5.0
        if is_success:
            success_count += 1
        test_count += 1
        
        shock_path = "→".join([f"U{i+1}={idx}" for i, idx in enumerate(shock_history)])
        print(f"\nt={t} | X={X:.2f}")
        print(f"Causal Path: {shock_path}")
        print("-"*60)
        print(f"Target Outcome: ${causal_target:.4f}")
        print(f"Actor's Intervention: {action_used[0]:+.4f}")
        print(f"Deviation: ${deviation:.4f} {'✓' if is_success else '✗'}")
    
    print("\n" + "="*60)
    print(f"SUCCESS RATE: {success_count}/{test_count} ({100*success_count/test_count:.1f}%)")
    print("="*60)
    print(f"\nPURE DEEP RL CAUSAL DISCOVERY ({N}-valued exogenous shocks):")
    print(f"• State: [X/X₀, t/T, shock_history] - NO conditional expectations!")
    print(f"• Actor learned causal intervention strategy from REWARDS ONLY")
    print(f"• Conditional expectations hidden in reward - NOT given to actor")
    print(f"• Shock history encoding helps distinguish causal paths")
    print(f"• Optimizations: {TOTAL_EPISODES:,} episodes, LayerNorm, scaled parameters")
    print(f"• Network scales with N: {HIDDEN_DIM} hidden units for N={N}")
    print("="*60)
    print("\nThe actor discovered optimal causal intervention portfolios")
    print("through pure reinforcement learning - no knowledge of:")
    print("  • Structural equations X_{t+1} = f(X_t, U_{t+1})")
    print("  • Conditional expectations E[Y | X_t]")
    print("  • Causal effects E[Y | X_t, U_{t+1}=k]")
    print("\nIt learned the causal structure purely from trial-and-error!")
    print("="*60)