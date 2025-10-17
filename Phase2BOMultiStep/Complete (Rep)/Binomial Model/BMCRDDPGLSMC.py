import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ============================================================
# CONFIGURATION
# ============================================================
S0 = 100.0
u = 1.2214
d = 0.8187
r = 0.05
K = 100.0
T_steps = 2  # ← CHANGE THIS: 2, 3, 4, 5, etc.
dt = 1.0

# LSMC Parameters
NUM_SIMULATIONS = 10000  # Number of paths to simulate
POLYNOMIAL_DEGREE = 3    # Degree for regression (1=linear, 2=quadratic, 3=cubic)
REGRESSION_ALPHA = 0.1   # Ridge regularization parameter

# Universal Agent Hyperparameters
TOTAL_EPISODES = 100000  # Total training episodes across all iterations
NUM_ITERATIONS = 6       # Number of LSMC → Train → Update cycles
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0003
CRITIC_LR = 0.0008
BUFFER_SIZE = 200000
HIDDEN_DIM = 256

# Reward shaping - FIXED FOR PERFECT REPLICATION
PENALTY_MULTIPLIER = 500  # Strong penalty for violations
COST_WEIGHT = 1.0  # INCREASED from 0.01 - now agent cares about cost!
EXCESS_PENALTY = 100.0  # INCREASED from 0.01 - penalize over-hedging strongly!

# Dynamic Action Scale
q = (np.exp(r * dt) - d) / (u - d)
max_stock_price = S0 * (u ** T_steps)
max_terminal_payoff = max(max_stock_price - K, 0)
CONTINUATION_MULTIPLIER = 3.0 + (T_steps * 1.0)
ACTION_SCALE = max(150.0, max_terminal_payoff * CONTINUATION_MULTIPLIER)

print("="*60)
print(f"LSMC + UNIVERSAL AGENT: BINOMIAL T={T_steps}")
print(f"MODE: PERFECT REPLICATION (Complete Market)")
print("="*60)
print(f"LSMC Simulations: {NUM_SIMULATIONS}")
print(f"Polynomial Degree: {POLYNOMIAL_DEGREE}")
print(f"Total Training Episodes: {TOTAL_EPISODES}")
print(f"Iterations: {NUM_ITERATIONS}")
print(f"Action Scale: ±{ACTION_SCALE:.1f}")
print(f"Risk-neutral prob q: {q:.4f}")
print(f"Cost Weight: {COST_WEIGHT} (HIGH - exact replication)")
print(f"Excess Penalty: {EXCESS_PENALTY} (HIGH - avoid over-hedging)")
print("="*60)

# ============================================================
# POLYNOMIAL FEATURES (Native Implementation)
# ============================================================
def create_polynomial_features(X, degree):
    """
    Create polynomial features up to specified degree
    X: array-like (will be converted to numpy array)
    Returns: (n_samples, degree+1) array with [1, X, X^2, X^3, ...]
    """
    # Convert to numpy array first
    X = np.array(X)
    
    # Ensure 2D shape
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples = X.shape[0]
    
    # Create polynomial features: [1, X, X^2, ..., X^degree]
    features = np.ones((n_samples, degree + 1))
    for d in range(1, degree + 1):
        features[:, d] = (X[:, 0] ** d)
    
    return features

# ============================================================
# RIDGE REGRESSION (Native Implementation)
# ============================================================
class RidgeRegression:
    """Native implementation of Ridge Regression"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit ridge regression: β = (X'X + αI)^(-1) X'y
        """
        X = np.array(X)
        y = np.array(y)
        
        # Add regularization to normal equations
        n_features = X.shape[1]
        XtX = X.T @ X
        reg_matrix = self.alpha * np.eye(n_features)
        
        # Solve: (X'X + αI)β = X'y
        try:
            self.coef_ = np.linalg.solve(XtX + reg_matrix, X.T @ y)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            self.coef_ = np.linalg.lstsq(XtX + reg_matrix, X.T @ y, rcond=None)[0]
        
        self.intercept_ = 0  # Already included in X (first column is 1s)
        
        return self
    
    def predict(self, X):
        """Predict using fitted model"""
        X = np.array(X)
        return X @ self.coef_
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

# ============================================================
# PATH SIMULATION (Monte Carlo)
# ============================================================
class PathSimulator:
    """Simulate paths through binomial lattice"""
    
    def __init__(self, S0, u, d, q, T_steps, dt):
        self.S0 = S0
        self.u = u
        self.d = d
        self.q = q
        self.T_steps = T_steps
        self.dt = dt
    
    def simulate_paths(self, num_paths):
        """
        Simulate random paths through binomial lattice
        Returns: List of paths, where each path is a list of (S, t) tuples
        """
        paths = []
        
        for _ in range(num_paths):
            path = []
            S = self.S0
            
            for t in range(self.T_steps + 1):
                path.append({
                    'S': S,
                    't': t,
                    'payoff': max(S - K, 0) if t == self.T_steps else None,
                    'value': None  # Will be filled by LSMC
                })
                
                # Generate next state (if not terminal)
                if t < self.T_steps:
                    # Random transition: up with prob q, down with prob 1-q
                    if np.random.random() < self.q:
                        S = S * self.u  # Up
                    else:
                        S = S * self.d  # Down
            
            paths.append(path)
        
        return paths

# ============================================================
# LSMC CONTINUATION VALUE ESTIMATOR
# ============================================================
class LSMCEstimator:
    """Least Squares Monte Carlo for continuation value estimation"""
    
    def __init__(self, polynomial_degree=3, alpha=0.1):
        self.poly_degree = polynomial_degree
        self.alpha = alpha
        self.continuation_models = {}  # Store regression models per time step
        self.T_steps = T_steps
    
    def estimate_continuation_values(self, paths, r, dt):
        """
        Use LSMC to estimate continuation values at each time step
        
        Algorithm:
        1. Start from terminal time (values = payoffs)
        2. Work backwards, fitting regression at each time
        3. V(S,t) ≈ β₀ + β₁·S + β₂·S² + β₃·S³ + ...
        """
        print("\n" + "="*60)
        print("RUNNING LSMC ESTIMATION")
        print("="*60)
        
        # Initialize terminal values
        for path in paths:
            terminal = path[-1]
            terminal['value'] = terminal['payoff']
        
        # Work backwards through time
        for t in range(self.T_steps - 1, -1, -1):
            print(f"\nTime t={t}: Fitting continuation value regression")
            
            # Collect (state, discounted_future_value) pairs
            X = []  # Stock prices (features)
            y = []  # Continuation values (targets)
            
            for path in paths:
                current_state = path[t]
                future_state = path[t + 1]
                
                S_t = current_state['S']
                V_future = future_state['value']
                
                # Discounted continuation value
                continuation_value = np.exp(-r * dt) * V_future
                
                X.append(S_t)
                y.append(continuation_value)
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Create polynomial features
            X_poly = create_polynomial_features(X, self.poly_degree)
            
            # Fit Ridge regression
            model = RidgeRegression(alpha=self.alpha)
            model.fit(X_poly, y)
            
            # Store model
            self.continuation_models[t] = model
            
            # Update path values using fitted model
            for path in paths:
                current_state = path[t]
                S_t = current_state['S']
                X_pred = create_polynomial_features([S_t], self.poly_degree)
                current_state['value'] = model.predict(X_pred)[0]
            
            # Report statistics
            predicted = model.predict(X_poly)
            mse = np.mean((y - predicted) ** 2)
            r2 = model.score(X_poly, y)
            print(f"  Regression MSE: {mse:.6f}, R²: {r2:.4f}")
            print(f"  Value range: [{np.min(y):.2f}, {np.max(y):.2f}]")
        
        print("\n" + "="*60)
        print("LSMC ESTIMATION COMPLETE")
        print("="*60)
        
        return paths
    
    def predict_continuation_value(self, S, t):
        """Predict continuation value for a given state"""
        if t not in self.continuation_models:
            return 0.0
        
        X = create_polynomial_features([S], self.poly_degree)
        return self.continuation_models[t].predict(X)[0]

# ============================================================
# UNIVERSAL AGENT
# ============================================================
class UniversalActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(UniversalActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)
        nn.init.uniform_(self.fc4.bias, 0.0, 0.01)
    
    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        x = torch.tanh(self.fc4(x)) * ACTION_SCALE
        return x

class UniversalCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(UniversalCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)
        nn.init.uniform_(self.fc4.bias, -0.003, 0.003)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self, decay_factor=1.0):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state * decay_factor

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class UniversalDDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.actor = UniversalActor(state_dim, action_dim, hidden_dim)
        self.actor_target = UniversalActor(state_dim, action_dim, hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = UniversalCritic(state_dim, action_dim, hidden_dim)
        self.critic_target = UniversalCritic(state_dim, action_dim, hidden_dim)
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
            return None, None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * GAMMA * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()

# ============================================================
# STATE CONSTRUCTION & REWARD - FIXED FOR EXACT REPLICATION
# ============================================================
def construct_state(S, t, target_value):
    """
    Construct state representation for universal agent
    Features: [normalized_S, normalized_target, normalized_time]
    """
    return np.array([
        S / S0,                           # Normalized stock price
        target_value / max_terminal_payoff if max_terminal_payoff > 0 else 0,  # Normalized target value
        t / T_steps if T_steps > 0 else 0  # Normalized time
    ])

def compute_reward(hedge, target_value, binary_prices, is_terminal):
    """
    Compute reward for hedging action
    FIXED: Penalize both shortfall AND excess heavily for exact replication
    """
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    realized = np.sum(hedge)
    
    shortfall = max(0, target_value - realized)
    excess = max(0, realized - target_value)
    
    # Normalize by max payoff
    normalized_shortfall = shortfall / max_terminal_payoff if max_terminal_payoff > 0 else shortfall
    normalized_excess = excess / max_terminal_payoff if max_terminal_payoff > 0 else excess
    
    # Penalty weight based on node type
    penalty_weight = 100 if is_terminal else 150
    
    # FIXED: Strong penalties for BOTH shortfall and excess
    reward = -(COST_WEIGHT * abs(cost) 
               + penalty_weight * PENALTY_MULTIPLIER * normalized_shortfall**2
               + penalty_weight * EXCESS_PENALTY * normalized_excess**2)  # INCREASED!
    
    reward = np.clip(reward, -10000, 0)
    
    return reward, cost, shortfall

# ============================================================
# ITERATIVE LSMC + UNIVERSAL AGENT TRAINING
# ============================================================
def train_universal_agent_with_lsmc():
    """
    Main training loop: LSMC → Train Agent → Repeat
    """
    print("\n" + "="*60)
    print("ITERATIVE LSMC + UNIVERSAL AGENT TRAINING")
    print("="*60)
    
    # Initialize components
    simulator = PathSimulator(S0, u, d, q, T_steps, dt)
    lsmc_estimator = LSMCEstimator(polynomial_degree=POLYNOMIAL_DEGREE, alpha=REGRESSION_ALPHA)
    
    # State: [S_normalized, target_normalized, t_normalized]
    state_dim = 3
    # Action: 2 binaries for intermediate nodes, 1 for terminals
    action_dim = 2  # Will use only first element for terminals
    
    agent = UniversalDDPGAgent(state_dim, action_dim, HIDDEN_DIM)
    
    best_total_shortfall = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'='*60}")
        
        # ============================================================
        # PHASE 1: RUN LSMC TO GET CONTINUATION VALUES
        # ============================================================
        print(f"\nPhase 1: Simulating {NUM_SIMULATIONS} paths and running LSMC...")
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        # ============================================================
        # PHASE 2: TRAIN UNIVERSAL AGENT ON SAMPLED STATES
        # ============================================================
        print(f"\nPhase 2: Training universal agent...")
        episodes_this_iter = TOTAL_EPISODES // NUM_ITERATIONS
        warmup_episodes = episodes_this_iter // 5
        
        reward_history = []
        
        for episode in range(episodes_this_iter):
            # Sample a random state from simulated paths
            path_idx = np.random.randint(0, len(paths))
            time_idx = np.random.randint(0, T_steps + 1)
            
            sampled_state = paths[path_idx][time_idx]
            S = sampled_state['S']
            t = sampled_state['t']
            
            # Determine target value and action dimension
            is_terminal = (t == T_steps)
            
            if is_terminal:
                target = sampled_state['payoff']
                n_binaries = 1
                prices = [np.exp(-r * dt)]
            else:
                # Use LSMC continuation value
                target = sampled_state['value']
                n_binaries = 2
                prices = [np.exp(-r * dt), np.exp(-r * dt)]
            
            # Construct state
            state = construct_state(S, t, target)
            
            # Agent selects action
            noise_decay = max(0.1, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.8
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            # Use appropriate number of actions
            if is_terminal:
                action_used = np.maximum(action[:1], 0)  # Force non-negative for terminals
                prices_used = prices[:1]
            else:
                action_used = action[:n_binaries]
                prices_used = prices[:n_binaries]
            
            # Compute reward
            reward, cost, shortfall = compute_reward(action_used, target, prices_used, is_terminal)
            
            # Store in replay buffer
            agent.replay_buffer.push(state, action, reward, state, True)
            
            # Update agent
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            reward_history.append(reward)
            
            # Logging
            if (episode + 1) % 5000 == 0:
                avg_reward = np.mean(reward_history[-1000:])
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.2f}")
        
        # ============================================================
        # PHASE 3: EVALUATE AGENT ON ALL STATES FROM PATHS
        # ============================================================
        print(f"\nPhase 3: Evaluating agent performance...")
        
        total_shortfall = 0
        total_cost = 0
        num_states = 0
        
        for path in paths[:1000]:  # Sample 1000 paths for evaluation
            for state_info in path:
                S = state_info['S']
                t = state_info['t']
                is_terminal = (t == T_steps)
                
                if is_terminal:
                    target = state_info['payoff']
                    n_binaries = 1
                    prices = [np.exp(-r * dt)]
                else:
                    target = state_info['value']
                    n_binaries = 2
                    prices = [np.exp(-r * dt), np.exp(-r * dt)]
                
                state = construct_state(S, t, target)
                action = agent.select_action(state, add_noise=False)
                
                if is_terminal:
                    action_used = np.maximum(action[:1], 0)
                    prices_used = prices[:1]
                else:
                    action_used = action[:n_binaries]
                    prices_used = prices[:n_binaries]
                
                _, cost, shortfall = compute_reward(action_used, target, prices_used, is_terminal)
                
                total_shortfall += shortfall
                total_cost += cost
                num_states += 1
        
        avg_shortfall = total_shortfall / num_states
        avg_cost = total_cost / num_states
        
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Avg Shortfall: {avg_shortfall:.6f}")
        print(f"  Avg Cost: ${avg_cost:.4f}")
        print(f"  Total Shortfall (1000 paths): {total_shortfall:.6f}")
        
        if total_shortfall < best_total_shortfall:
            best_total_shortfall = total_shortfall
            print(f"  ✓ New best shortfall!")
        
        if avg_shortfall < 0.01:
            print(f"\n🎉 SUCCESS! Avg Shortfall < 0.01 at iteration {iteration + 1}")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best Total Shortfall: {best_total_shortfall:.6f}")
    print("="*60)
    
    return agent, lsmc_estimator

# ============================================================
# MAIN EXECUTION
# ============================================================
agent, lsmc_estimator = train_universal_agent_with_lsmc()

# ============================================================
# FINAL EVALUATION
# ============================================================
# ============================================================
# FINAL EVALUATION — FORMATTED HEDGE STRUCTURE REPORT
# ============================================================
print("\n" + "=" * 60)
print("HEDGE STRUCTURE — BINOMIAL MODEL")
print("=" * 60)
print(f"Goal: Determine binary holdings at each node to replicate continuation value")
print(f"Mode: PERFECT REPLICATION (Complete Market)")
print("-" * 60)
print(f"Binary contract discount: exp(-r*dt) = {np.exp(-r*dt):.4f}")
print("=" * 60)

# Theoretical comparison function
def calculate_theoretical_value(S, t):
    """Theoretical European call price by backward induction"""
    if t == T_steps:
        return max(S - K, 0)
    V_up = calculate_theoretical_value(S * u, t + 1)
    V_down = calculate_theoretical_value(S * d, t + 1)
    return np.exp(-r * dt) * (q * V_up + (1 - q) * V_down)

# Evaluate hedge at key states
test_states = [
    (S0, 0),            # initial
    (S0 * u, 1),        # up
    (S0 * d, 1),        # down
]
if T_steps >= 2:
    test_states += [
        (S0 * u * u, 2),
        (S0, 2),
        (S0 * d * d, 2),
    ]

node_results = []
for S, t in test_states:
    is_terminal = (t == T_steps)
    theoretical = calculate_theoretical_value(S, t)
    target = max(S - K, 0) if is_terminal else lsmc_estimator.predict_continuation_value(S, t)

    n_binaries = 1 if is_terminal else 2
    prices = [np.exp(-r * dt)] * n_binaries
    state = construct_state(S, t, target)
    action = agent.select_action(state, add_noise=False)
    action_used = np.maximum(action[:n_binaries], 0) if is_terminal else action[:n_binaries]

    _, cost, shortfall = compute_reward(action_used, target, prices, is_terminal)
    realized = np.sum(action_used)
    excess = max(0, realized - target)

    node_results.append({
        "S": S,
        "t": t,
        "target": target,
        "theoretical": theoretical,
        "hedge": action_used,
        "cost": cost,
        "realized": realized,
        "shortfall": shortfall,
        "excess": excess
    })

# Pretty print all nodes
for node in node_results:
    S, t = node["S"], node["t"]
    print(f"\nt = {t} | S = {S:.2f}")
    print("-" * 60)
    print(f"Target (continuation) value : {node['target']:.4f}")
    print(f"Theoretical value           : {node['theoretical']:.4f}")
    print("-" * 60)

    hedge = node['hedge']
    if len(hedge) == 2:
        print("Hedge composition (Binary Holdings):")
        print(f"  Down binary : {hedge[0]:+,.4f}")
        print(f"  Up binary   : {hedge[1]:+,.4f}")
    else:
        print("Hedge composition (Binary Holdings):")
        print(f"  Terminal binary : {hedge[0]:+,.4f}")
    print("-" * 60)

    print(f"Replication cost : {node['cost']:.4f}")
    print(f"Realized value   : {node['realized']:.4f}")
    print(f"Shortfall        : {node['shortfall']:.4f}")
    print(f"Excess           : {node['excess']:.4f}")
    print("-" * 60)

    # Interpretation
    if abs(node['shortfall']) < 1e-3 and abs(node['excess']) < 1e-3:
        interp = "→ Perfect replication achieved"
    elif node['shortfall'] > 0:
        interp = f"→ Under-replication (shortfall = {node['shortfall']:.4f})"
    else:
        interp = f"→ Over-replication (excess = {node['excess']:.4f})"
    print(f"Interpretation:\n  {interp}")
    print("=" * 60)

# Global summary
avg_shortfall = np.mean([n["shortfall"] for n in node_results])
avg_excess = np.mean([n["excess"] for n in node_results])
max_mag = np.max([abs(v) for n in node_results for v in n["hedge"]])

print("\n" + "=" * 60)
print("REPLICATION SUMMARY")
print("=" * 60)
print(f"• Total nodes evaluated      : {len(node_results)}")
print(f"• Avg shortfall              : {avg_shortfall:.6f}")
print(f"• Avg excess                 : {avg_excess:.6f}")
print(f"• Max abs hedge magnitude    : {max_mag:.4f}")
print(f"• Market type                : COMPLETE (binomial)")
print(f"• Result                     : {'Exact replication' if avg_shortfall < 0.01 else 'Approximate replication'}")
print("=" * 60)
print("DONE!")
print("=" * 60)
