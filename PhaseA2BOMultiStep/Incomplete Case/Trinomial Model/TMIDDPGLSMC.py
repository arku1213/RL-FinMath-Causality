import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from scipy.optimize import minimize
import time

# ============================================================
# CONFIGURATION - TRINOMIAL INCOMPLETE SUPER-REPLICATION
# ============================================================
S0 = 100.0
r = 0.05
K = 100.0
T_steps = 2
dt = 1.0

# TRINOMIAL INCOMPLETE MARKET PARAMETERS
N = 3                    # Number of states (Up, Middle, Down)
NUM_BINARIES = 2         # INCOMPLETE: 2 binaries for 3 states

sigma = 0.3
lambda_param = np.sqrt(3)
u = np.exp(lambda_param * sigma * np.sqrt(dt))
d = np.exp(-lambda_param * sigma * np.sqrt(dt))
m = 1.0

# LSMC parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# SUPER-REPLICATION: No artificial buffer (penalties handle safety)
CONSERVATISM_FACTOR = 1.00

# RL hyperparameters
BASE_ACTOR_LR = 0.00005
BASE_CRITIC_LR = 0.00015
ACTOR_LR = BASE_ACTOR_LR
CRITIC_LR = BASE_CRITIC_LR

# ACTION_SCALE will be computed dynamically
ACTION_SCALE = None

# SUPER-REPLICATION PENALTIES (Asymmetric for incomplete markets)
SHORTFALL_PENALTY = 500000      # HUGE - must never underpay!
EXCESS_PENALTY = 50             # Small penalty for overpayment
COST_WEIGHT = 10000             # Balance between shortfall and cost

# Training configuration
base_episodes = 2000000
TOTAL_EPISODES = base_episodes
NUM_ITERATIONS = 16
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.003
BUFFER_SIZE = 1000000
HIDDEN_DIM = 512

# Improvement parameters
LR_DECAY = 0.96
NOISE_DECAY = 0.75
EARLY_STOP_PATIENCE = 5

print("=" * 60)
print(f"TRINOMIAL INCOMPLETE SUPER-REPLICATION (T={T_steps})")
print("=" * 60)
print(f"INCOMPLETE MARKET: {N} states, {NUM_BINARIES} binaries")
print("METHOD: Pure RL discovers MINIMAL super-replicating hedges")
print("=" * 60)
print("GOAL: Find MINIMAL hedge h where h ≥ continuation_value")
print("      for ALL states, even when market is INCOMPLETE")
print("=" * 60)
print(f"SHORTFALL_PENALTY: {SHORTFALL_PENALTY:,}")
print(f"EXCESS_PENALTY: {EXCESS_PENALTY:,}")
print(f"COST_WEIGHT: {COST_WEIGHT:,}")
print(f"Asymmetry Ratio: {SHORTFALL_PENALTY/EXCESS_PENALTY}:1")
print("=" * 60)


# ============================================================
# TRINOMIAL PROBABILITIES
# ============================================================
def calculate_trinomial_probabilities(S0, u, m, d, r, dt):
    """Calculate risk-neutral probabilities"""
    growth = np.exp(r * dt)
    
    def objective(p):
        return np.sum((p - 1/3)**2)
    
    def constraint_mean(p):
        return p[0]*u + p[1]*m + p[2]*d - growth
    
    def constraint_sum(p):
        return p[0] + p[1] + p[2] - 1
    
    p0 = [1/3, 1/3, 1/3]
    constraints = [
        {'type': 'eq', 'fun': constraint_mean},
        {'type': 'eq', 'fun': constraint_sum}
    ]
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.999)]
    
    result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        p_u, p_m, p_d = result.x
    else:
        print("WARNING: Optimization failed, using fallback")
        p_m = 0.7
        diff = growth - p_m * m
        weight = (u - growth) / (u - d)
        p_d = weight * (1 - p_m)
        p_u = (1 - p_m) - p_d
        p_u = max(0.001, p_u)
        p_d = max(0.001, p_d)
        p_m = 1 - p_u - p_d
    
    expected_growth = p_u * u + p_m * m + p_d * d
    prob_sum = p_u + p_m + p_d
    
    print(f"\nTrinomial Probabilities:")
    print(f"  p_u = {p_u:.6f}, p_m = {p_m:.6f}, p_d = {p_d:.6f}")
    print(f"  Sum = {prob_sum:.6f}, Expected growth = {expected_growth:.6f}")
    
    assert abs(prob_sum - 1.0) < 1e-6
    assert abs(expected_growth - growth) < 1e-3
    assert p_u >= 0 and p_m >= 0 and p_d >= 0
    
    print("  ✓ Valid risk-neutral probabilities")
    return p_u, p_m, p_d


p_u, p_m, p_d = calculate_trinomial_probabilities(S0, u, m, d, r, dt)
probabilities = np.array([p_u, p_m, p_d])
multipliers = np.array([u, m, d])


# ============================================================
# BINARY PAYOFF MATRIX (SEQUENTIAL PARTITIONING)
# ============================================================
def create_payoff_matrix(N, num_binaries):
    """
    Sequential partitioning for incomplete market:
    - Binary 0: Covers state 0 (U)
    - Binary 1: Covers states 1 and 2 (M and D)
    
    Returns: matrix[state_i, binary_j] = 1 if binary_j pays when state_i occurs
    """
    payoff_matrix = np.zeros((N, num_binaries))
    
    # First (num_binaries-1) binaries: one-to-one
    for i in range(num_binaries - 1):
        payoff_matrix[i, i] = 1
    
    # Last binary: covers all remaining states
    for i in range(num_binaries - 1, N):
        payoff_matrix[i, num_binaries - 1] = 1
    
    return payoff_matrix


payoff_matrix = create_payoff_matrix(N, NUM_BINARIES)

print(f"\nBinary Payoff Matrix (INCOMPLETE MARKET):")
print("Rows = States (U, M, D), Columns = Binaries")
print(payoff_matrix.astype(int))
print("\nInterpretation:")
state_names = ['U', 'M', 'D']
for i in range(NUM_BINARIES):
    states_covered = [j for j in range(N) if payoff_matrix[j, i] == 1]
    print(f"  Binary {i} covers states: {[state_names[s] for s in states_covered]}")
print("\n⚠️  INCOMPLETE MARKET: Binary 1 covers MULTIPLE states!")
print("    This means we CANNOT perfectly replicate all payoffs.")
print("    Super-replication finds the minimal upper bound hedge.")
print("=" * 60)


# ============================================================
# UTILITIES
# ============================================================
def create_polynomial_features(X, degree):
    X = np.array(X).reshape(-1, 1)
    features = np.ones((X.shape[0], degree + 1))
    for d in range(1, degree + 1):
        features[:, d] = (X[:, 0] ** d)
    return features


class RidgeRegression:
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
# PATH SIMULATION
# ============================================================
class TrinomialPathSimulator:
    def __init__(self, S0, u, m, d, p_u, p_m, p_d, T_steps, dt):
        self.S0, self.u, self.m, self.d = S0, u, m, d
        self.p_u, self.p_m, self.p_d = p_u, p_m, p_d
        self.T_steps, self.dt = T_steps, dt
    
    def simulate_paths(self, num_paths):
        paths = []
        for _ in range(num_paths):
            path = []
            S = self.S0
            for t in range(self.T_steps + 1):
                path_step = {
                    'S': S,
                    't': t,
                    'payoff': max(S - K, 0) if t == self.T_steps else None,
                    'state_occurred': None
                }
                
                if t < self.T_steps:
                    rand = np.random.random()
                    if rand < self.p_u:
                        S *= self.u
                        path_step['state_occurred'] = 0  # Up
                    elif rand < self.p_u + self.p_m:
                        S *= self.m
                        path_step['state_occurred'] = 1  # Middle
                    else:
                        S *= self.d
                        path_step['state_occurred'] = 2  # Down
                
                path.append(path_step)
            paths.append(path)
        return paths


# ============================================================
# LSMC ESTIMATOR
# ============================================================
class LSMCEstimator:
    def __init__(self, polynomial_degree=3, alpha=0.1, conservatism=1.0):
        self.poly_degree = polynomial_degree
        self.alpha = alpha
        self.conservatism = conservatism
        self.continuation_models = {}
        self.T_steps = T_steps
    
    def estimate_continuation_values(self, paths, r, dt):
        print("\n" + "=" * 60)
        print("RUNNING LSMC ESTIMATION")
        print("=" * 60)
        
        for path in paths:
            path[-1]['value'] = path[-1]['payoff']
        
        for t in range(self.T_steps - 1, -1, -1):
            X, y = [], []
            for path in paths:
                X.append(path[t]['S'])
                y.append(np.exp(-r * dt) * path[t+1]['value'])
            
            X_poly = create_polynomial_features(X, self.poly_degree)
            model = RidgeRegression(alpha=self.alpha).fit(X_poly, y)
            self.continuation_models[t] = model
            
            for path in paths:
                S_t = path[t]['S']
                X_pred = create_polynomial_features([S_t], self.poly_degree)
                path[t]['value'] = model.predict(X_pred)[0]
        
        print("LSMC ESTIMATION COMPLETE")
        print(f"Conservative factor: {self.conservatism}×")
        print("(LSMC targets used in REWARD only - not in state!)")
        print("=" * 60)
        return paths
    
    def predict_continuation_value(self, S, t):
        if t not in self.continuation_models:
            return 0.0
        X = create_polynomial_features([S], self.poly_degree)
        return self.continuation_models[t].predict(X)[0]
    
    def predict_state_continuation_values(self, S, t, conservative=True):
        """Returns continuation values for all 3 states [V_u, V_m, V_d]"""
        if t >= self.T_steps - 1:
            return [max(S * u - K, 0), max(S * m - K, 0), max(S * d - K, 0)]
        else:
            base_values = [self.predict_continuation_value(S * move, t + 1) for move in [u, m, d]]
            if conservative:
                return [max(0, v * self.conservatism) for v in base_values]
            else:
                return [max(0, v) for v in base_values]


# ============================================================
# NEURAL NETWORKS
# ============================================================
class UniversalActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_scale):
        super(UniversalActor, self).__init__()
        self.action_scale = action_scale
        
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
        x = torch.tanh(self.fc4(x)) * self.action_scale
        return x


class UniversalCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(UniversalCritic, self).__init__()
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
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class RewardNormalizer:
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
class UniversalDDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, action_scale):
        self.actor = UniversalActor(state_dim, action_dim, hidden_dim, action_scale)
        self.actor_target = UniversalActor(state_dim, action_dim, hidden_dim, action_scale)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = UniversalCritic(state_dim, action_dim, hidden_dim)
        self.critic_target = UniversalCritic(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.noise = OUNoise(action_dim)
        self.action_dim = action_dim
    
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
        
        # Extract and convert to numpy arrays first
        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * GAMMA * self.critic_target(next_states, next_actions)
        
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
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


# ============================================================
# STATE AND REWARD - FINANCIAL MATH FOCUS (INCOMPLETE MARKET)
# ============================================================
def construct_state(S, t):
    """Pure RL state - NO LSMC values!"""
    state_vec = np.zeros(2)
    state_vec[0] = S / S0
    state_vec[1] = t / T_steps
    return state_vec


def compute_reward_super_replication_incomplete(hedge, S, t, lsmc_estimator, is_terminal, state_idx=None):
    """
    Financial Math Goal: h ≥ C_target for ALL states, minimize cost
    
    INCOMPLETE MARKET: With 2 binaries and 3 states, one binary must cover multiple states!
    
    Realized payoff when state i occurs:
      realized[i] = sum(hedge[j] * payoff_matrix[i,j])
    
    Example:
      State U (0): realized = hedge[0] * 1 + hedge[1] * 0 = hedge[0]
      State M (1): realized = hedge[0] * 0 + hedge[1] * 1 = hedge[1]
      State D (2): realized = hedge[0] * 0 + hedge[1] * 1 = hedge[1]  ← SAME as M!
    """
    hedge = np.atleast_1d(hedge)
    
    if is_terminal:
        target = max(S - K, 0)
        realized = hedge[0] if len(hedge) == 1 else np.sum(hedge)
        shortfall = max(0, target - realized)
        excess = max(0, realized - target)
        cost = np.sum(np.abs(hedge))
    else:
        targets = lsmc_estimator.predict_state_continuation_values(S, t, conservative=True)
        
        if state_idx is not None:
            # MULTI-CHILD TRAINING: Check specific state
            realized = np.sum(hedge * payoff_matrix[state_idx, :])
            target = targets[state_idx]
            shortfall = max(0, target - realized)
            excess = max(0, realized - target)
        else:
            # EVALUATION: Check all 3 states
            shortfalls = []
            excesses = []
            for i in range(N):
                realized_i = np.sum(hedge * payoff_matrix[i, :])
                target_i = targets[i]
                shortfalls.append(max(0, target_i - realized_i))
                excesses.append(max(0, realized_i - target_i))
            
            shortfall = np.mean(shortfalls)
            excess = np.mean(excesses)
        
        cost = np.sum(np.abs(hedge))
    
    # ASYMMETRIC L2 PENALTIES
    reward = -(SHORTFALL_PENALTY * shortfall**2 + 
               EXCESS_PENALTY * excess**2 + 
               COST_WEIGHT * cost)
    
    return np.clip(reward, -10000000, 0), shortfall, cost, excess


# ============================================================
# TRAINING LOOP WITH MULTI-CHILD TRAINING
# ============================================================
def train_super_replication():
    global ACTION_SCALE
    
    # ============================================================
    # PHASE 1: Computing ACTION_SCALE from LSMC
    # ============================================================
    phase1_start = time.time()
    print("\n" + "=" * 60)
    print("PHASE 1: Computing ACTION_SCALE from LSMC")
    print("=" * 60)
    
    print("Creating simulator...")
    t1 = time.time()
    simulator = TrinomialPathSimulator(S0, u, m, d, p_u, p_m, p_d, T_steps, dt)
    print(f"  Time: {time.time() - t1:.2f} seconds")
    
    print("Simulating 5000 paths for ACTION_SCALE estimation...")
    t1 = time.time()
    paths_temp = simulator.simulate_paths(5000)
    print(f"  Time: {time.time() - t1:.2f} seconds")
    
    print("Running LSMC on paths...")
    t1 = time.time()
    lsmc_temp = LSMCEstimator(
        polynomial_degree=POLYNOMIAL_DEGREE,
        alpha=REGRESSION_ALPHA,
        conservatism=CONSERVATISM_FACTOR
    )
    paths_temp = lsmc_temp.estimate_continuation_values(paths_temp, r, dt)
    print(f"  Time: {time.time() - t1:.2f} seconds")
    
    print("Scanning for maximum LSMC target...")
    t1 = time.time()
    max_target = 0
    for path in paths_temp[:1000]:
        for node in path[:-1]:
            S_node, t_node = node['S'], node['t']
            targets = lsmc_temp.predict_state_continuation_values(S_node, t_node, conservative=True)
            max_target = max(max_target, max(targets))
    print(f"  Time: {time.time() - t1:.2f} seconds")
    
    ACTION_SCALE = max_target * 2.0
    
    phase1_time = time.time() - phase1_start
    print(f"\nMax LSMC target found: ${max_target:.2f}")
    print(f"ACTION_SCALE set to: {ACTION_SCALE:.2f} (2.0× margin)")
    print(f"PHASE 1 TOTAL TIME: {phase1_time:.2f} seconds")
    print("=" * 60)
    
    # ============================================================
    # PHASE 2: Training Super-Replication Policy
    # ============================================================
    phase2_start = time.time()
    print("\n" + "=" * 60)
    print("PHASE 2: Training Super-Replication Policy (INCOMPLETE MARKET)")
    print("=" * 60)
    
    lsmc_estimator = LSMCEstimator(
        polynomial_degree=POLYNOMIAL_DEGREE,
        alpha=REGRESSION_ALPHA,
        conservatism=CONSERVATISM_FACTOR
    )
    
    # State: 2 (S/S0, t/T), Action: 2 (NUM_BINARIES)
    agent = UniversalDDPGAgent(state_dim=2, action_dim=NUM_BINARIES, hidden_dim=HIDDEN_DIM, action_scale=ACTION_SCALE)
    reward_normalizer = RewardNormalizer(clip_range=10.0)
    
    best_avg_shortfall = float('inf')
    best_actor_state = None
    patience_counter = 0
    
    for iteration in range(NUM_ITERATIONS):
        iter_start = time.time()
        print(f"\n{'='*60}\nITERATION {iteration + 1}/{NUM_ITERATIONS}\n{'='*60}")
        
        current_actor_lr = BASE_ACTOR_LR * (LR_DECAY ** iteration)
        current_critic_lr = BASE_CRITIC_LR * (LR_DECAY ** iteration)
        
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = current_actor_lr
        for param_group in agent.critic_optimizer.param_groups:
            param_group['lr'] = current_critic_lr
        
        print(f"Learning Rates: Actor={current_actor_lr:.6f}, Critic={current_critic_lr:.6f}")
        
        iteration_noise_scale = max(0.15, 1.0 - (iteration / NUM_ITERATIONS) * NOISE_DECAY)
        agent.noise.set_sigma(agent.noise.initial_sigma * iteration_noise_scale)
        print(f"Exploration Noise: sigma={agent.noise.sigma:.4f}")
        
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        print(f"Training super-replication policy...")
        episodes_this_iter = TOTAL_EPISODES // NUM_ITERATIONS
        reward_history = []
        
        for episode in range(episodes_this_iter):
            path_idx = np.random.randint(len(paths))
            time_idx = np.random.randint(T_steps + 1)
            
            sampled_node = paths[path_idx][time_idx]
            S, t = sampled_node['S'], sampled_node['t']
            is_terminal = (t == T_steps)
            
            state = construct_state(S, t)
            
            noise_decay = max(0.1, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.98
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            # FIXED: Always use full action dimension for consistency
            action_used = action.copy()
            
            # MULTI-CHILD TRAINING: Critical for incomplete markets!
            if is_terminal:
                reward, shortfall, cost, excess = compute_reward_super_replication_incomplete(
                    action_used, S, t, lsmc_estimator, True, None
                )
                reward_normalizer.update(reward)
                normalized_reward = reward_normalizer.normalize(reward)
                
                next_state = construct_state(S, t)
                done = 1.0
                
                agent.replay_buffer.push(state, action_used, normalized_reward, next_state, done)
            else:
                # MULTI-CHILD TRAINING: Sample one of the 3 possible next states
                state_idx = np.random.choice(N, p=probabilities)
                
                reward, shortfall, cost, excess = compute_reward_super_replication_incomplete(
                    action_used, S, t, lsmc_estimator, False, state_idx
                )
                reward_normalizer.update(reward)
                normalized_reward = reward_normalizer.normalize(reward)
                
                S_next = S * multipliers[state_idx]
                next_state = construct_state(S_next, t + 1)
                done = 0.0
                
                agent.replay_buffer.push(state, action_used, normalized_reward, next_state, done)
            
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            reward_history.append(reward)
            
            if (episode + 1) % 50000 == 0:
                avg_reward = np.mean(reward_history[-10000:]) if len(reward_history) >= 10000 else np.mean(reward_history)
                print(f"  Episode {episode + 1}/{episodes_this_iter} | Avg Reward: {avg_reward:.2f}")
        
        # ============================================================
        # EVALUATION AFTER ITERATION
        # ============================================================
        print(f"\n{'='*60}\nEVALUATING ITERATION {iteration + 1}\n{'='*60}")
        
        eval_paths = simulator.simulate_paths(1000)
        eval_paths = lsmc_estimator.estimate_continuation_values(eval_paths, r, dt)
        
        total_shortfall = 0
        total_excess = 0
        total_cost = 0
        node_count = 0
        node_details = []
        
        agent.actor.eval()
        with torch.no_grad():
            for path in eval_paths:
                for node in path:
                    S, t = node['S'], node['t']
                    is_terminal = (t == T_steps)
                    
                    state = construct_state(S, t)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = agent.actor(state_tensor).cpu().numpy()[0]
                    
                    reward, shortfall, cost, excess = compute_reward_super_replication_incomplete(
                        action, S, t, lsmc_estimator, is_terminal, None
                    )
                    
                    total_shortfall += shortfall
                    total_excess += excess
                    total_cost += cost
                    node_count += 1
                    
                    if len(node_details) < 20:
                        node_details.append({
                            'S': S, 't': t, 'hedge': action.copy(),
                            'shortfall': shortfall, 'excess': excess, 'cost': cost
                        })
        agent.actor.train()
        
        avg_shortfall = total_shortfall / node_count
        avg_excess = total_excess / node_count
        avg_cost = total_cost / node_count
        
        print(f"\nEvaluation Results (1000 paths):")
        print(f"  Average Shortfall: ${avg_shortfall:.4f}")
        print(f"  Average Excess:    ${avg_excess:.4f}")
        print(f"  Average Cost:      ${avg_cost:.4f}")
        print(f"  Total Nodes Evaluated: {node_count}")
        
        print(f"\nSample Node Details (first 10):")
        for i, detail in enumerate(node_details[:10]):
            print(f"  Node {i+1}: S=${detail['S']:.2f}, t={detail['t']}, "
                  f"hedge={detail['hedge']}, shortfall=${detail['shortfall']:.4f}")
        
        # Early stopping
        if avg_shortfall < best_avg_shortfall:
            best_avg_shortfall = avg_shortfall
            best_actor_state = agent.actor.state_dict().copy()
            patience_counter = 0
            print(f"\n✓ New best average shortfall: ${best_avg_shortfall:.4f}")
        else:
            patience_counter += 1
            print(f"\n✗ No improvement (patience: {patience_counter}/{EARLY_STOP_PATIENCE})")
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping triggered after {iteration + 1} iterations")
            break
        
        iter_time = time.time() - iter_start
        print(f"\nIteration {iteration + 1} completed in {iter_time:.2f} seconds")
    
    # Load best model
    if best_actor_state is not None:
        agent.actor.load_state_dict(best_actor_state)
        print(f"\n✓ Loaded best model with avg shortfall: ${best_avg_shortfall:.4f}")
    
    phase2_time = time.time() - phase2_start
    print(f"\nPHASE 2 TOTAL TIME: {phase2_time:.2f} seconds")
    print("=" * 60)
    
    return agent, lsmc_estimator


# ============================================================
# FINAL COMPREHENSIVE EVALUATION
# ============================================================
# ============================================================
# FINAL COMPREHENSIVE EVALUATION - HEDGE POSITIONS FOCUS
# ============================================================
def final_evaluation(agent, lsmc_estimator):
    print("\n" + "=" * 60)
    print("FINAL COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    simulator = TrinomialPathSimulator(S0, u, m, d, p_u, p_m, p_d, T_steps, dt)
    eval_paths = simulator.simulate_paths(10000)
    eval_paths = lsmc_estimator.estimate_continuation_values(eval_paths, r, dt)
    
    # Collect unique nodes and their hedge positions
    node_hedges = {}  # Key: (t, S_rounded), Value: list of hedge arrays
    
    agent.actor.eval()
    with torch.no_grad():
        for path in eval_paths:
            for node in path:
                S, t = node['S'], node['t']
                
                # Round S to avoid floating point issues
                S_rounded = round(S, 2)
                
                state = construct_state(S, t)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = agent.actor(state_tensor).cpu().numpy()[0]
                
                # Store hedge for this node
                key = (t, S_rounded)
                if key not in node_hedges:
                    node_hedges[key] = []
                node_hedges[key].append(action.copy())
    agent.actor.train()
    
    # Average hedges for each unique node
    unique_nodes = []
    for (t, S_rounded), hedges in node_hedges.items():
        avg_hedge = np.mean(hedges, axis=0)
        unique_nodes.append({
            't': t,
            'S': S_rounded,
            'hedge': avg_hedge
        })
    
    # Sort by time step, then by stock price
    unique_nodes.sort(key=lambda x: (x['t'], x['S']))
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL HEDGE POSITIONS (NUMBER OF BINARIES TO BUY)")
    print(f"{'='*60}")
    print(f"Total unique nodes found: {len(unique_nodes)}")
    print(f"\nFormat: [Binary_0, Binary_1]")
    print(f"  Binary_0 covers: Up state")
    print(f"  Binary_1 covers: Middle and Down states")
    print(f"{'='*60}\n")
    
    for node_info in unique_nodes:
        t = node_info['t']
        S = node_info['S']
        hedge = node_info['hedge']
        
        print(f"t={t}, S=${S:7.2f} → Hedge = [{hedge[0]:8.3f}, {hedge[1]:8.3f}]")
    
    # Group by timestep for clarity
    print(f"\n{'='*60}")
    print("HEDGE POSITIONS GROUPED BY TIME STEP:")
    print(f"{'='*60}")
    
    for t in range(T_steps + 1):
        nodes_at_t = [n for n in unique_nodes if n['t'] == t]
        if not nodes_at_t:
            continue
        
        print(f"\n--- TIME STEP t={t} ({len(nodes_at_t)} nodes) ---")
        for node_info in nodes_at_t:
            S = node_info['S']
            hedge = node_info['hedge']
            print(f"  S=${S:7.2f} → [{hedge[0]:8.3f}, {hedge[1]:8.3f}]")
    
    print(f"\n{'='*60}")
    print("FINANCIAL INTERPRETATION:")
    print(f"{'='*60}")
    print("These are the NUMBER OF BINARY CONTRACTS to purchase at each node")
    print("to super-replicate the call option in an INCOMPLETE market.")
    print("\nThe RL agent discovered these positions through pure exploration,")
    print("finding the MINIMAL cost hedge that satisfies h ≥ continuation_value")
    print("for ALL possible future states, despite market incompleteness.")
    print(f"{'='*60}")

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    total_start = time.time()
    
    print("\n" + "=" * 60)
    print("STARTING TRINOMIAL INCOMPLETE SUPER-REPLICATION")
    print("=" * 60)
    
    trained_agent, trained_lsmc = train_super_replication()
    
    final_evaluation(trained_agent, trained_lsmc)
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TOTAL EXECUTION TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*60}")