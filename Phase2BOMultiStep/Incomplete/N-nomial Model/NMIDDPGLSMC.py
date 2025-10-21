import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from scipy.optimize import minimize
import time

# ============================================================
# CONFIGURATION - N-NOMIAL INCOMPLETE SUPER-REPLICATION
# ============================================================
S0 = 100.0
r = 0.05
K = 100.0
T_steps = 2
dt = 1.0

# N-NOMIAL INCOMPLETE MARKET PARAMETERS
N = 5                    # Number of states (e.g., 5-nomial)
NUM_BINARIES = 2         # INCOMPLETE: 4 binaries for 5 states

sigma = 0.3

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
SHORTFALL_PENALTY = 5000000      # HUGE - must never underpay!
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
print(f"N-NOMIAL INCOMPLETE SUPER-REPLICATION (T={T_steps}, N={N})")
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
# N-NOMIAL PROBABILITIES AND MULTIPLIERS
# ============================================================
def calculate_n_nomial_probabilities_and_multipliers(S0, N, r, dt, sigma):
    """
    Calculate risk-neutral probabilities and multipliers for N-nomial tree.
    Uses symmetric distribution around the middle state.
    """
    growth = np.exp(r * dt)
    
    # Create symmetric multipliers
    if N % 2 == 1:  # Odd N: middle state has multiplier = 1
        middle_idx = N // 2
        multipliers = np.zeros(N)
        multipliers[middle_idx] = 1.0
        
        # Symmetric jumps
        lambda_param = sigma * np.sqrt(dt * N)
        for i in range(middle_idx):
            steps_from_middle = middle_idx - i
            multipliers[i] = np.exp(lambda_param * steps_from_middle)
            multipliers[N - 1 - i] = np.exp(-lambda_param * steps_from_middle)
    else:  # Even N: no middle state = 1
        multipliers = np.zeros(N)
        lambda_param = sigma * np.sqrt(dt * N)
        for i in range(N):
            steps_from_center = (N - 1) / 2 - i
            multipliers[i] = np.exp(lambda_param * steps_from_center)
    
    # Optimize for risk-neutral probabilities
    def objective(p):
        return np.sum((p - 1/N)**2)
    
    def constraint_mean(p):
        return np.sum(p * multipliers) - growth
    
    def constraint_sum(p):
        return np.sum(p) - 1
    
    p0 = np.ones(N) / N
    constraints = [
        {'type': 'eq', 'fun': constraint_mean},
        {'type': 'eq', 'fun': constraint_sum}
    ]
    bounds = [(0.001, 0.999) for _ in range(N)]
    
    result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        probabilities = result.x
    else:
        print("WARNING: Optimization failed, using uniform probabilities")
        probabilities = np.ones(N) / N
    
    expected_growth = np.sum(probabilities * multipliers)
    prob_sum = np.sum(probabilities)
    
    print(f"\nN-nomial (N={N}) Configuration:")
    print(f"Multipliers: {multipliers}")
    print(f"Probabilities: {probabilities}")
    print(f"  Sum = {prob_sum:.6f}, Expected growth = {expected_growth:.6f}")
    
    assert abs(prob_sum - 1.0) < 1e-6, "Probabilities must sum to 1"
    assert abs(expected_growth - growth) < 1e-2, "Must satisfy risk-neutral condition"
    assert np.all(probabilities >= 0), "Probabilities must be non-negative"
    
    print("  ✓ Valid risk-neutral probabilities")
    return probabilities, multipliers


probabilities, multipliers = calculate_n_nomial_probabilities_and_multipliers(S0, N, r, dt, sigma)


# ============================================================
# BINARY PAYOFF MATRIX (SEQUENTIAL PARTITIONING)
# ============================================================
def create_payoff_matrix(N, num_binaries):
    """
    Sequential partitioning for incomplete market:
    - First (num_binaries-1) binaries: one-to-one coverage
    - Last binary: covers all remaining states
    
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
print(f"Rows = States (0 to {N-1}), Columns = Binaries")
print(payoff_matrix.astype(int))
print("\nInterpretation:")
for i in range(NUM_BINARIES):
    states_covered = [j for j in range(N) if payoff_matrix[j, i] == 1]
    print(f"  Binary {i} covers states: {states_covered}")
print(f"\n⚠️  INCOMPLETE MARKET: Binary {NUM_BINARIES-1} covers MULTIPLE states!")
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
class NnomialPathSimulator:
    def __init__(self, S0, multipliers, probabilities, T_steps, dt):
        self.S0 = S0
        self.multipliers = multipliers
        self.probabilities = probabilities
        self.T_steps = T_steps
        self.dt = dt
        self.N = len(multipliers)
    
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
                    state_idx = np.random.choice(self.N, p=self.probabilities)
                    S *= self.multipliers[state_idx]
                    path_step['state_occurred'] = state_idx
                
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
        """Returns continuation values for all N states"""
        if t >= self.T_steps - 1:
            return [max(S * mult - K, 0) for mult in multipliers]
        else:
            base_values = [self.predict_continuation_value(S * mult, t + 1) for mult in multipliers]
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
        x = F.softplus(self.fc4(x)) * self.action_scale  # ← Fixed!
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
    
    INCOMPLETE MARKET: With NUM_BINARIES binaries and N states, 
    the last binary must cover multiple states!
    
    Realized payoff when state i occurs:
      realized[i] = sum(hedge[j] * payoff_matrix[i,j])
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
            # EVALUATION: Check all N states
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
    simulator = NnomialPathSimulator(S0, multipliers, probabilities, T_steps, dt)
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
    
    # State: 2 (S/S0, t/T), Action: NUM_BINARIES
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
                # MULTI-CHILD TRAINING: Sample one of the N possible next states
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
        agent.actor.train()
        
        avg_shortfall = total_shortfall / node_count
        avg_excess = total_excess / node_count
        avg_cost = total_cost / node_count
        
        print(f"\nEvaluation Results (1000 paths):")
        print(f"  Average Shortfall: ${avg_shortfall:.4f}")
        print(f"  Average Excess:    ${avg_excess:.4f}")
        print(f"  Average Cost:      ${avg_cost:.4f}")
        print(f"  Total Nodes Evaluated: {node_count}")
        
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
# FINAL COMPREHENSIVE EVALUATION - HEDGE POSITIONS FOCUS
# ============================================================
def final_evaluation(agent, lsmc_estimator):
    print("\n" + "=" * 60)
    print("FINAL COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    simulator = NnomialPathSimulator(S0, multipliers, probabilities, T_steps, dt)
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
    print(f"\nN={N}-nomial tree with {NUM_BINARIES} binaries (INCOMPLETE MARKET)")
    print(f"Format: [Binary_0, Binary_1, ..., Binary_{NUM_BINARIES-1}]")
    print(f"\nBinary Coverage:")
    for i in range(NUM_BINARIES):
        states_covered = [j for j in range(N) if payoff_matrix[j, i] == 1]
        print(f"  Binary {i} covers states: {states_covered}")
    print(f"{'='*60}\n")
    
    # Print all hedge positions
    for node_info in unique_nodes:
        t = node_info['t']
        S = node_info['S']
        hedge = node_info['hedge']
        
        hedge_str = ", ".join([f"{h:9.3f}" for h in hedge])
        print(f"t={t}, S=${S:8.2f} → [{hedge_str}]")
    
    # Group by timestep for clarity
    print(f"\n{'='*60}")
    print("HEDGE POSITIONS GROUPED BY TIME STEP:")
    print(f"{'='*60}")
    
    for t in range(T_steps + 1):
        nodes_at_t = [n for n in unique_nodes if n['t'] == t]
        if not nodes_at_t:
            continue
        
        print(f"\n{'─'*60}")
        print(f"TIME STEP t={t} ({len(nodes_at_t)} distinct price nodes)")
        print(f"{'─'*60}")
        
        for node_info in nodes_at_t:
            S = node_info['S']
            hedge = node_info['hedge']
            hedge_str = ", ".join([f"{h:9.3f}" for h in hedge])
            print(f"  S=${S:8.2f} → [{hedge_str}]")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("HEDGE POSITION STATISTICS:")
    print(f"{'='*60}")
    
    all_hedges = np.array([n['hedge'] for n in unique_nodes])
    
    for i in range(NUM_BINARIES):
        binary_positions = all_hedges[:, i]
        print(f"\nBinary {i}:")
        print(f"  Min position:  {np.min(binary_positions):9.3f}")
        print(f"  Max position:  {np.max(binary_positions):9.3f}")
        print(f"  Mean position: {np.mean(binary_positions):9.3f}")
        print(f"  Median position: {np.median(binary_positions):9.3f}")
    
    # Total hedge cost analysis
    print(f"\n{'='*60}")
    print("TOTAL HEDGE COST ANALYSIS:")
    print(f"{'='*60}")
    
    for t in range(T_steps + 1):
        nodes_at_t = [n for n in unique_nodes if n['t'] == t]
        if not nodes_at_t:
            continue
        
        costs_at_t = [np.sum(np.abs(n['hedge'])) for n in nodes_at_t]
        print(f"\nt={t}:")
        print(f"  Min total cost:  ${np.min(costs_at_t):8.2f}")
        print(f"  Max total cost:  ${np.max(costs_at_t):8.2f}")
        print(f"  Mean total cost: ${np.mean(costs_at_t):8.2f}")
    
    print(f"\n{'='*60}")
    print("FINANCIAL INTERPRETATION:")
    print(f"{'='*60}")
    print("These are the NUMBER OF BINARY CONTRACTS to purchase at each node")
    print(f"to super-replicate the call option (K=${K}) in an INCOMPLETE market.")
    print(f"\nWith {NUM_BINARIES} binaries covering {N} states, Binary {NUM_BINARIES-1} must cover")
    print(f"MULTIPLE states simultaneously, making perfect replication impossible.")
    print("\nThe RL agent discovered these MINIMAL COST hedge positions through")
    print("pure exploration, finding hedges that satisfy h ≥ continuation_value")
    print("for ALL possible future states, despite market incompleteness.")
    print(f"{'='*60}")
    
    # Print super-replication quality
    print(f"\n{'='*60}")
    print("SUPER-REPLICATION QUALITY CHECK:")
    print(f"{'='*60}")
    
    total_shortfall = 0
    max_shortfall = 0
    nodes_with_shortfall = 0
    
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
                if shortfall > 0:
                    nodes_with_shortfall += 1
                max_shortfall = max(max_shortfall, shortfall)
    agent.actor.train()
    
    total_nodes = len(eval_paths) * (T_steps + 1)
    avg_shortfall = total_shortfall / total_nodes
    pct_with_shortfall = 100 * nodes_with_shortfall / total_nodes
    
    print(f"\nTotal nodes evaluated: {total_nodes:,}")
    print(f"Average shortfall:     ${avg_shortfall:.6f}")
    print(f"Maximum shortfall:     ${max_shortfall:.6f}")
    print(f"Nodes with shortfall:  {nodes_with_shortfall:,} ({pct_with_shortfall:.2f}%)")
    
    if avg_shortfall < 0.10:
        print(f"\n✓ EXCELLENT: Avg shortfall ${avg_shortfall:.6f} << $0.10")
    elif avg_shortfall < 1.0:
        print(f"\n✓ GOOD: Avg shortfall ${avg_shortfall:.6f} < $1.00")
    else:
        print(f"\n✗ NEEDS IMPROVEMENT: Avg shortfall ${avg_shortfall:.6f} ≥ $1.00")
    
    print(f"{'='*60}")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    total_start = time.time()
    
    print("\n" + "=" * 60)
    print(f"STARTING N-NOMIAL INCOMPLETE SUPER-REPLICATION")
    print(f"N={N} states, {NUM_BINARIES} binaries, T={T_steps} time steps")
    print("=" * 60)
    
    trained_agent, trained_lsmc = train_super_replication()
    
    final_evaluation(trained_agent, trained_lsmc)
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TOTAL EXECUTION TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*60}")