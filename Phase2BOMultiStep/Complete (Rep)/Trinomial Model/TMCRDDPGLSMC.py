import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION
# ============================================================
S0 = 100.0
r = 0.05
K = 100.0
T_steps = 2
dt = 1.0

# FIXED TRINOMIAL PARAMETERS (Boyle trinomial - guaranteed positive probabilities)
sigma = 0.3  # Volatility
lambda_param = np.sqrt(3)  # Stretching parameter for trinomial
u = np.exp(lambda_param * sigma * np.sqrt(dt))    # ≈ 1.67
d = np.exp(-lambda_param * sigma * np.sqrt(dt))   # ≈ 0.60
m = 1.0  # Middle state (stock price unchanged)

# LSMC Parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# ============================================================
# FINANCIAL MATH FOCUS: EXACT REPLICATION
# ============================================================
ACTOR_LR = 0.0001
CRITIC_LR = 0.0003

max_stock_price = S0 * (u ** T_steps)
max_terminal_payoff = max(max_stock_price - K, 0)
ACTION_SCALE = max_terminal_payoff * 3.0

# CRITICAL: Find EXACT hedge ratios (financial math goal)
REPLICATION_PENALTY = 1000000   # Massive - must match exactly!
COST_WEIGHT = 0.0               # ZERO - don't care about cost at all!
EXTREME_PENALTY_WEIGHT = 0      # No extreme penalty - allow any position

# Training schedule
TOTAL_EPISODES = 300000  # More training for exact solution
NUM_ITERATIONS = 12      # More iterations for convergence
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 500000
HIDDEN_DIM = 256

print("="*60)
print("FINANCIAL MATH: EXACT REPLICATION HEDGE RATIOS")
print("="*60)
print("GOAL: Find theoretically correct number of binaries at each node")
print(f"Cost Weight: {COST_WEIGHT} (ZERO - exact hedge regardless of cost)")
print(f"Replication Penalty: {REPLICATION_PENALTY:,} (find exact match)")
print("="*60)

# ============================================================
# TRINOMIAL PROBABILITIES
# ============================================================
def calculate_trinomial_probabilities(S0, u, m, d, r, dt):
    """
    Calculate trinomial probabilities using direct optimization
    to ensure all probabilities are positive
    """
    growth = np.exp(r * dt)
    
    # We need to solve:
    # p_u * u + p_m * m + p_d * d = growth
    # p_u + p_m + p_d = 1
    # p_u, p_m, p_d >= 0
    
    # Use a simple heuristic: minimize variance while matching growth
    from scipy.optimize import minimize
    
    def objective(p):
        # Minimize variance (keep probabilities reasonable)
        return np.sum((p - 1/3)**2)
    
    def constraint_mean(p):
        # Expected growth must equal risk-free rate
        return p[0]*u + p[1]*m + p[2]*d - growth
    
    def constraint_sum(p):
        # Probabilities sum to 1
        return p[0] + p[1] + p[2] - 1
    
    # Initial guess: uniform
    p0 = [1/3, 1/3, 1/3]
    
    # Constraints
    cons = [
        {'type': 'eq', 'fun': constraint_mean},
        {'type': 'eq', 'fun': constraint_sum}
    ]
    
    # Bounds: all probabilities between 0.001 and 0.999
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.999)]
    
    result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons)
    
    if result.success:
        p_u, p_m, p_d = result.x
    else:
        # Fallback: simple heuristic
        print("WARNING: Optimization failed, using fallback method")
        # Set p_m high since m is close to growth
        p_m = 0.7
        # Distribute remainder
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
    
    assert abs(prob_sum - 1.0) < 1e-6, "Probabilities must sum to 1"
    assert abs(expected_growth - growth) < 1e-3, "Expected growth must match risk-free rate"
    assert p_u >= 0 and p_m >= 0 and p_d >= 0, "All probabilities must be positive!"
    
    print("  ✓ Valid risk-neutral probabilities")
    
    return p_u, p_m, p_d

p_u, p_m, p_d = calculate_trinomial_probabilities(S0, u, m, d, r, dt)
# UTILITIES
# ============================================================
# UTILITIES
# ============================================================
def create_polynomial_features(X, degree):
    X = np.array(X).reshape(-1, 1)
    features = np.ones((X.shape[0], degree + 1))
    for d_ in range(1, degree + 1):
        features[:, d_] = (X[:, 0] ** d_)
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
                    'child_occurred': None
                }
                
                if t < self.T_steps:
                    rand = np.random.random()
                    if rand < self.p_u:
                        S *= self.u
                        path_step['child_occurred'] = 0
                    elif rand < self.p_u + self.p_m:
                        S *= self.m
                        path_step['child_occurred'] = 1
                    else:
                        S *= self.d
                        path_step['child_occurred'] = 2
                
                path.append(path_step)
            paths.append(path)
        return paths

# ============================================================
# LSMC ESTIMATOR
# ============================================================
class LSMCEstimator:
    def __init__(self, polynomial_degree=3, alpha=0.1):
        self.poly_degree = polynomial_degree
        self.alpha = alpha
        self.continuation_models = {}
        self.T_steps = T_steps
    
    def estimate_continuation_values(self, paths, r, dt):
        print("\n" + "="*60 + "\nRUNNING LSMC ESTIMATION\n" + "="*60)
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
        
        print("LSMC ESTIMATION COMPLETE\n" + "="*60)
        return paths
    
    def predict_continuation_value(self, S, t):
        if t not in self.continuation_models: 
            return 0.0
        X = create_polynomial_features([S], self.poly_degree)
        return self.continuation_models[t].predict(X)[0]
    
    def predict_child_continuation_values(self, S, t):
        if t >= self.T_steps - 1:
            return [max(S * u - K, 0), max(S * m - K, 0), max(S * d - K, 0)]
        else:
            return [self.predict_continuation_value(S * move, t + 1) for move in [u, m, d]]

# ============================================================
# DDPG AGENT
# ============================================================
class UniversalActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(UniversalActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x)) * ACTION_SCALE
        return x

class UniversalCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(UniversalCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.action_dim, self.mu, self.theta, self.sigma = action_dim, mu, theta, sigma
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
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

class UniversalDDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim):
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
            return
        
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

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
# REWARD: ABSOLUTE DEVIATION (EXACT MATCH REQUIRED)
# ============================================================
def construct_state(S, t, target_values):
    is_intermediate = isinstance(target_values, (list, np.ndarray))
    norm_factor = max_terminal_payoff if max_terminal_payoff > 0 else 1.0
    
    state_vec = np.zeros(6)
    state_vec[0] = S / S0
    state_vec[2] = t / T_steps
    
    if is_intermediate:
        state_vec[1] = np.mean(target_values) / norm_factor
        state_vec[3:6] = np.array(target_values) / norm_factor
    else:
        state_vec[1] = target_values / norm_factor
    
    return state_vec

def compute_reward(hedge, target_values, binary_prices, is_terminal, child_occurred=None):
    """
    FINANCIAL MATH: Find exact hedge ratios (ABSOLUTE deviation)
    
    Goal: hedge should EXACTLY equal target_values
    Cost: ZERO weight (don't care about cost)
    Penalty: MASSIVE for any deviation
    """
    hedge = np.atleast_1d(hedge)
    
    # NO COST PENALTY (financial math goal: exact hedge regardless of cost)
    cost = np.sum(hedge * binary_prices)
    
    if is_terminal:
        # Terminal: hedge[0] should equal target exactly
        deviation = abs(hedge[0] - target_values)
    else:
        if child_occurred is not None:
            # Training: focus on specific child
            deviation = abs(hedge[child_occurred] - target_values[child_occurred])
        else:
            # Evaluation: check all 3
            deviations = [abs(hedge[i] - target_values[i]) for i in range(3)]
            deviation = np.mean(deviations)
    
    # ABSOLUTE DEVIATION (not normalized, not relative)
    # We want: |hedge - target| → 0 (exact match)
    
    # Reward: Pure accuracy penalty (no cost, no extreme penalty)
    reward = -REPLICATION_PENALTY * deviation**2
    
    return np.clip(reward, -1000000, 0), cost, deviation

# ============================================================
# TRAINING LOOP
# ============================================================
def train_exact_replication_agent():
    simulator = TrinomialPathSimulator(S0, u, m, d, p_u, p_m, p_d, T_steps, dt)
    lsmc_estimator = LSMCEstimator(polynomial_degree=POLYNOMIAL_DEGREE, alpha=REGRESSION_ALPHA)
    agent = UniversalDDPGAgent(state_dim=6, action_dim=3, hidden_dim=HIDDEN_DIM)
    
    best_avg_deviation = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}\nITERATION {iteration + 1}/{NUM_ITERATIONS}\n{'='*60}")
        
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        print(f"\nTraining to find EXACT hedge ratios...")
        episodes_this_iter = TOTAL_EPISODES // NUM_ITERATIONS
        
        reward_history = []
        
        for episode in range(episodes_this_iter):
            path_idx = np.random.randint(len(paths))
            time_idx = np.random.randint(T_steps + 1)
            
            sampled_node = paths[path_idx][time_idx]
            S, t = sampled_node['S'], sampled_node['t']
            is_terminal = (t == T_steps)
            
            if is_terminal:
                target = sampled_node['payoff']
                prices = [np.exp(-r * dt)]
                child_occurred = None
            else:
                target = lsmc_estimator.predict_child_continuation_values(S, t)
                prices = [np.exp(-r * dt) * p for p in [p_u, p_m, p_d]]
                child_occurred = sampled_node['child_occurred']
            
            state = construct_state(S, t, target)
            
            noise_decay = max(0.01, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.95
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            action_used = action[:1] if is_terminal else action[:3]
            
            # Multi-child training
            if is_terminal:
                reward, _, _ = compute_reward(action_used, target, prices, is_terminal, None)
                agent.replay_buffer.push(state, action, reward, state, False)
                reward_history.append(reward)
            else:
                for child_idx in range(3):
                    reward, _, _ = compute_reward(action_used, target, prices, False, child_idx)
                    agent.replay_buffer.push(state, action, reward, state, False)
                    reward_history.append(reward)
            
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            if (episode + 1) % (episodes_this_iter // 8) == 0:
                avg_reward = np.mean(reward_history[-1000:]) if len(reward_history) >= 1000 else np.mean(reward_history)
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.1f}")

        print(f"\nEvaluating hedge accuracy...")
        
        total_deviation, num_evals = 0, 0
        
        for path in paths[:1000]:
            for node in path:
                S_eval, t_eval = node['S'], node['t']
                is_terminal_eval = (t_eval == T_steps)
                
                if is_terminal_eval:
                    target_eval = node['payoff']
                    prices_eval = [np.exp(-r * dt)]
                    child_eval = None
                else:
                    target_eval = lsmc_estimator.predict_child_continuation_values(S_eval, t_eval)
                    prices_eval = [np.exp(-r * dt) * p for p in [p_u, p_m, p_d]]
                    child_eval = node['child_occurred']

                state_eval = construct_state(S_eval, t_eval, target_eval)
                action_eval = agent.select_action(state_eval, add_noise=False)
                action_used_eval = action_eval[:1] if is_terminal_eval else action_eval[:3]
                
                _, _, deviation = compute_reward(action_used_eval, target_eval, prices_eval, is_terminal_eval, child_eval)
                
                total_deviation += deviation
                num_evals += 1

        avg_deviation = total_deviation / num_evals
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Average ABSOLUTE Deviation: ${avg_deviation:.4f}")
        
        if avg_deviation < best_avg_deviation:
            best_avg_deviation = avg_deviation
            print(f"  ✓ NEW BEST!")
        
        if avg_deviation < 0.5:
            print(f"\n🎉 EXCELLENT! Avg Deviation < $0.50 at iteration {iteration + 1}")
            if avg_deviation < 0.1:
                print(f"🎯 OUTSTANDING! Within $0.10 - essentially exact!")
                break
            
    print(f"\n{'='*60}\nTRAINING COMPLETE!\n{'='*60}")
    print(f"Best Average Absolute Deviation: ${best_avg_deviation:.4f}")
    return agent, lsmc_estimator

# ============================================================
# MAIN EXECUTION
# ============================================================
agent, lsmc_estimator = train_exact_replication_agent()

# ============================================================
# FINAL EVALUATION - FINANCIAL MATH VALIDATION
# ============================================================
print("\n" + "="*60 + "\nFINAL EVALUATION - HEDGE RATIOS\n" + "="*60)

simulator = TrinomialPathSimulator(S0, u, m, d, p_u, p_m, p_d, T_steps, dt)
paths = simulator.simulate_paths(1000)

unique_states = set()
for path in paths:
    for node in path:
        unique_states.add((node['S'], node['t']))

unique_states = sorted(list(unique_states), key=lambda x: (x[1], -x[0]))

print(f"\nEvaluating {len(unique_states)} unique states")
print("="*60)

success_count = 0
total_tests = len(unique_states)

for S, t in unique_states:
    is_terminal = (t == T_steps)
    
    if is_terminal:
        target = max(S - K, 0)
        prices = [np.exp(-r * dt)]
    else:
        target = lsmc_estimator.predict_child_continuation_values(S, t)
        prices = [np.exp(-r * dt) * p for p in [p_u, p_m, p_d]]
    
    state = construct_state(S, t, target)
    action = agent.select_action(state, add_noise=False)
    action_used = action[:1] if is_terminal else action[:3]
    
    _, cost, deviation = compute_reward(action_used, target, prices, is_terminal, child_occurred=None)
    
    is_success = deviation < 0.5  # Within $0.50
    if is_success:
        success_count += 1
    
    # Print results
    print(f"\nt={t} | S=${S:.2f}")
    print("-"*60)
    
    if is_terminal:
        print(f"LSMC Target (continuation value): ${target:.4f}")
        print(f"RL Hedge (# of binaries):          {action_used[0]:+.4f}")
        print(f"Absolute Deviation:                ${deviation:.4f} {'✓' if is_success else '✗'}")
        print(f"Total Cost:                        ${cost:+.4f}")
    else:
        print(f"LSMC Targets (continuation values):")
        print(f"  Up   : ${target[0]:.4f}")
        print(f"  Mid  : ${target[1]:.4f}")
        print(f"  Down : ${target[2]:.4f}")
        print("-"*60)
        print(f"RL Hedge (# of binaries to hold):")
        print(f"  Up   : {action_used[0]:+.4f}")
        print(f"  Mid  : {action_used[1]:+.4f}")
        print(f"  Down : {action_used[2]:+.4f}")
        print("-"*60)
        print(f"Absolute Deviations:")
        dev_up = abs(action_used[0] - target[0])
        dev_mid = abs(action_used[1] - target[1])
        dev_down = abs(action_used[2] - target[2])
        print(f"  Up   : ${dev_up:.4f}")
        print(f"  Mid  : ${dev_mid:.4f}")
        print(f"  Down : ${dev_down:.4f}")
        print(f"Average: ${deviation:.4f} {'✓' if is_success else '✗'}")
        print(f"Total Cost: ${cost:+.4f}")
    
    print("-"*60)
    if is_success:
        print("✓ Hedge ratios match LSMC targets (within $0.50)")
    else:
        print("✗ Hedge ratios deviate from LSMC targets")
    print("="*60)

print("\n" + "="*60)
print(f"SUCCESS RATE: {success_count}/{total_tests} ({100*success_count/total_tests:.1f}%)")
print("="*60)
print("\nFINANCIAL MATH VALIDATION:")
print(f"• Complete market: 3 states, 3 binaries")
print(f"• Theoretical result: Hedge = LSMC continuation values")
print(f"• Success criterion: Within $0.50 of theoretical hedge")
print("="*60)
print("\nINTERPRETATION:")
print("The numbers shown represent the EXACT number of binary contracts")
print("to purchase at each node to replicate the option's continuation value.")
print("In complete markets, these should match the LSMC targets closely.")
print("="*60)