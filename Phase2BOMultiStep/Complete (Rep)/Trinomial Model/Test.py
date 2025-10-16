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
dt = 1.0
sigma = 0.2
u = np.exp(sigma * np.sqrt(2 * dt))   # ≈ 1.3499
m = 1.0                                
d = np.exp(-sigma * np.sqrt(2 * dt))  # ≈ 0.7408
r = 0.05
K = 100.0
T_steps = 2

# LSMC Parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# ============================================================
# AGGRESSIVE HYPERPARAMETERS FOR COMPLETE MARKET
# ============================================================
ACTOR_LR = 0.0001      # Increased from 0.00005
CRITIC_LR = 0.0003     # Increased from 0.00015

max_stock_price = S0 * (u ** T_steps)
max_terminal_payoff = max(max_stock_price - K, 0)
ACTION_SCALE = max_terminal_payoff * 3.0  # Increased from 2.0 to allow larger hedges

# CRITICAL: Make accuracy MUCH more important than cost
REPLICATION_PENALTY = 10000  # Increased from 800 - THIS IS KEY!
EXTREME_PENALTY_WEIGHT = 100  # Increased from 30
COST_WEIGHT = 0.0001          # Decreased from 0.01 - cost barely matters!

# More training
TOTAL_EPISODES = 300000  # Increased from 240000
NUM_ITERATIONS = 12      # Increased from 10
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 300000  # Increased
HIDDEN_DIM = 256      # Increased from 128 for more capacity

print("="*60)
print(f"LSMC + DDPG: TRINOMIAL T={T_steps}")
print(f"MODE: COMPLETE MARKET - AGGRESSIVE ACCURACY PRIORITY")
print("="*60)
print("CRITICAL CHANGES:")
print(f"  → Replication Penalty: {REPLICATION_PENALTY} (VERY HIGH)")
print(f"  → Cost Weight: {COST_WEIGHT} (VERY LOW)")
print(f"  → Accuracy/Cost Ratio: {REPLICATION_PENALTY/COST_WEIGHT:,.0f}:1")
print(f"  → Extreme Penalty: {EXTREME_PENALTY_WEIGHT}")
print(f"  → Action Scale: ±{ACTION_SCALE:.1f}")
print(f"  → Hidden Dim: {HIDDEN_DIM}, Episodes: {TOTAL_EPISODES:,}")
print("="*60)

# ============================================================
# TRINOMIAL PROBABILITIES
# ============================================================
def calculate_trinomial_probabilities(S0, u, m, d, r, dt):
    growth = np.exp(r * dt)
    p_d = (growth - u) * (growth - m) / ((d - u) * (d - m))
    p_u = (growth - m) * (growth - d) / ((u - m) * (u - d))
    p_m = 1.0 - p_u - p_d
    
    expected_growth = p_u * u + p_m * m + p_d * d
    prob_sum = p_u + p_m + p_d
    
    print(f"\nTrinomial Probabilities:")
    print(f"  p_u = {p_u:.6f}, p_m = {p_m:.6f}, p_d = {p_d:.6f}")
    print(f"  Sum = {prob_sum:.6f}, Expected growth = {expected_growth:.6f} (target: {growth:.6f})")
    
    assert abs(prob_sum - 1.0) < 1e-6
    assert abs(expected_growth - growth) < 1e-4
    
    if p_u < 0 or p_m < 0 or p_d < 0:
        print("WARNING: Negative probabilities! Adjusting...")
        p_u = max(0.01, p_u)
        p_d = max(0.01, p_d)
        p_m = 1.0 - p_u - p_d
    
    return p_u, p_m, p_d

p_u, p_m, p_d = calculate_trinomial_probabilities(S0, u, m, d, r, dt)

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
# PATH SIMULATION WITH CHILD TRACKING
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
# ENHANCED DDPG AGENT WITH LARGER NETWORKS
# ============================================================
class UniversalActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(UniversalActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Extra layer
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        # Better initialization
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
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Extra layer
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
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):  # Increased sigma
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
# STATE & REWARD - AGGRESSIVE ACCURACY FOCUS
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
    AGGRESSIVE: Accuracy is 100,000x more important than cost!
    """
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    
    if is_terminal:
        deviation = abs(hedge[0] - target_values)
        extreme_penalty = 0
        
        # Extra penalty for high-value terminals
        if target_values > 40:
            penalty_multiplier = 2.0
        else:
            penalty_multiplier = 1.0
    else:
        # CRITICAL: Use child_occurred if available
        if child_occurred is not None:
            # Only the binary that pays off matters!
            deviation = abs(hedge[child_occurred] - target_values[child_occurred])
            penalty_multiplier = 1.0
        else:
            # Evaluation mode: check all 3
            deviations = [abs(hedge[i] - target_values[i]) for i in range(3)]
            deviation = np.mean(deviations)
            penalty_multiplier = 1.0
        
        # Extreme position penalty
        all_targets_near_zero = all(abs(tv) < 0.1 for tv in target_values)
        if all_targets_near_zero:
            max_reasonable = 2.0  # Very strict for zero targets
        else:
            max_reasonable = max([abs(tv) * 3.0 for tv in target_values] + [10.0])
        
        extreme_penalty = sum(max(0, abs(h) - max_reasonable)**2 for h in hedge)
    
    normalized_deviation = deviation / (max_terminal_payoff if max_terminal_payoff > 0 else 1.0)
    
    # CRITICAL: Make accuracy overwhelmingly important
    reward = -(COST_WEIGHT * abs(cost) 
               + penalty_multiplier * REPLICATION_PENALTY * normalized_deviation**2
               + EXTREME_PENALTY_WEIGHT * extreme_penalty)
    
    return np.clip(reward, -100000, 0), cost, deviation

# ============================================================
# TRAINING LOOP
# ============================================================
def train_universal_agent_with_lsmc():
    simulator = TrinomialPathSimulator(S0, u, m, d, p_u, p_m, p_d, T_steps, dt)
    lsmc_estimator = LSMCEstimator(polynomial_degree=POLYNOMIAL_DEGREE, alpha=REGRESSION_ALPHA)
    agent = UniversalDDPGAgent(state_dim=6, action_dim=3, hidden_dim=HIDDEN_DIM)
    
    best_avg_deviation = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}\nITERATION {iteration + 1}/{NUM_ITERATIONS}\n{'='*60}")
        
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        print(f"\nTraining agent with AGGRESSIVE accuracy focus...")
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
            
            # More aggressive noise schedule
            noise_decay = max(0.05, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.9  # Explore longer
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            action_used = action[:1] if is_terminal else action[:3]
            
            reward, _, _ = compute_reward(action_used, target, prices, is_terminal, child_occurred)
            reward_history.append(reward)
            
            agent.replay_buffer.push(state, action, reward, state, False)
            
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            # More frequent updates
            if (episode + 1) % (episodes_this_iter // 8) == 0:
                avg_reward = np.mean(reward_history[-1000:]) if len(reward_history) >= 1000 else np.mean(reward_history)
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.1f}")

        print(f"\nEvaluating on paths with child tracking...")
        
        total_deviation, num_evals = 0, 0
        max_deviation = 0
        deviation_by_type = {'terminal': [], 'intermediate': []}
        
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
                max_deviation = max(max_deviation, deviation)
                num_evals += 1
                
                if is_terminal_eval:
                    deviation_by_type['terminal'].append(deviation)
                else:
                    deviation_by_type['intermediate'].append(deviation)

        avg_deviation = total_deviation / num_evals
        avg_terminal = np.mean(deviation_by_type['terminal']) if deviation_by_type['terminal'] else 0
        avg_intermediate = np.mean(deviation_by_type['intermediate']) if deviation_by_type['intermediate'] else 0
        
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Overall Avg Deviation: {avg_deviation:.6f}")
        print(f"  Terminal Avg Deviation: {avg_terminal:.6f}")
        print(f"  Intermediate Avg Deviation: {avg_intermediate:.6f}")
        print(f"  Max Deviation: {max_deviation:.6f}")
        
        if avg_deviation < best_avg_deviation:
            best_avg_deviation = avg_deviation
            print(f"  ✓ NEW BEST!")
        
        if avg_deviation < 0.5:
            print(f"\n🎉 SUCCESS! Avg Deviation < 0.5 at iteration {iteration + 1}")
            break
            
    print(f"\n{'='*60}\nTRAINING COMPLETE!\n{'='*60}")
    print(f"Best Average Deviation: {best_avg_deviation:.6f}")
    print(f"Target was: < 1.0 for success")
    return agent, lsmc_estimator

# ============================================================
# MAIN EXECUTION
# ============================================================
agent, lsmc_estimator = train_universal_agent_with_lsmc()

# ============================================================
# FINAL EVALUATION
# ============================================================
print("\n" + "="*60 + "\nFINAL EVALUATION\n" + "="*60)

test_states = [(S0, 0)]
test_states.extend([(S0 * u, 1), (S0 * m, 1), (S0 * d, 1)])
if T_steps >= 2:
    test_states.extend([
        (S0 * u * u, 2), (S0 * u * m, 2), (S0 * m * m, 2),
        (S0 * u * d, 2), (S0 * m * d, 2), (S0 * d * d, 2)
    ])

print("\nAgent Performance at Key States:")
print("-" * 100)

success_count = 0
total_tests = 0

for S, t in sorted(list(set(test_states)), key=lambda x: (x[1], -x[0])):
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
    
    is_success = deviation < 1.0
    if is_success:
        success_count += 1
    total_tests += 1
    
    print(f"\nState: S=${S:.2f}, t={t}")
    if is_terminal:
        print(f"  Target: ${target:.4f}, Hedge: ${action_used[0]:.4f}")
        print(f"  Cost: ${cost:.4f}, Deviation: {deviation:.6f} {'✓' if is_success else '✗'}")
    else:
        print(f"  Targets: [Up=${target[0]:.4f}, Mid=${target[1]:.4f}, Down=${target[2]:.4f}]")
        print(f"  Hedges: [{action_used[0]:.4f}, {action_used[1]:.4f}, {action_used[2]:.4f}]")
        print(f"  Cost: ${cost:.4f}")
        
        dev_up = abs(action_used[0] - target[0])
        dev_mid = abs(action_used[1] - target[1])
        dev_down = abs(action_used[2] - target[2])
        
        print(f"  Individual Devs: Up={dev_up:.4f}, Mid={dev_mid:.4f}, Down={dev_down:.4f}")
        print(f"  Average Deviation: {deviation:.6f} {'✓' if is_success else '✗'}")

print("\n" + "="*60)
print(f"SUCCESS RATE: {success_count}/{total_tests} ({100*success_count/total_tests:.1f}%)")
print("="*60)
print("AGGRESSIVE COMPLETE MARKET APPROACH:")
print(f"  • Accuracy/Cost Ratio: {REPLICATION_PENALTY/COST_WEIGHT:,.0f}:1")
print(f"  • Total Training: {TOTAL_EPISODES:,} episodes")
print(f"  • Network Capacity: {HIDDEN_DIM} hidden units, 4 layers")
print("  • Child tracking for correct complete market structure")
print("="*60)