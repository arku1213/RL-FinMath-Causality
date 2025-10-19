import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION - N-NOMIAL SUPER-REPLICATION
# ============================================================
S0 = 100.0
r = 0.05
K = 100.0
T_steps = 2
dt = 1.0

# N-NOMIAL PARAMETERS (Scalable!)
N = 5  # Number of branches
sigma = 0.3
lambda_param = np.sqrt(N - 1)  # Scales with N

# Generate N symmetric moves
moves = []
for i in range(N):
    ratio = (i - (N-1)/2) / ((N-1)/2)  # -1 to +1
    move = np.exp(ratio * lambda_param * sigma * np.sqrt(dt))
    moves.append(move)

moves = sorted(moves, reverse=True)  # [largest_up, ..., middle, ..., largest_down]

# LSMC parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# SUPER-REPLICATION: Conservative LSMC
CONSERVATISM_FACTOR = 1.10  # 10% buffer

# RL hyperparameters - SCALED FOR N=5
BASE_ACTOR_LR = 0.00003 * (3/N)**0.5
BASE_CRITIC_LR = 0.00010 * (3/N)**0.5
ACTOR_LR = BASE_ACTOR_LR
CRITIC_LR = BASE_CRITIC_LR

max_stock_price = S0 * (moves[0] ** T_steps)
max_terminal_payoff = max(max_stock_price - K, 0)
ACTION_SCALE = max_terminal_payoff * 1.2

# SUPER-REPLICATION PENALTIES (Scaled with N)
SHORTFALL_PENALTY = int(10000000 * (N/3))
COST_WEIGHT = int(200000 * (N/3)**0.5)

# Training configuration (Scaled with N)
base_episodes = 800000
TOTAL_EPISODES = int(base_episodes * (N/3)**0.8)
NUM_ITERATIONS = 16
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.003
BUFFER_SIZE = int(800000 * (N/3)**0.5)
HIDDEN_DIM = 256 + 64 * (N - 3)

# Improvement parameters
LR_DECAY = 0.93
NOISE_DECAY = 0.75
EARLY_STOP_PATIENCE = 4

print("=" * 60)
print(f"{N}-NOMIAL COMPLETE SUPER-REPLICATION (T={T_steps})")
print("=" * 60)
print("METHOD: Pure RL discovers MINIMAL-cost super-replicating hedges")
print(f"Market: {N} states → {N} binaries (COMPLETE)")
print("=" * 60)
print("GOAL: Find MINIMAL hedge h where h ≥ LSMC_conservative")
print("      AND minimize cost(h) = sum(|h_i|)")
print("=" * 60)
print(f"SHORTFALL_PENALTY: {SHORTFALL_PENALTY:,} (scaled with N)")
print(f"COST_WEIGHT: {COST_WEIGHT:,} (balanced for N={N})")
print(f"CONSERVATISM: {CONSERVATISM_FACTOR}× LSMC (safety buffer)")
print(f"Penalty Ratio: {SHORTFALL_PENALTY/COST_WEIGHT:.0f}:1")
print("=" * 60)
print(f"SCALING INFO:")
print(f"  Moves: {[f'{m:.4f}' for m in moves]}")
print(f"  Total Episodes: {TOTAL_EPISODES:,}")
print(f"  Episodes/Iter: {TOTAL_EPISODES//NUM_ITERATIONS:,}")
print(f"  Buffer Size: {BUFFER_SIZE:,}")
print(f"  Hidden Dim: {HIDDEN_DIM}")
print("=" * 60)


# ============================================================
# N-NOMIAL PROBABILITIES
# ============================================================
def calculate_nnomial_probabilities(moves, r, dt):
    """Calculate risk-neutral probabilities for N-nomial model."""
    N = len(moves)
    growth = np.exp(r * dt)
    
    def objective(p):
        return np.sum((p - 1/N)**2)
    
    def constraint_mean(p):
        return np.sum(p * np.array(moves)) - growth
    
    def constraint_sum(p):
        return np.sum(p) - 1
    
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
    
    prob_sum = np.sum(probs)
    expected_growth = np.sum(probs * np.array(moves))
    
    print(f"\n{N}-nomial Probabilities:")
    for i, p in enumerate(probs):
        print(f"  p[{i}] = {p:.6f}")
    print(f"  Sum = {prob_sum:.6f}, Expected growth = {expected_growth:.6f}")
    
    assert abs(prob_sum - 1.0) < 1e-6
    assert abs(expected_growth - growth) < 1e-2
    assert all(p >= 0 for p in probs)
    
    print("  ✓ Valid risk-neutral probabilities")
    return probs


probs = calculate_nnomial_probabilities(moves, r, dt)


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
    def __init__(self, S0, moves, probs, T_steps, dt):
        self.S0 = S0
        self.moves = moves
        self.probs = probs
        self.N = len(moves)
        self.T_steps = T_steps
        self.dt = dt
    
    def simulate_paths(self, num_paths):
        paths = []
        for _ in range(num_paths):
            path = []
            S = self.S0
            path_history = []
            
            for t in range(self.T_steps + 1):
                path_step = {
                    'S': S,
                    't': t,
                    'payoff': max(S - K, 0) if t == self.T_steps else None,
                    'child_occurred': None,
                    'path_history': path_history.copy()
                }
                
                if t < self.T_steps:
                    child_idx = np.random.choice(self.N, p=self.probs)
                    S *= self.moves[child_idx]
                    path_step['child_occurred'] = child_idx
                    path_history.append(child_idx)
                
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
        print("RUNNING LSMC ESTIMATION (CONSERVATIVE)")
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
        print(f"Conservative buffer: {self.conservatism}× for safety")
        print("(Conservative LSMC used in REWARD only - not in state!)")
        print("=" * 60)
        return paths
    
    def predict_continuation_value(self, S, t):
        if t not in self.continuation_models:
            return 0.0
        X = create_polynomial_features([S], self.poly_degree)
        return self.continuation_models[t].predict(X)[0]
    
    def predict_child_continuation_values(self, S, t, conservative=True):
        """Returns N continuation values."""
        if t >= self.T_steps - 1:
            return [max(S * move - K, 0) for move in moves]
        else:
            base_values = [self.predict_continuation_value(S * move, t + 1) for move in moves]
            if conservative:
                return [max(0, v * self.conservatism) for v in base_values]
            else:
                return [max(0, v) for v in base_values]


# ============================================================
# NEURAL NETWORKS
# ============================================================
class UniversalActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(UniversalActor, self).__init__()
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
# STATE AND REWARD - FINANCIAL MATH FOCUS
# ============================================================
def construct_state(S, t, path_history):
    """Pure RL state - NO LSMC values!"""
    state_vec = np.zeros(4)
    state_vec[0] = S / S0
    state_vec[1] = t / T_steps
    
    # Path encoding: normalize to [-1, +1]
    if len(path_history) >= 1:
        state_vec[2] = 1.0 - 2.0 * path_history[-1] / (N - 1)
    if len(path_history) >= 2:
        state_vec[3] = 1.0 - 2.0 * path_history[-2] / (N - 1)
    
    return state_vec


def compute_reward_super_replication(hedge, S, t, lsmc_estimator, is_terminal):
    """
    Financial Math Goal: h ≥ C_conservative, minimize cost
    """
    hedge = np.atleast_1d(hedge)
    
    if is_terminal:
        target = max(S - K, 0)
        shortfall = max(0, target - hedge[0])
        cost = abs(hedge[0])
    else:
        conservative_targets = lsmc_estimator.predict_child_continuation_values(S, t, conservative=True)
        shortfalls = [max(0, conservative_targets[i] - hedge[i]) for i in range(N)]
        shortfall = np.mean(shortfalls)
        cost = np.sum(np.abs(hedge))
    
    reward = -SHORTFALL_PENALTY * shortfall**2 - COST_WEIGHT * cost
    return np.clip(reward, -100000000, 0), shortfall, cost


# ============================================================
# TRAINING LOOP
# ============================================================
def train_super_replication():
    simulator = NnomialPathSimulator(S0, moves, probs, T_steps, dt)
    lsmc_estimator = LSMCEstimator(
        polynomial_degree=POLYNOMIAL_DEGREE,
        alpha=REGRESSION_ALPHA,
        conservatism=CONSERVATISM_FACTOR
    )
    
    agent = UniversalDDPGAgent(state_dim=4, action_dim=N, hidden_dim=HIDDEN_DIM)
    reward_normalizer = RewardNormalizer(clip_range=10.0)
    
    best_avg_shortfall = float('inf')
    best_actor_state = None
    patience_counter = 0
    
    for iteration in range(NUM_ITERATIONS):
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
        
        print(f"\nTraining super-replication policy...")
        episodes_this_iter = TOTAL_EPISODES // NUM_ITERATIONS
        reward_history = []
        
        for episode in range(episodes_this_iter):
            path_idx = np.random.randint(len(paths))
            time_idx = np.random.randint(T_steps + 1)
            
            sampled_node = paths[path_idx][time_idx]
            S, t = sampled_node['S'], sampled_node['t']
            path_history = sampled_node['path_history']
            is_terminal = (t == T_steps)
            
            state = construct_state(S, t, path_history)
            
            noise_decay = max(0.1, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.98
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            action_used = action[:1] if is_terminal else action[:N]
            
            reward, shortfall, cost = compute_reward_super_replication(
                action_used, S, t, lsmc_estimator, is_terminal
            )
            
            reward_normalizer.update(reward)
            normalized_reward = reward_normalizer.normalize(reward)
            
            agent.replay_buffer.push(state, action, normalized_reward, state, False)
            reward_history.append(reward)
            
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            if (episode + 1) % (episodes_this_iter // 10) == 0:
                recent_rewards = reward_history[-1000:] if len(reward_history) >= 1000 else reward_history
                avg_reward = np.mean(recent_rewards)
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.1f}")
        
        print(f"\nEvaluating super-replication hedges...")
        total_shortfall, total_cost, num_evals = 0, 0, 0
        
        for path in paths[:1000]:
            for node in path:
                S_eval, t_eval = node['S'], node['t']
                path_history_eval = node['path_history']
                is_terminal_eval = (t_eval == T_steps)
                
                state_eval = construct_state(S_eval, t_eval, path_history_eval)
                action_eval = agent.select_action(state_eval, add_noise=False)
                action_used_eval = action_eval[:1] if is_terminal_eval else action_eval[:N]
                
                _, shortfall, cost = compute_reward_super_replication(
                    action_used_eval, S_eval, t_eval, lsmc_estimator, is_terminal_eval
                )
                
                total_shortfall += shortfall
                total_cost += cost
                num_evals += 1
        
        avg_shortfall = total_shortfall / num_evals
        avg_cost = total_cost / num_evals
        
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Average Shortfall: ${avg_shortfall:.4f} (target: $0.00)")
        print(f"  Average Cost: ${avg_cost:.2f}")
        
        if avg_shortfall < best_avg_shortfall:
            best_avg_shortfall = avg_shortfall
            best_actor_state = agent.actor.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ NEW BEST! Saving model...")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{EARLY_STOP_PATIENCE})")
        
        if patience_counter >= EARLY_STOP_PATIENCE and iteration >= 8:
            print(f"\n✅ EARLY STOPPING!")
            agent.actor.load_state_dict(best_actor_state)
            break
        
        if avg_shortfall > 100 and iteration >= 4:
            print(f"\n⚠️  Training not improving")
            if best_actor_state is not None:
                agent.actor.load_state_dict(best_actor_state)
            break
        
        if avg_shortfall < 0.5:
            print(f"\n🎯 EXCELLENT! Shortfall < $0.50!")
            break
    
    print(f"\n{'='*60}\nTRAINING COMPLETE!\n{'='*60}")
    print(f"Best Average Shortfall: ${best_avg_shortfall:.4f}")
    
    if best_actor_state is not None:
        agent.actor.load_state_dict(best_actor_state)
        print("Restored best model for final evaluation")
    
    return agent, lsmc_estimator


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    agent, lsmc_estimator = train_super_replication()
    
    # ============================================================
    # FINAL EVALUATION - ALL 31 NODES (FINANCIAL MATH POV)
    # ============================================================
    print("\n" + "="*60)
    print(f"FINAL EVALUATION - {N}-NOMIAL SUPER-REPLICATION")
    print("="*60)
    print("Financial Math Goal: Find minimal h where h ≥ C_conservative")
    print("="*60)
    
    # Generate all 31 unique nodes (1 root + 5 t=1 + 25 t=2)
    test_cases = []
    
    # Root
    test_cases.append((S0, 0, []))
    
    # All t=1 nodes
    for i in range(N):
        S = S0 * moves[i]
        test_cases.append((S, 1, [i]))
    
    # All t=2 nodes
    for i in range(N):
        for j in range(N):
            S = S0 * moves[i] * moves[j]
            test_cases.append((S, 2, [i, j]))
    
    print(f"\nEvaluating all {len(test_cases)} unique nodes")
    print("="*60)
    
    nodes_dominated = 0
    total_shortfall = 0
    total_cost = 0
    
    for idx, (S, t, path_history) in enumerate(test_cases):
        is_terminal = (t == T_steps)
        
        # Get conservative LSMC targets
        if is_terminal:
            conservative_targets = max(S - K, 0)
        else:
            conservative_targets = lsmc_estimator.predict_child_continuation_values(S, t, conservative=True)
        
        # RL-discovered hedge
        state = construct_state(S, t, path_history)
        action = agent.select_action(state, add_noise=False)
        hedge = action[:1] if is_terminal else action[:N]
        
        # Check dominance
        if is_terminal:
            shortfall = max(0, conservative_targets - hedge[0])
            cost = abs(hedge[0])
            dominates = (shortfall < 0.01)  # Allow tiny numerical errors
        else:
            shortfalls = [max(0, conservative_targets[i] - hedge[i]) for i in range(N)]
            shortfall = np.mean(shortfalls)
            cost = np.sum(np.abs(hedge))
            dominates = all(s < 0.01 for s in shortfalls)
        
        if dominates:
            nodes_dominated += 1
        total_shortfall += shortfall
        total_cost += cost
        
        # Print detailed results for key nodes (root, t=1, and sample of t=2)
        should_print = (t <= 1) or (idx % 5 == 0)  # Print all non-terminal + every 5th terminal
        
        if should_print:
            path_str = "→".join([str(p) for p in path_history]) if path_history else "ROOT"
            
            print(f"\nNode {idx+1}/{len(test_cases)}: t={t}, S=${S:.2f}, Path: {path_str}")
            print("-"*60)
            
            if is_terminal:
                print(f"Conservative Target: ${conservative_targets:.4f}")
                print(f"RL Hedge: {hedge[0]:+.4f}")
                print(f"Shortfall: ${shortfall:.4f}")
                print(f"Cost: ${cost:.2f}")
                print(f"Dominates: {'YES ✓' if dominates else 'NO ✗'}")
            else:
                print(f"Conservative Targets (C×{CONSERVATISM_FACTOR}):")
                print(f"  {[f'${v:.2f}' for v in conservative_targets]}")
                print(f"RL Hedge:")
                print(f"  {[f'{h:+.2f}' for h in hedge]}")
                
                shortfalls_list = [max(0, conservative_targets[i] - hedge[i]) for i in range(N)]
                print(f"Shortfalls:")
                print(f"  {[f'${s:.2f}' for s in shortfalls_list]}")
                
                print(f"Avg Shortfall: ${shortfall:.4f}")
                print(f"Total Cost: ${cost:.2f}")
                print(f"Dominates: {'YES ✓' if dominates else 'NO ✗'}")
            print("="*60)
    
    # Summary
    print("\n" + "="*60)
    print(f"{N}-NOMIAL SUPER-REPLICATION SUMMARY")
    print("="*60)
    print(f"Nodes with Complete Dominance: {nodes_dominated}/{len(test_cases)}")
    print(f"Dominance Rate: {100*nodes_dominated/len(test_cases):.1f}%")
    print(f"Total Shortfall (all nodes): ${total_shortfall:.4f}")
    print(f"Average Cost per Node: ${total_cost/len(test_cases):.2f}")
    print("="*60)
    
    if nodes_dominated == len(test_cases) and total_cost/len(test_cases) < 80:
        print("\n🎯 EXCELLENT! Minimal super-replication achieved!")
        print(f"RL discovered efficient hedges for N={N} with minimal excess.")
        print(f"Average cost ${total_cost/len(test_cases):.2f}/node is near-optimal!")
    elif nodes_dominated == len(test_cases) and total_cost/len(test_cases) < 120:
        print(f"\n✓ VERY GOOD! All {len(test_cases)} nodes dominated.")
        print(f"Average cost: ${total_cost/len(test_cases):.2f}/node")
        print("Cost is reasonable for N=5 super-replication.")
    elif nodes_dominated >= 0.85 * len(test_cases):
        print(f"\n✓ GOOD: {nodes_dominated}/{len(test_cases)} nodes dominated ({100*nodes_dominated/len(test_cases):.1f}%)")
        print(f"Average cost: ${total_cost/len(test_cases):.2f}/node")
        print("Most nodes achieve super-replication with reasonable efficiency.")
    elif nodes_dominated >= 0.70 * len(test_cases):
        print(f"\n⚠️  PARTIAL: {nodes_dominated}/{len(test_cases)} nodes dominated ({100*nodes_dominated/len(test_cases):.1f}%)")
        print(f"Average cost: ${total_cost/len(test_cases):.2f}/node")
        print("Consider adjusting COST_WEIGHT for better dominance/cost balance.")
    else:
        print(f"\n✗ NEEDS IMPROVEMENT: Only {nodes_dominated}/{len(test_cases)} nodes dominated")
        print(f"Average cost: ${total_cost/len(test_cases):.2f}/node")
        print("Consider reducing COST_WEIGHT to prioritize dominance.")
    
    print("="*60)
    print("\nFINANCIAL MATH INTERPRETATION:")
    print(f"• RL discovered hedge positions h_i at each of {len(test_cases)} nodes")
    print(f"• N={N} children per non-terminal node → {N} hedge positions needed")
    print(f"• Conservative LSMC includes {(CONSERVATISM_FACTOR-1)*100:.0f}% safety buffer")
    print(f"• Dominance: h_i ≥ C_conservative for all {N} children")
    print(f"• Cost minimization: Minimize Σ|h_i| subject to dominance")
    print(f"• Penalty ratio: {SHORTFALL_PENALTY/COST_WEIGHT:.0f}:1 (shortfall:cost)")
    print("• Pure RL: No LSMC values in state, learned through rewards only")
    print("="*60)
    print(f"\n✓ {N}-NOMIAL SUPER-REPLICATION COMPLETE")
    print("="*60)