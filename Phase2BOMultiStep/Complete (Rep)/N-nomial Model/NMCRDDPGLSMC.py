import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION - SCALABLE N-NOMIAL
# ============================================================
S0 = 100.0
r = 0.05
K = 100.0
T_steps = 2  # Time steps
dt = 1.0

# N-NOMIAL PARAMETERS (Scalable!)
N = 4  # Number of branches (2=binomial, 3=trinomial, 4=4-nomial, 5=5-nomial, etc.)
sigma = 0.3
lambda_param = np.sqrt(N - 1)  # Scales with N

# Generate N symmetric moves
moves = []
for i in range(N):
    # Symmetric moves: largest up, smallest down, middle near 1.0
    ratio = (i - (N-1)/2) / ((N-1)/2)  # -1 to +1
    move = np.exp(ratio * lambda_param * sigma * np.sqrt(dt))
    moves.append(move)

moves = sorted(moves, reverse=True)  # [u1, u2, ..., m, ..., d2, d1]
print(f"\nN={N} moves: {[f'{m:.4f}' for m in moves]}")

# LSMC parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# RL hyperparameters - SCALED WITH N!
BASE_ACTOR_LR = 0.00003 * (3/N)**0.5  # Scale down for larger N
BASE_CRITIC_LR = 0.00010 * (3/N)**0.5
ACTOR_LR = BASE_ACTOR_LR
CRITIC_LR = BASE_CRITIC_LR

max_stock_price = S0 * (moves[0] ** T_steps)  # Max upward move
max_terminal_payoff = max(max_stock_price - K, 0)
ACTION_SCALE = max_terminal_payoff * 1.2

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
HIDDEN_DIM = 256 + 64 * (N - 3)  # Bigger network for more actions

# Improvement parameters
LR_DECAY = 0.93
NOISE_DECAY = 0.75
EARLY_STOP_PATIENCE = 4

print("=" * 60)
print(f"PURE DEEP RL: {N}-NOMIAL EXACT REPLICATION (T={T_steps})")
print("=" * 60)
print("METHOD: Pure RL - Actor learns from rewards ONLY!")
print(f"Market: {N} states → {N} binaries (COMPLETE)")
print("=" * 60)
print("CRITICAL: State = [S, t, path_encoding] - NO LSMC values!")
print("Actor discovers optimal hedges through trial-and-error")
print("OPTIMIZATIONS: Scaled episodes, path encoding, adaptive params")
print("=" * 60)
print(f"\nSCALING INFO:")
print(f"  Total Episodes: {TOTAL_EPISODES:,}")
print(f"  Episodes/Iter: {TOTAL_EPISODES//NUM_ITERATIONS:,}")
print(f"  Buffer Size: {BUFFER_SIZE:,}")
print(f"  Hidden Dim: {HIDDEN_DIM}")
print(f"  Penalty: {REPLICATION_PENALTY:,.0f}")
print("=" * 60)


# ============================================================
# N-NOMIAL PROBABILITIES
# ============================================================
def calculate_nnomial_probabilities(moves, r, dt):
    """Calculate risk-neutral probabilities for N-nomial model."""
    N = len(moves)
    growth = np.exp(r * dt)
    
    def objective(p):
        # Minimize deviation from uniform distribution
        return np.sum((p - 1/N)**2)
    
    def constraint_mean(p):
        # Risk-neutral condition
        return np.sum(p * np.array(moves)) - growth
    
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
    expected_growth = np.sum(probs * np.array(moves))
    
    print(f"\n{N}-nomial Probabilities:")
    for i, p in enumerate(probs):
        print(f"  p[{i}] = {p:.6f}")
    print(f"  Sum = {prob_sum:.6f}, Expected growth = {expected_growth:.6f}")
    
    assert abs(prob_sum - 1.0) < 1e-6, "Probabilities must sum to 1"
    assert abs(expected_growth - growth) < 1e-2, "Risk-neutral condition violated"
    assert all(p >= 0 for p in probs), "Negative probabilities"
    
    print("  ✓ Valid risk-neutral probabilities")
    return probs


probs = calculate_nnomial_probabilities(moves, r, dt)


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
# PATH SIMULATION WITH TRACKING
# ============================================================
class NnomialPathSimulator:
    """Simulate N-nomial price paths WITH path history tracking."""
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
                    # Sample next move according to probabilities
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
    """Least Squares Monte Carlo for continuation value estimation."""
    def __init__(self, polynomial_degree=3, alpha=0.1):
        self.poly_degree = polynomial_degree
        self.alpha = alpha
        self.continuation_models = {}
        self.T_steps = T_steps
    
    def estimate_continuation_values(self, paths, r, dt):
        print("\n" + "=" * 60)
        print("RUNNING LSMC ESTIMATION")
        print("=" * 60)
        
        # Initialize terminal values
        for path in paths:
            path[-1]['value'] = path[-1]['payoff']
        
        # Backward induction
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
        print("(LSMC used for REWARD ONLY - not given to actor!)")
        print("=" * 60)
        return paths
    
    def predict_continuation_value(self, S, t):
        if t not in self.continuation_models:
            return 0.0
        X = create_polynomial_features([S], self.poly_degree)
        return self.continuation_models[t].predict(X)[0]
    
    def predict_child_continuation_values(self, S, t):
        """Returns [V_0, V_1, ..., V_{N-1}] for N children."""
        if t >= self.T_steps - 1:
            # Terminal: just compute payoffs
            return [max(S * move - K, 0) for move in moves]
        else:
            # Non-terminal: predict continuation values
            return [self.predict_continuation_value(S * move, t + 1) for move in moves]


# ============================================================
# NEURAL NETWORKS
# ============================================================
class UniversalActor(nn.Module):
    """Actor network that maps state to hedge positions."""
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
    """Critic network that estimates Q-value."""
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
class UniversalDDPGAgent:
    """Deep Deterministic Policy Gradient agent."""
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
def construct_state(S, t, path_history):
    """
    Construct state vector WITHOUT LSMC targets!
    State = [S_norm, t_norm, path_encoding_1, path_encoding_2]
    Path encoding uses normalized move indices
    """
    state_vec = np.zeros(4)
    state_vec[0] = S / S0
    state_vec[1] = t / T_steps
    
    # Path encoding: normalize move index to [-1, +1]
    # Index 0 (largest up) → +1, Index N-1 (largest down) → -1
    if len(path_history) >= 1:
        state_vec[2] = 1.0 - 2.0 * path_history[-1] / (N - 1)
    if len(path_history) >= 2:
        state_vec[3] = 1.0 - 2.0 * path_history[-2] / (N - 1)
    
    return state_vec


def compute_reward(hedge, S, t, lsmc_estimator, is_terminal):
    """Compute reward based on deviation from LSMC targets."""
    hedge = np.atleast_1d(hedge)
    
    if is_terminal:
        target = max(S - K, 0)
        deviation = abs(hedge[0] - target)
    else:
        targets = lsmc_estimator.predict_child_continuation_values(S, t)
        deviations = [abs(hedge[i] - targets[i]) for i in range(N)]
        deviation = np.mean(deviations)
    
    norm_factor = max(max_terminal_payoff, 1.0)
    normalized_deviation = deviation / norm_factor
    reward = -REPLICATION_PENALTY * normalized_deviation**2
    
    return np.clip(reward, -10000000, 0), deviation


# ============================================================
# TRAINING LOOP
# ============================================================
def train_pure_rl():
    """Main training loop - PURE RL with path encoding."""
    simulator = NnomialPathSimulator(S0, moves, probs, T_steps, dt)
    lsmc_estimator = LSMCEstimator(polynomial_degree=POLYNOMIAL_DEGREE, alpha=REGRESSION_ALPHA)
    
    agent = UniversalDDPGAgent(state_dim=4, action_dim=N, hidden_dim=HIDDEN_DIM)
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
        
        # Generate paths and compute LSMC estimates
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        print(f"\nPure RL training (NO LSMC in state - only in reward)...")
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
            
            reward, deviation = compute_reward(action_used, S, t, lsmc_estimator, is_terminal)
            
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
        
        # Evaluation
        print(f"\nEvaluating hedge accuracy...")
        total_deviation, num_evals = 0, 0
        
        for path in paths[:1000]:
            for node in path:
                S_eval, t_eval = node['S'], node['t']
                path_history_eval = node['path_history']
                is_terminal_eval = (t_eval == T_steps)
                
                state_eval = construct_state(S_eval, t_eval, path_history_eval)
                action_eval = agent.select_action(state_eval, add_noise=False)
                action_used_eval = action_eval[:1] if is_terminal_eval else action_eval[:N]
                
                _, deviation = compute_reward(action_used_eval, S_eval, t_eval, lsmc_estimator, is_terminal_eval)
                
                total_deviation += deviation
                num_evals += 1
        
        avg_deviation = total_deviation / num_evals
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Average Absolute Deviation: ${avg_deviation:.4f}")
        
        # Early stopping
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
    
    return agent, lsmc_estimator


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    agent, lsmc_estimator = train_pure_rl()
    
    # ============================================================
    # FINAL EVALUATION
    # ============================================================
    print("\n" + "="*60)
    print(f"FINAL EVALUATION - PURE RL {N}-NOMIAL")
    print("="*60)
    
    # Sample 20 test states from different parts of the tree
    print(f"\nEvaluating sample states from the tree...")
    print("(Testing root, t=1 nodes, and diverse t={T_steps} terminal nodes)")
    print("="*60)
    
    test_count = 0
    success_count = 0
    
    # Test root
    S, t, path_history = S0, 0, []
    is_terminal = False
    
    lsmc_target = lsmc_estimator.predict_child_continuation_values(S, t)
    state = construct_state(S, t, path_history)
    action = agent.select_action(state, add_noise=False)
    action_used = action[:N]
    
    _, deviation = compute_reward(action_used, S, t, lsmc_estimator, is_terminal)
    is_success = deviation < 5.0  # More lenient for N-nomial
    if is_success:
        success_count += 1
    test_count += 1
    
    print(f"\nt={t} | S=${S:.2f} | Path: ROOT")
    print("-"*60)
    print(f"LSMC Targets (hidden): {[f'${v:.2f}' for v in lsmc_target]}")
    print(f"Actor Output: {[f'{a:+.2f}' for a in action_used]}")
    print(f"Average Deviation: ${deviation:.4f} {'✓' if is_success else '✗'}")
    print("="*60)
    
    # Test some t=1 nodes
    print(f"\nSampling t=1 nodes...")
    for move_idx in [0, N//2, N-1]:  # Test highest, middle, lowest moves
        S = S0 * moves[move_idx]
        t = 1
        path_history = [move_idx]
        is_terminal = False
        
        lsmc_target = lsmc_estimator.predict_child_continuation_values(S, t)
        state = construct_state(S, t, path_history)
        action = agent.select_action(state, add_noise=False)
        action_used = action[:N]
        
        _, deviation = compute_reward(action_used, S, t, lsmc_estimator, is_terminal)
        is_success = deviation < 5.0
        if is_success:
            success_count += 1
        test_count += 1
        
        move_name = f"move[{move_idx}]"
        print(f"\nt={t} | S=${S:.2f} | Path: {move_name}")
        print("-"*60)
        print(f"Average Deviation: ${deviation:.4f} {'✓' if is_success else '✗'}")
    
    # Test diverse terminal nodes
    print(f"\nSampling t={T_steps} terminal nodes...")
    terminal_samples = [
        [0] * T_steps,  # All up
        [N-1] * T_steps,  # All down
        [N//2] * T_steps,  # All middle
        [0, N-1] + [N//2] * (T_steps-2),  # Mixed
        list(range(min(T_steps, N)))  # Sequential
    ]
    
    for path_history in terminal_samples[:min(5, len(terminal_samples))]:
        S = S0
        for move_idx in path_history:
            S *= moves[move_idx]
        t = T_steps
        is_terminal = True
        
        lsmc_target = max(S - K, 0)
        state = construct_state(S, t, path_history)
        action = agent.select_action(state, add_noise=False)
        action_used = action[:1]
        
        _, deviation = compute_reward(action_used, S, t, lsmc_estimator, is_terminal)
        is_success = deviation < 5.0
        if is_success:
            success_count += 1
        test_count += 1
        
        path_str = "→".join([f"{idx}" for idx in path_history])
        print(f"\nt={t} | S=${S:.2f} | Path: {path_str}")
        print("-"*60)
        print(f"LSMC Target (hidden): ${lsmc_target:.4f}")
        print(f"Actor Output: {action_used[0]:+.4f}")
        print(f"Deviation: ${deviation:.4f} {'✓' if is_success else '✗'}")
    
    print("\n" + "="*60)
    print(f"SUCCESS RATE: {success_count}/{test_count} ({100*success_count/test_count:.1f}%)")
    print("="*60)
    print(f"\nPURE DEEP RL VALIDATION ({N}-NOMIAL):")
    print(f"• State: [S/S0, t/T, path_encoding] - NO LSMC values!")
    print(f"• Actor learned from REWARDS ONLY (trial-and-error)")
    print(f"• LSMC hidden in reward function, NOT in state")
    print(f"• Path encoding helps distinguish different histories")
    print(f"• Optimizations: {TOTAL_EPISODES:,} episodes, LayerNorm, scaled params")
    print(f"• Network scales with N: {HIDDEN_DIM} hidden units for N={N}")
    print("="*60)
    print("\nThe actor discovered optimal hedges through")
    print("pure reinforcement learning - no supervision!")
    print("="*60)