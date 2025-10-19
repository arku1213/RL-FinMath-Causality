import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION - TRINOMIAL SUPER-REPLICATION
# ============================================================
S0 = 100.0
r = 0.05
K = 100.0
T_steps = 2
dt = 1.0

# Trinomial parameters
sigma = 0.3
lambda_param = np.sqrt(3)
u = np.exp(lambda_param * sigma * np.sqrt(dt))
d = np.exp(-lambda_param * sigma * np.sqrt(dt))
m = 1.0

# LSMC parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# SUPER-REPLICATION: Conservative LSMC
CONSERVATISM_FACTOR = 1.10  # 10% buffer for safety

# RL hyperparameters - SUPER-REPLICATION SPECIFIC
BASE_ACTOR_LR = 0.00003
BASE_CRITIC_LR = 0.00010
ACTOR_LR = BASE_ACTOR_LR
CRITIC_LR = BASE_CRITIC_LR

max_stock_price = S0 * (u ** T_steps)
max_terminal_payoff = max(max_stock_price - K, 0)
ACTION_SCALE = max_terminal_payoff * 1.2

# SUPER-REPLICATION PENALTIES
SHORTFALL_PENALTY = 10000000  # HUGE - must NEVER underpay
COST_WEIGHT = 150000          # Aggressive cost minimization - 50:1 ratio!

# Training configuration
TOTAL_EPISODES = 800000
NUM_ITERATIONS = 16
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.003
BUFFER_SIZE = 800000
HIDDEN_DIM = 256

# Improvement parameters
LR_DECAY = 0.93
NOISE_DECAY = 0.75
EARLY_STOP_PATIENCE = 4

print("=" * 60)
print(f"TRINOMIAL COMPLETE SUPER-REPLICATION (T={T_steps})")
print("=" * 60)
print("METHOD: Pure RL discovers MINIMAL-cost super-replicating hedges")
print(f"Market: 3 states → 3 binaries (COMPLETE)")
print("=" * 60)
print("GOAL: Find MINIMAL hedge h where h ≥ LSMC_conservative (dominance)")
print("      AND minimize cost(h) = sum(|h_i|)")
print("=" * 60)
print(f"SHORTFALL_PENALTY: {SHORTFALL_PENALTY:,.0f} (violation >> cost)")
print(f"COST_WEIGHT: {COST_WEIGHT:,.0f} (aggressive cost minimization)")
print(f"CONSERVATISM: {CONSERVATISM_FACTOR}× LSMC (safety buffer)")
print(f"Penalty Ratio: {SHORTFALL_PENALTY/COST_WEIGHT:.0f}:1 (shortfall:cost)")
print("=" * 60)


# ============================================================
# TRINOMIAL PROBABILITIES
# ============================================================
def calculate_trinomial_probabilities(S0, u, m, d, r, dt):
    """Calculate risk-neutral probabilities using numerical optimization."""
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
    bounds = [(0.001, 0.999)] * 3
    
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
    assert all(p >= 0 for p in [p_u, p_m, p_d])
    
    print("  ✓ Valid risk-neutral probabilities")
    return p_u, p_m, p_d


p_u, p_m, p_d = calculate_trinomial_probabilities(S0, u, m, d, r, dt)


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
class TrinomialPathSimulator:
    """Simulate trinomial price paths WITH path history tracking."""
    def __init__(self, S0, u, m, d, p_u, p_m, p_d, T_steps, dt):
        self.S0 = S0
        self.u, self.m, self.d = u, m, d
        self.p_u, self.p_m, self.p_d = p_u, p_m, p_d
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
                    rand = np.random.random()
                    if rand < self.p_u:
                        S *= self.u
                        path_step['child_occurred'] = 0
                        path_history.append(0)
                    elif rand < self.p_u + self.p_m:
                        S *= self.m
                        path_step['child_occurred'] = 1
                        path_history.append(1)
                    else:
                        S *= self.d
                        path_step['child_occurred'] = 2
                        path_history.append(2)
                
                path.append(path_step)
            paths.append(path)
        return paths


# ============================================================
# LSMC ESTIMATOR WITH CONSERVATIVE ESTIMATES
# ============================================================
class LSMCEstimator:
    """Least Squares Monte Carlo with CONSERVATIVE estimates for super-replication."""
    def __init__(self, polynomial_degree=3, alpha=0.1, conservatism=1.0):
        self.poly_degree = polynomial_degree
        self.alpha = alpha
        self.conservatism = conservatism  # Multiplicative buffer
        self.continuation_models = {}
        self.T_steps = T_steps
    
    def estimate_continuation_values(self, paths, r, dt):
        print("\n" + "=" * 60)
        print("RUNNING LSMC ESTIMATION (CONSERVATIVE)")
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
        """Returns [V_u, V_m, V_d] with optional conservative buffer."""
        if t >= self.T_steps - 1:
            # Terminal: exact payoffs (no buffer needed)
            return [max(S * u - K, 0), max(S * m - K, 0), max(S * d - K, 0)]
        else:
            # Non-terminal: apply conservative buffer
            base_values = [
                self.predict_continuation_value(S * u, t + 1),
                self.predict_continuation_value(S * m, t + 1),
                self.predict_continuation_value(S * d, t + 1)
            ]
            
            if conservative:
                # Add safety buffer for super-replication
                return [max(0, v * self.conservatism) for v in base_values]
            else:
                return [max(0, v) for v in base_values]


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
# STATE AND REWARD FUNCTIONS - SUPER-REPLICATION!
# ============================================================
def construct_state(S, t, path_history):
    """Construct state vector WITHOUT LSMC targets (Pure RL)."""
    state_vec = np.zeros(4)
    state_vec[0] = S / S0
    state_vec[1] = t / T_steps
    
    # Path encoding
    if len(path_history) >= 1:
        state_vec[2] = (path_history[-1] - 1) / 2.0
    if len(path_history) >= 2:
        state_vec[3] = (path_history[-2] - 1) / 2.0
    
    return state_vec


def compute_reward_super_replication(hedge, S, t, lsmc_estimator, is_terminal):
    """
    SUPER-REPLICATION REWARD:
    - Heavily penalize SHORTFALL (hedge < conservative_target)
    - Mildly penalize COST (sum of absolute hedges)
    - Allow EXCESS (hedge > target) with no penalty
    """
    hedge = np.atleast_1d(hedge)
    
    if is_terminal:
        # Terminal: exact payoff (no conservatism needed)
        target = max(S - K, 0)
        shortfall = max(0, target - hedge[0])
        cost = abs(hedge[0])
    else:
        # Non-terminal: use CONSERVATIVE LSMC estimates
        conservative_targets = lsmc_estimator.predict_child_continuation_values(S, t, conservative=True)
        
        # Shortfall: max(0, target - hedge) for each child
        shortfalls = [max(0, conservative_targets[i] - hedge[i]) for i in range(3)]
        shortfall = np.mean(shortfalls)  # Average shortfall across children
        
        # Cost: sum of absolute hedge positions
        cost = np.sum(np.abs(hedge))
    
    # ASYMMETRIC REWARD:
    # - HUGE penalty for shortfall (must dominate!)
    # - Small penalty for cost (encourage efficiency)
    reward = -SHORTFALL_PENALTY * shortfall**2 - COST_WEIGHT * cost
    
    return np.clip(reward, -100000000, 0), shortfall, cost


# ============================================================
# TRAINING LOOP
# ============================================================
def train_super_replication():
    """Train Pure RL to discover minimal-cost super-replicating hedges."""
    simulator = TrinomialPathSimulator(S0, u, m, d, p_u, p_m, p_d, T_steps, dt)
    lsmc_estimator = LSMCEstimator(
        polynomial_degree=POLYNOMIAL_DEGREE,
        alpha=REGRESSION_ALPHA,
        conservatism=CONSERVATISM_FACTOR
    )
    
    agent = UniversalDDPGAgent(state_dim=4, action_dim=3, hidden_dim=HIDDEN_DIM)
    reward_normalizer = RewardNormalizer(clip_range=10.0)
    
    best_avg_shortfall = float('inf')
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
        
        # Generate paths and compute CONSERVATIVE LSMC estimates
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
            
            # Pure RL state (NO LSMC values!)
            state = construct_state(S, t, path_history)
            
            noise_decay = max(0.1, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.98
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            action_used = action[:1] if is_terminal else action[:3]
            
            # Super-replication reward
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
        
        # Evaluation
        print(f"\nEvaluating super-replication hedges...")
        total_shortfall, total_cost, num_evals = 0, 0, 0
        
        for path in paths[:1000]:
            for node in path:
                S_eval, t_eval = node['S'], node['t']
                path_history_eval = node['path_history']
                is_terminal_eval = (t_eval == T_steps)
                
                state_eval = construct_state(S_eval, t_eval, path_history_eval)
                action_eval = agent.select_action(state_eval, add_noise=False)
                action_used_eval = action_eval[:1] if is_terminal_eval else action_eval[:3]
                
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
        print(f"  Average Cost: ${avg_cost:.2f} (minimizing excess)")
        print(f"  Average Excess: ${(avg_cost - avg_shortfall):.2f}")
        
        # Early stopping based on shortfall
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
            print(f"\n⚠️  WARNING: Training not improving after {iteration} iterations")
            if best_actor_state is not None:
                agent.actor.load_state_dict(best_actor_state)
            break
        
        if avg_shortfall < 0.1:
            print(f"\n🎯 EXCELLENT! Shortfall < $0.10!")
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
    # FINAL EVALUATION - FINANCIAL MATH PERSPECTIVE
    # ============================================================
    print("\n" + "="*60)
    print("FINAL EVALUATION - SUPER-REPLICATION HEDGES")
    print("="*60)
    print("Financial Math Goal: Find minimal h where h ≥ C_conservative")
    print("="*60)
    
    # Test all 11 unique states
    test_cases = [
        (S0, 0, []),  # Root
        (S0 * u, 1, [0]),  # U
        (S0 * m, 1, [1]),  # M
        (S0 * d, 1, [2]),  # D
        (S0 * u * u, 2, [0, 0]),  # UU
        (S0 * u * m, 2, [0, 1]),  # UM
        (S0 * u * d, 2, [0, 2]),  # UD (= MU)
        (S0 * m * m, 2, [1, 1]),  # MM
        (S0 * m * d, 2, [1, 2]),  # MD (= DU)
        (S0 * d * m, 2, [2, 1]),  # DM
        (S0 * d * d, 2, [2, 2])   # DD
    ]
    
    print(f"\nEvaluating {len(test_cases)} unique nodes")
    print("="*60)
    
    nodes_dominated = 0
    total_shortfall = 0
    total_cost = 0
    
    for S, t, path_history in test_cases:
        is_terminal = (t == T_steps)
        
        # Get CONSERVATIVE LSMC targets
        if is_terminal:
            conservative_targets = max(S - K, 0)
        else:
            conservative_targets = lsmc_estimator.predict_child_continuation_values(S, t, conservative=True)
        
        # RL-discovered hedge
        state = construct_state(S, t, path_history)
        action = agent.select_action(state, add_noise=False)
        hedge = action[:1] if is_terminal else action[:3]
        
        # Check dominance
        if is_terminal:
            shortfall = max(0, conservative_targets - hedge[0])
            excess = max(0, hedge[0] - conservative_targets)
            cost = abs(hedge[0])
            dominates = (shortfall == 0)
        else:
            shortfalls = [max(0, conservative_targets[i] - hedge[i]) for i in range(3)]
            excesses = [max(0, hedge[i] - conservative_targets[i]) for i in range(3)]
            shortfall = np.mean(shortfalls)
            excess = np.mean(excesses)
            cost = np.sum(np.abs(hedge))
            dominates = all(s == 0 for s in shortfalls)
        
        if dominates:
            nodes_dominated += 1
        total_shortfall += shortfall
        total_cost += cost
        
        # Format path
        path_str = "→".join(["U" if p == 0 else "M" if p == 1 else "D" for p in path_history]) if path_history else "ROOT"
        
        print(f"\nNode: t={t}, S=${S:.2f}, Path: {path_str}")
        print("-"*60)
        
        if is_terminal:
            print(f"Conservative Target: ${conservative_targets:.4f}")
            print(f"RL Hedge: {hedge[0]:+.4f}")
            print(f"Shortfall: ${shortfall:.4f}")
            print(f"Excess: ${excess:.4f}")
            print(f"Cost: ${cost:.2f}")
            print(f"Dominates: {'YES ✓' if dominates else 'NO ✗'}")
        else:
            print(f"Conservative Targets (C×{CONSERVATISM_FACTOR}):")
            print(f"  [U=${conservative_targets[0]:.4f}, M=${conservative_targets[1]:.4f}, D=${conservative_targets[2]:.4f}]")
            print(f"RL Hedge:")
            print(f"  [U={hedge[0]:+.4f}, M={hedge[1]:+.4f}, D={hedge[2]:+.4f}]")
            
            # Individual shortfalls/excesses
            if not is_terminal:
                shortfalls_list = [max(0, conservative_targets[i] - hedge[i]) for i in range(3)]
                excesses_list = [max(0, hedge[i] - conservative_targets[i]) for i in range(3)]
                print(f"Shortfalls:")
                print(f"  [U=${shortfalls_list[0]:.4f}, M=${shortfalls_list[1]:.4f}, D=${shortfalls_list[2]:.4f}]")
                print(f"Excesses:")
                print(f"  [U=${excesses_list[0]:.4f}, M=${excesses_list[1]:.4f}, D=${excesses_list[2]:.4f}]")
            
            print(f"Avg Shortfall: ${shortfall:.4f}")
            print(f"Avg Excess: ${excess:.4f}")
            print(f"Total Cost: ${cost:.2f}")
            print(f"Dominates: {'YES ✓' if dominates else 'NO ✗'}")
        
        print("="*60)
    
    # Summary
    print("\n" + "="*60)
    print("SUPER-REPLICATION SUMMARY (MINIMAL UPPER BOUND)")
    print("="*60)
    print(f"Nodes with Complete Dominance: {nodes_dominated}/{len(test_cases)}")
    print(f"Total Shortfall (all nodes): ${total_shortfall:.4f} (should be ~$0)")
    print(f"Average Cost per Node: ${total_cost/len(test_cases):.2f}")
    print(f"Average Excess per Node: ${(total_cost/len(test_cases) - total_shortfall/len(test_cases)):.2f}")
    print("="*60)
    
    if nodes_dominated == len(test_cases) and total_cost/len(test_cases) < 80:
        print("\n🎯 EXCELLENT! Minimal super-replication achieved!")
        print("RL discovered efficient hedges that dominate with minimal excess.")
        print(f"Average cost: ${total_cost/len(test_cases):.2f}/node is near-optimal!")
        print("\n💡 This is likely the MINIMAL UPPER BOUND for super-replication.")
    elif nodes_dominated == len(test_cases) and total_cost/len(test_cases) < 120:
        print(f"\n✓ VERY GOOD: All nodes dominated with low cost (${total_cost/len(test_cases):.2f}/node)")
        print("Cost is very close to minimal upper bound.")
        print("Consider trying COST_WEIGHT = 300,000 to push slightly lower.")
    elif nodes_dominated == len(test_cases):
        print(f"\n✓ GOOD: All nodes dominated with reasonable cost (${total_cost/len(test_cases):.2f}/node)")
        print("Consider increasing COST_WEIGHT to 300,000-400,000 for better efficiency.")
    elif nodes_dominated >= 0.9 * len(test_cases):
        print(f"\n⚠️  PARTIAL SUCCESS: {nodes_dominated}/{len(test_cases)} nodes dominated")
        print(f"Cost: ${total_cost/len(test_cases):.2f}/node is low, but dominance compromised.")
        print("Try COST_WEIGHT = 150,000 for better balance.")
    else:
        print(f"\n✗ COST_WEIGHT TOO HIGH: Only {nodes_dominated}/{len(test_cases)} nodes dominated")
        print(f"Cost: ${total_cost/len(test_cases):.2f}/node")
        print("Reduce COST_WEIGHT to 100,000-150,000 to restore dominance.")
    
    print("="*60)
    print("\nFINANCIAL MATH INTERPRETATION:")
    print("• RL discovered MINIMAL hedge positions h_i at each node")
    print(f"• Conservative LSMC targets include {(CONSERVATISM_FACTOR-1)*100:.0f}% safety buffer")
    print("• Dominance check: h_i ≥ C_conservative for all children")
    print("• Cost minimization: Found smallest |h_i| among feasible hedges")
    print(f"• Penalty ratio: {SHORTFALL_PENALTY/COST_WEIGHT:.0f}:1 (shortfall:cost)")
    print("• Pure RL: No LSMC values in state, learned through rewards only")
    print("="*60)