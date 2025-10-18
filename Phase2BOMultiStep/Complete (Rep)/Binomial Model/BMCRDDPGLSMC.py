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
r = 0.05
K = 100.0
T_steps = 2
dt = 1.0

# BINOMIAL PARAMETERS
sigma = 0.2
u = np.exp(sigma * np.sqrt(dt))      # ≈ 1.221
d = np.exp(-sigma * np.sqrt(dt))     # ≈ 0.819

# LSMC Parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# ============================================================
# PURE RL: EXACT REPLICATION (NO TARGETS IN STATE!)
# ============================================================
ACTOR_LR = 0.00003    # Very conservative (as recommended)
CRITIC_LR = 0.00010   # Slightly larger

# Binomial specific
q = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
max_stock_price = S0 * (u ** T_steps)
max_terminal_payoff = max(max_stock_price - K, 0)
ACTION_SCALE = max_terminal_payoff * 5.0  # Reduced for stability

# PURE RL: Reward-based learning
REPLICATION_PENALTY = 10000000   # Massive penalty for deviation
COST_WEIGHT = 0.0                # ZERO - exact replication only

# Training schedule (expect slower convergence!)
TOTAL_EPISODES = 540000
NUM_ITERATIONS = 12
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 500000
HIDDEN_DIM = 256

print("="*60)
print(f"PURE DEEP RL: BINOMIAL EXACT REPLICATION (T={T_steps})")
print("="*60)
print("METHOD: Pure RL - Actor learns from rewards only!")
print(f"Market: 2 states → 2 binaries (COMPLETE)")
print(f"Risk-neutral prob q: {q:.6f}")
print("="*60)
print("CRITICAL DIFFERENCE FROM SUPERVISED:")
print("  • State does NOT include LSMC targets")
print("  • Actor learns hedge ratios from reward signal")
print("  • LSMC only used inside reward function")
print("  • This is GENUINE Deep RL!")
print("="*60)

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
class BinomialPathSimulator:
    def __init__(self, S0, u, d, q, T_steps, dt):
        self.S0, self.u, self.d, self.q = S0, u, d, q
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
                    child_occurred = 0 if np.random.random() < self.q else 1
                    S *= self.u if child_occurred == 0 else self.d
                    path_step['child_occurred'] = child_occurred
                
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
        
        print("LSMC ESTIMATION COMPLETE")
        print("(LSMC targets will be used in REWARD only, not given to actor!)")
        print("="*60)
        return paths
    
    def predict_continuation_value(self, S, t):
        if t not in self.continuation_models: 
            return 0.0
        X = create_polynomial_features([S], self.poly_degree)
        return self.continuation_models[t].predict(X)[0]
    
    def predict_child_continuation_values(self, S, t):
        """Returns [V_up, V_down] - used ONLY in reward computation"""
        if t >= self.T_steps - 1:
            return [max(S * u - K, 0), max(S * d - K, 0)]
        else:
            return [self.predict_continuation_value(S * u, t + 1),
                    self.predict_continuation_value(S * d, t + 1)]

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
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):  # Reduced sigma
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
# PURE RL: STATE WITHOUT TARGETS!
# ============================================================
def construct_state(S, t):
    """
    CRITICAL: State does NOT include LSMC targets!
    
    Pure RL: Actor must learn from (S, t) → hedge mapping
    through reward signal alone
    """
    state_vec = np.zeros(2)
    state_vec[0] = S / S0      # Normalized stock price
    state_vec[1] = t / T_steps # Normalized time
    
    return state_vec

# ============================================================
# PURE RL: REWARD FUNCTION (LSMC USED HERE ONLY!)
# ============================================================
def compute_reward(hedge, S, t, lsmc_estimator, is_terminal):
    """
    PURE RL REWARD: Actor never sees LSMC targets!
    
    LSMC targets are computed inside this function and used
    ONLY to calculate the reward. The actor learns what actions
    get high rewards through trial-and-error.
    
    This is genuine Deep RL!
    """
    hedge = np.atleast_1d(hedge)
    
    if is_terminal:
        # Terminal: Compare hedge to actual payoff
        target = max(S - K, 0)
        deviation = abs(hedge[0] - target)
    else:
        # Intermediate: Get LSMC targets (NOT given to actor!)
        targets = lsmc_estimator.predict_child_continuation_values(S, t)
        
        # Check ALL children (no multi-child bug!)
        deviations = [abs(hedge[i] - targets[i]) for i in range(2)]
        deviation = np.mean(deviations)  # or np.max() for worst-case
    
    # Reward = negative squared deviation (want to minimize)
    norm_factor = max(max_terminal_payoff, 1.0)
    normalized_deviation = deviation / norm_factor
    
    reward = -REPLICATION_PENALTY * normalized_deviation**2
    
    return np.clip(reward, -100000000, 0), deviation

# ============================================================
# PURE RL: TRAINING LOOP
# ============================================================
def train_pure_rl():
    simulator = BinomialPathSimulator(S0, u, d, q, T_steps, dt)
    lsmc_estimator = LSMCEstimator(polynomial_degree=POLYNOMIAL_DEGREE, alpha=REGRESSION_ALPHA)
    
    # State: ONLY 2 features [S, t] - NO targets!
    # Action: 2 (hedge positions for up/down)
    agent = UniversalDDPGAgent(state_dim=2, action_dim=2, hidden_dim=HIDDEN_DIM)
    
    best_avg_deviation = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}\nITERATION {iteration + 1}/{NUM_ITERATIONS}\n{'='*60}")
        
        # Recompute LSMC (periodic refresh as recommended)
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        print(f"\nPure RL training (actor learns from rewards only)...")
        episodes_this_iter = TOTAL_EPISODES // NUM_ITERATIONS
        
        reward_history = []
        
        for episode in range(episodes_this_iter):
            # Sample a node
            path_idx = np.random.randint(len(paths))
            time_idx = np.random.randint(T_steps + 1)
            
            sampled_node = paths[path_idx][time_idx]
            S, t = sampled_node['S'], sampled_node['t']
            is_terminal = (t == T_steps)
            
            # Construct state WITHOUT targets!
            state = construct_state(S, t)
            
            # Actor selects action (with exploration)
            noise_decay = max(0.05, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.95
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            # Use appropriate number of actions
            action_used = action[:1] if is_terminal else action[:2]
            
            # Compute reward (LSMC targets used here, NOT given to actor!)
            reward, deviation = compute_reward(action_used, S, t, lsmc_estimator, is_terminal)
            
            # Store single transition (NO multi-child training!)
            agent.replay_buffer.push(state, action, reward, state, False)
            reward_history.append(reward)
            
            # DDPG update
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            if (episode + 1) % (episodes_this_iter // 8) == 0:
                avg_reward = np.mean(reward_history[-1000:]) if len(reward_history) >= 1000 else np.mean(reward_history)
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.1f}")

        # Evaluate
        print(f"\nEvaluating hedge accuracy...")
        
        total_deviation, num_evals = 0, 0
        
        for path in paths[:1000]:
            for node in path:
                S_eval, t_eval = node['S'], node['t']
                is_terminal_eval = (t_eval == T_steps)
                
                state_eval = construct_state(S_eval, t_eval)
                action_eval = agent.select_action(state_eval, add_noise=False)
                action_used_eval = action_eval[:1] if is_terminal_eval else action_eval[:2]
                
                _, deviation = compute_reward(action_used_eval, S_eval, t_eval, lsmc_estimator, is_terminal_eval)
                
                total_deviation += deviation
                num_evals += 1

        avg_deviation = total_deviation / num_evals
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Average Absolute Deviation: ${avg_deviation:.4f}")
        
        if avg_deviation < best_avg_deviation:
            best_avg_deviation = avg_deviation
            print(f"  ✓ NEW BEST!")
        
        if avg_deviation < 1.0:
            print(f"\n🎉 EXCELLENT! Avg Deviation < $1.00")
            if avg_deviation < 0.5:
                print(f"🎯 OUTSTANDING! Within $0.50 - essentially exact!")
                break
            
    print(f"\n{'='*60}\nTRAINING COMPLETE!\n{'='*60}")
    print(f"Best Average Absolute Deviation: ${best_avg_deviation:.4f}")
    return agent, lsmc_estimator

# ============================================================
# MAIN EXECUTION
# ============================================================
agent, lsmc_estimator = train_pure_rl()

# ============================================================
# FINAL EVALUATION
# ============================================================
print("\n" + "="*60 + "\nFINAL EVALUATION - PURE RL HEDGE RATIOS\n" + "="*60)

# Test key states
test_states = [(S0, 0), (S0 * u, 1), (S0 * d, 1)]
if T_steps >= 2:
    test_states += [(S0 * u * u, 2), (S0 * u * d, 2), (S0 * d * d, 2)]

print(f"\nEvaluating {len(test_states)} key states")
print("="*60)

success_count = 0

for S, t in test_states:
    is_terminal = (t == T_steps)
    
    # Get LSMC targets (for comparison only)
    if is_terminal:
        lsmc_target = max(S - K, 0)
    else:
        lsmc_target = lsmc_estimator.predict_child_continuation_values(S, t)
    
    # Actor chooses hedge (without seeing targets!)
    state = construct_state(S, t)
    action = agent.select_action(state, add_noise=False)
    action_used = action[:1] if is_terminal else action[:2]
    
    # Evaluate
    _, deviation = compute_reward(action_used, S, t, lsmc_estimator, is_terminal)
    
    is_success = deviation < 1.0
    if is_success:
        success_count += 1
    
    print(f"\nt={t} | S=${S:.2f}")
    print("-"*60)
    print(f"State Given to Actor: [S/S0={S/S0:.4f}, t/T={t/T_steps:.4f}]")
    
    if is_terminal:
        print(f"LSMC Target (not seen by actor): ${lsmc_target:.4f}")
        print(f"Actor Output (learned from rewards): {action_used[0]:+.4f}")
        print(f"Deviation: ${deviation:.4f} {'✓' if is_success else '✗'}")
    else:
        print(f"LSMC Targets (not seen by actor): [Up=${lsmc_target[0]:.4f}, Down=${lsmc_target[1]:.4f}]")
        print(f"Actor Output (learned from rewards): [Up={action_used[0]:+.4f}, Down={action_used[1]:+.4f}]")
        dev_up = abs(action_used[0] - lsmc_target[0])
        dev_down = abs(action_used[1] - lsmc_target[1])
        print(f"Deviations: [Up=${dev_up:.4f}, Down=${dev_down:.4f}]")
        print(f"Average: ${deviation:.4f} {'✓' if is_success else '✗'}")
    
    print("-"*60)
    if is_success:
        print("✓ RL agent learned correct hedge!")
    else:
        print("✗ RL agent hasn't converged yet")
    print("="*60)

print("\n" + "="*60)
print(f"SUCCESS RATE: {success_count}/{len(test_states)} ({100*success_count/len(test_states):.1f}%)")
print("="*60)
print("\nPURE DEEP RL VALIDATION:")
print(f"• State space: [S/S0, t/T] - NO targets!")
print(f"• Actor learns hedge ratios from rewards only")
print(f"• LSMC targets hidden inside reward function")
print(f"• This is GENUINE Deep RL - not supervised learning!")
print("="*60)
print("\nThe actor discovered these hedge positions through")
print("trial-and-error reinforcement learning, not by being")
print("told what the correct answer is!")
print("="*60)