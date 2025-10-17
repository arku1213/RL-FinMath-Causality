import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from scipy.optimize import linprog

# ============================================================
# CONFIGURATION
# ============================================================
S0 = 100.0
r = 0.05
K = 100.0
T_steps = 2
dt = 1.0

# N-NOMIAL INCOMPLETE MARKET PARAMETERS
N = 5                    # Number of states (CHANGE THIS: 3, 4, 5, 7, 10, etc.)
NUM_BINARIES = N - 1     # Number of binaries (can also make this N-2, or user-specified)

sigma = 0.2

# LSMC Parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# ============================================================
# SUPER-REPLICATION HYPERPARAMETERS (ASYMMETRIC L2)
# ============================================================
ACTOR_LR = 0.0001
CRITIC_LR = 0.0003

max_stock_price = S0 * (np.exp(sigma * np.sqrt(2 * dt)) ** T_steps)
max_terminal_payoff = max(max_stock_price - K, 0)
ACTION_SCALE = max_terminal_payoff * 3.0

# ASYMMETRIC PENALTIES FOR SUPER-REPLICATION
SHORTFALL_PENALTY = 10000
EXCESS_PENALTY = 10     # INCREASED: penalize over-hedging more
COST_WEIGHT = 0.1       # INCREASED: care more about cost

# Training schedule
TOTAL_EPISODES = 300000
NUM_ITERATIONS = 12
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 300000
HIDDEN_DIM = 256

print("="*60)
print(f"N-NOMIAL INCOMPLETE MARKET SUPER-REPLICATION")
print(f"N={N}, BINARIES={NUM_BINARIES}")
print("="*60)
print(f"Market: {N} states, {NUM_BINARIES} binaries → INCOMPLETE")
print(f"Goal: SUPER-REPLICATION (never underpay)")
print(f"Penalties: Shortfall={SHORTFALL_PENALTY}, Excess={EXCESS_PENALTY}")
print(f"Asymmetry Ratio: {SHORTFALL_PENALTY/EXCESS_PENALTY}:1")
print("="*60)

# ============================================================
# N-NOMIAL PARAMETERS
# ============================================================
def calculate_n_nomial_parameters(N, S0, r, sigma, dt):
    """Calculate multipliers and probabilities for N-nomial tree"""
    growth = np.exp(r * dt)
    
    if N == 2:
        u = np.exp(sigma * np.sqrt(dt))
        multipliers = [u, 1/u]
    elif N == 3:
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = 1 / u
        multipliers = [u, 1.0, d]
    else:
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = 1 / u
        exponents = np.linspace((N-1)/2, -(N-1)/2, N)
        base = u
        multipliers = [base ** exp for exp in exponents]
    
    multipliers = np.array(multipliers)
    
    # Solve for risk-neutral probabilities
    min_prob = 0.001
    A_eq = np.array([np.ones(N), multipliers])
    b_eq = np.array([1.0, growth])
    bounds = [(min_prob, 1.0 - min_prob * (N-1)) for _ in range(N)]
    c = np.ones(N)
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        probabilities = result.x
    else:
        print("WARNING: Could not solve for probabilities. Using fallback.")
        probabilities = np.ones(N) / N
        current_growth = np.sum(probabilities * multipliers)
        adjustment = (growth - current_growth) / np.sum(multipliers)
        probabilities = probabilities + adjustment * multipliers / np.sum(multipliers**2)
        probabilities = np.maximum(probabilities, min_prob)
        probabilities = probabilities / np.sum(probabilities)
    
    prob_sum = np.sum(probabilities)
    expected_growth = np.sum(probabilities * multipliers)
    
    print(f"\n{N}-nomial Configuration:")
    for i, (m, p) in enumerate(zip(multipliers, probabilities)):
        print(f"  State {i}: multiplier={m:.4f}, prob={p:.4f}")
    print(f"  Sum of probs = {prob_sum:.6f}, Expected growth = {expected_growth:.6f}")
    
    return multipliers, probabilities

multipliers, probabilities = calculate_n_nomial_parameters(N, S0, r, sigma, dt)

# ============================================================
# BINARY PAYOFF MATRIX (SEQUENTIAL PARTITIONING)
# ============================================================
def create_payoff_matrix(N, num_binaries):
    """
    Create binary payoff matrix for incomplete market.
    Sequential partitioning (Option A):
    - First (num_binaries-1) binaries each cover 1 state
    - Last binary covers all remaining states
    
    Returns: matrix[state_i, binary_j] = 1 if binary_j pays when state_i occurs
    """
    payoff_matrix = np.zeros((N, num_binaries))
    
    # First (num_binaries-1) binaries: one-to-one mapping
    for i in range(num_binaries - 1):
        payoff_matrix[i, i] = 1
    
    # Last binary: covers all remaining states
    remaining_states = N - (num_binaries - 1)
    for i in range(num_binaries - 1, N):
        payoff_matrix[i, num_binaries - 1] = 1
    
    return payoff_matrix

payoff_matrix = create_payoff_matrix(N, NUM_BINARIES)

print(f"\nBinary Payoff Matrix (Sequential Partitioning):")
print("Rows = States, Columns = Binaries")
for i in range(N):
    row_str = " ".join([str(int(payoff_matrix[i, j])) for j in range(NUM_BINARIES)])
    print(f"  State {i}: [{row_str}]")
print("\nBinary Coverage:")
for j in range(NUM_BINARIES):
    states_covered = [i for i in range(N) if payoff_matrix[i, j] == 1]
    print(f"  Binary {j} → States {states_covered} ({len(states_covered)} state(s))")
print(f"\nIncompleteness: Last binary covers {N - (NUM_BINARIES-1)} states")
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
class NnomialPathSimulator:
    def __init__(self, S0, multipliers, probabilities, T_steps, dt):
        self.S0 = S0
        self.multipliers = multipliers
        self.probabilities = probabilities
        self.N = len(multipliers)
        self.T_steps = T_steps
        self.dt = dt
    
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
                    state_occurred = np.random.choice(self.N, p=self.probabilities)
                    S *= self.multipliers[state_occurred]
                    path_step['state_occurred'] = state_occurred
                
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
    
    def predict_state_continuation_values(self, S, t):
        """Returns continuation values for all N states"""
        if t >= self.T_steps - 1:
            return [max(S * m - K, 0) for m in multipliers]
        else:
            return [self.predict_continuation_value(S * m, t + 1) for m in multipliers]

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
# STATE & REWARD - SUPER-REPLICATION (ASYMMETRIC L2)
# ============================================================
def construct_state(S, t, target_values):
    """State: [S_norm, avg_target, t_norm, target_0, ..., target_{N-1}]"""
    norm_factor = max_terminal_payoff if max_terminal_payoff > 0 else 1.0
    
    state_vec = np.zeros(3 + N)
    state_vec[0] = S / S0
    state_vec[1] = np.mean(target_values) / norm_factor
    state_vec[2] = t / T_steps
    state_vec[3:3+N] = np.array(target_values) / norm_factor
    
    return state_vec

def compute_super_replication_reward(hedge, target_values, binary_prices, is_terminal, state_idx=None):
    """
    SUPER-REPLICATION with ASYMMETRIC L2 penalties
    """
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    
    if is_terminal:
        realized = hedge[0]
        target = target_values
        shortfall = max(0, target - realized)
        excess = max(0, realized - target)
    else:
        if state_idx is not None:
            # Training: specific state
            realized = np.sum(hedge * payoff_matrix[state_idx, :])
            target = target_values[state_idx]
            shortfall = max(0, target - realized)
            excess = max(0, realized - target)
        else:
            # Evaluation: all N states
            shortfalls = []
            excesses = []
            for i in range(N):
                realized_i = np.sum(hedge * payoff_matrix[i, :])
                target_i = target_values[i]
                shortfalls.append(max(0, target_i - realized_i))
                excesses.append(max(0, realized_i - target_i))
            
            shortfall = np.mean(shortfalls)
            excess = np.mean(excesses)
    
    norm_factor = max_terminal_payoff if max_terminal_payoff > 0 else 1.0
    normalized_shortfall = shortfall / norm_factor
    normalized_excess = excess / norm_factor
    
    # ASYMMETRIC L2
    reward = -(COST_WEIGHT * abs(cost)
               + SHORTFALL_PENALTY * normalized_shortfall**2
               + EXCESS_PENALTY * normalized_excess**2)
    
    return np.clip(reward, -100000, 0), cost, shortfall, excess

# ============================================================
# TRAINING LOOP WITH MULTI-CHILD TRAINING
# ============================================================
def train_super_replication_agent():
    simulator = NnomialPathSimulator(S0, multipliers, probabilities, T_steps, dt)
    lsmc_estimator = LSMCEstimator(polynomial_degree=POLYNOMIAL_DEGREE, alpha=REGRESSION_ALPHA)
    
    agent = UniversalDDPGAgent(state_dim=3+N, action_dim=NUM_BINARIES, hidden_dim=HIDDEN_DIM)
    
    best_avg_shortfall = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}\nITERATION {iteration + 1}/{NUM_ITERATIONS}\n{'='*60}")
        
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        print(f"\nTraining N={N} super-replication agent...")
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
            else:
                target = lsmc_estimator.predict_state_continuation_values(S, t)
                # Binary prices weighted by probability
                prices = [np.exp(-r * dt) * np.sum(probabilities[i] for i in range(N) if payoff_matrix[i, j] == 1) 
                         for j in range(NUM_BINARIES)]
            
            state = construct_state(S, t, target)
            
            noise_decay = max(0.05, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.9
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            action_used = action[:1] if is_terminal else action[:NUM_BINARIES]
            
            # MULTI-CHILD TRAINING
            if is_terminal:
                reward, _, _, _ = compute_super_replication_reward(
                    action_used, target, prices, True, None
                )
                agent.replay_buffer.push(state, action, reward, state, False)
                reward_history.append(reward)
            else:
                for state_idx in range(N):
                    reward, _, _, _ = compute_super_replication_reward(
                        action_used, target, prices, False, state_idx
                    )
                    agent.replay_buffer.push(state, action, reward, state, False)
                    reward_history.append(reward)
            
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            if (episode + 1) % (episodes_this_iter // 8) == 0:
                avg_reward = np.mean(reward_history[-1000:]) if len(reward_history) >= 1000 else np.mean(reward_history)
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.1f}")

        print(f"\nEvaluating...")
        
        total_shortfall, total_excess, num_evals = 0, 0, 0
        max_shortfall = 0
        
        for path in paths[:1000]:
            for node in path:
                S_eval, t_eval = node['S'], node['t']
                is_terminal_eval = (t_eval == T_steps)
                
                if is_terminal_eval:
                    target_eval = node['payoff']
                    prices_eval = [np.exp(-r * dt)]
                else:
                    target_eval = lsmc_estimator.predict_state_continuation_values(S_eval, t_eval)
                    prices_eval = [np.exp(-r * dt) * np.sum(probabilities[i] for i in range(N) if payoff_matrix[i, j] == 1) 
                                  for j in range(NUM_BINARIES)]

                state_eval = construct_state(S_eval, t_eval, target_eval)
                action_eval = agent.select_action(state_eval, add_noise=False)
                action_used_eval = action_eval[:1] if is_terminal_eval else action_eval[:NUM_BINARIES]
                
                _, _, shortfall, excess = compute_super_replication_reward(
                    action_used_eval, target_eval, prices_eval, is_terminal_eval, None
                )
                
                total_shortfall += shortfall
                total_excess += excess
                max_shortfall = max(max_shortfall, shortfall)
                num_evals += 1

        avg_shortfall = total_shortfall / num_evals
        avg_excess = total_excess / num_evals
        
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Avg Shortfall: {avg_shortfall:.6f} {'✓' if avg_shortfall < 0.1 else '✗'}")
        print(f"  Max Shortfall: {max_shortfall:.6f}")
        print(f"  Avg Excess: {avg_excess:.6f}")
        
        if avg_shortfall < best_avg_shortfall:
            best_avg_shortfall = avg_shortfall
            print(f"  ✓ NEW BEST!")
        
        if avg_shortfall < 0.1:
            print(f"\n🎉 SUCCESS! Shortfall < 0.1 at iteration {iteration + 1}")
            break
            
    print(f"\n{'='*60}\nTRAINING COMPLETE!\n{'='*60}")
    print(f"Best Avg Shortfall: {best_avg_shortfall:.6f}")
    return agent, lsmc_estimator

# ============================================================
# MAIN EXECUTION
# ============================================================
agent, lsmc_estimator = train_super_replication_agent()

# ============================================================
# FINAL EVALUATION
# ============================================================
print("\n" + "="*60 + "\nFINAL EVALUATION\n" + "="*60)

# Root state
S, t = S0, 0
target = lsmc_estimator.predict_state_continuation_values(S, t)
prices = [np.exp(-r * dt) * np.sum(probabilities[i] for i in range(N) if payoff_matrix[i, j] == 1) 
         for j in range(NUM_BINARIES)]
state = construct_state(S, t, target)
action = agent.select_action(state, add_noise=False)
_, cost, shortfall, excess = compute_super_replication_reward(
    action[:NUM_BINARIES], target, prices, False, None
)

print(f"\nRoot State: S=${S:.2f}, t={t}")
print(f"  State Targets: {[f'${v:.2f}' for v in target]}")
print(f"  Binary Hedges: {[f'{h:.2f}' for h in action[:NUM_BINARIES]]}")
print(f"\n  Per-State Analysis:")
for i in range(N):
    realized_i = np.sum(action[:NUM_BINARIES] * payoff_matrix[i, :])
    shortfall_i = max(0, target[i] - realized_i)
    excess_i = max(0, realized_i - target[i])
    status = "✓" if shortfall_i < 0.1 else "✗"
    print(f"    State {i}: Target=${target[i]:6.2f}, Realized=${realized_i:6.2f}, "
          f"Short={shortfall_i:5.2f}, Excess={excess_i:5.2f} {status}")

print(f"\n  Summary: Cost=${cost:.2f}, Avg Shortfall={shortfall:.4f}, Avg Excess={excess:.4f}")

print("\n" + "="*60)
print(f"N-NOMIAL INCOMPLETE SUPER-REPLICATION")
print("="*60)
print(f"• N={N} states, {NUM_BINARIES} binaries → INCOMPLETE")
print(f"• Asymmetric L2: Short={SHORTFALL_PENALTY}, Excess={EXCESS_PENALTY}")
print(f"• Multi-child training: {N}× data per sample")
print(f"• Sequential partitioning: Last binary covers {N-(NUM_BINARIES-1)} states")
print("="*60)
print("\nTo test different N: Change N at top (3, 4, 5, 7, 10, etc.)")
print("To change incompleteness: Adjust NUM_BINARIES (e.g., N-2 for more incomplete)")
print("="*60)