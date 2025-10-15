import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from scipy.optimize import linprog

# ============================================================
# CONFIGURATION - CHANGE N AND T HERE!
# ============================================================
S0 = 100.0
K = 100.0
r = 0.05
dt = 1.0

# N-nomial parameters
N = 5  # ← CHANGE THIS: Number of states (2=binomial, 3=trinomial, 4, 5, etc.)
T_steps = 5  # ← CHANGE THIS: Time steps

# LSMC Parameters
NUM_SIMULATIONS = 10000  # Paths to simulate
POLYNOMIAL_DEGREE = 3    # Polynomial degree for regression
REGRESSION_ALPHA = 0.1   # Ridge regularization

# Universal Agent Hyperparameters
TOTAL_EPISODES = 150000  # Total training episodes
NUM_ITERATIONS = 10  # INCREASED for better convergence with N-nomial
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0003
CRITIC_LR = 0.0008
BUFFER_SIZE = 200000
HIDDEN_DIM = 256

# Super-Replication Parameters
SUPER_REPLICATION = True
SHORTFALL_MULTIPLIER = 600 + T_steps * 200 + (N - 2) * 100  # Scales with N and T
COST_WEIGHT = 0.01  # Low - accept higher cost for super-replication
EXCESS_PENALTY = 0.01  # Low - excess is okay

# ============================================================
# N-NOMIAL PARAMETERS
# ============================================================
def calculate_n_nomial_parameters(N, S0, r, dt):
    """Calculate stock price multipliers and risk-neutral probabilities"""
    sigma = 0.2
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Generate multipliers symmetrically
    multipliers = []
    if N == 2:
        multipliers = [u, d]
    elif N == 3:
        multipliers = [u, 1.0, d]
    else:
        for i in range(N):
            exponent = ((N - 1 - 2*i) / (N - 1))
            if exponent > 0:
                multipliers.append(u ** exponent)
            elif exponent < 0:
                multipliers.append(d ** abs(exponent))
            else:
                multipliers.append(1.0)
    
    # Solve for risk-neutral probabilities using linear programming
    growth = np.exp(r * dt)
    c = np.zeros(N)  # Objective (all zeros - just find feasible solution)
    A_eq = np.array([np.ones(N), multipliers])
    b_eq = np.array([1.0, growth])
    
    # Bounds to ensure positive probabilities
    MIN_PROB = 0.01
    max_prob = 1.0 - MIN_PROB * (N - 1)
    bounds = [(MIN_PROB, max_prob) for _ in range(N)]
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        print("WARNING: Could not find valid probabilities, using uniform")
        probabilities = np.ones(N) / N
    else:
        probabilities = result.x
    
    # Validation
    prob_sum = np.sum(probabilities)
    expected_growth = np.sum(probabilities * multipliers)
    min_prob = np.min(probabilities)
    
    print(f"\nN={N}-nomial Parameters:")
    print(f"  Multipliers: {[f'{m:.4f}' for m in multipliers]}")
    print(f"  Probabilities: {[f'{p:.4f}' for p in probabilities]}")
    print(f"  Min probability: {min_prob:.4f}")
    print(f"  Sum of probabilities: {prob_sum:.6f}")
    print(f"  Expected growth: {expected_growth:.6f} (target: {growth:.6f})")
    
    return multipliers, probabilities

multipliers, probabilities = calculate_n_nomial_parameters(N, S0, r, dt)

# ============================================================
# DYNAMIC ACTION_SCALE
# ============================================================
max_stock_price = S0 * max(multipliers) ** T_steps
max_terminal_payoff = max(max_stock_price - K, 0)

# Adaptive scaling based on N and T
if SUPER_REPLICATION:
    CONTINUATION_MULTIPLIER = 3.0 + (T_steps * 1.0) + (N - 2) * 1.5 + 2.0  # Extra for super-rep
else:
    CONTINUATION_MULTIPLIER = 3.0 + (T_steps * 0.7) + (N - 2) * 1.0

ACTION_SCALE = max(300.0, max_terminal_payoff * CONTINUATION_MULTIPLIER)

print("="*60)
print(f"LSMC + UNIVERSAL AGENT: N={N}-NOMIAL, T={T_steps}")
print(f"MODE: SUPER-REPLICATION (Incomplete Market)")
print("="*60)
print(f"LSMC Simulations: {NUM_SIMULATIONS}")
print(f"Polynomial Degree: {POLYNOMIAL_DEGREE}")
print(f"Total Training Episodes: {TOTAL_EPISODES}")
print(f"Iterations: {NUM_ITERATIONS}")
print(f"Action Scale: ±{ACTION_SCALE:.1f}")
print(f"Continuation Multiplier: {CONTINUATION_MULTIPLIER:.1f}×")
print(f"Shortfall Multiplier: {SHORTFALL_MULTIPLIER}× (STRONG)")
print(f"Cost Weight: {COST_WEIGHT} (LOW)")
print(f"Excess Penalty: {EXCESS_PENALTY} (LOW)")
print("="*60)

# ============================================================
# POLYNOMIAL FEATURES
# ============================================================
def create_polynomial_features(X, degree):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples = X.shape[0]
    features = np.ones((n_samples, degree + 1))
    for d in range(1, degree + 1):
        features[:, d] = (X[:, 0] ** d)
    
    return features

# ============================================================
# RIDGE REGRESSION
# ============================================================
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_features = X.shape[1]
        XtX = X.T @ X
        reg_matrix = self.alpha * np.eye(n_features)
        
        try:
            self.coef_ = np.linalg.solve(XtX + reg_matrix, X.T @ y)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(XtX + reg_matrix, X.T @ y, rcond=None)[0]
        
        self.intercept_ = 0
        return self
    
    def predict(self, X):
        X = np.array(X)
        return X @ self.coef_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

# ============================================================
# PATH SIMULATION - N-NOMIAL
# ============================================================
class NnomialPathSimulator:
    """Simulate paths through N-nomial lattice"""
    
    def __init__(self, S0, multipliers, probabilities, T_steps, dt):
        self.S0 = S0
        self.multipliers = multipliers
        self.probabilities = probabilities
        self.N = len(multipliers)
        self.T_steps = T_steps
        self.dt = dt
    
    def simulate_paths(self, num_paths):
        """Simulate random paths through N-nomial lattice"""
        paths = []
        
        for _ in range(num_paths):
            path = []
            S = self.S0
            
            for t in range(self.T_steps + 1):
                path.append({
                    'S': S,
                    't': t,
                    'payoff': max(S - K, 0) if t == self.T_steps else None,
                    'value': None
                })
                
                if t < self.T_steps:
                    # Random transition based on probabilities
                    rand = np.random.random()
                    cumsum = 0
                    for i in range(self.N):
                        cumsum += self.probabilities[i]
                        if rand < cumsum:
                            S = S * self.multipliers[i]
                            break
            
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
            
            X = []
            y = []
            
            for path in paths:
                current_state = path[t]
                future_state = path[t + 1]
                
                S_t = current_state['S']
                V_future = future_state['value']
                
                continuation_value = np.exp(-r * dt) * V_future
                
                X.append(S_t)
                y.append(continuation_value)
            
            X = np.array(X)
            y = np.array(y)
            
            X_poly = create_polynomial_features(X, self.poly_degree)
            
            model = RidgeRegression(alpha=self.alpha)
            model.fit(X_poly, y)
            
            self.continuation_models[t] = model
            
            # Update path values
            for path in paths:
                current_state = path[t]
                S_t = current_state['S']
                X_pred = create_polynomial_features([S_t], self.poly_degree)
                current_state['value'] = model.predict(X_pred)[0]
            
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
# STATE & REWARD - SUPER-REPLICATION
# ============================================================
def construct_state(S, t, target_value):
    return np.array([
        S / S0,
        target_value / max_terminal_payoff if max_terminal_payoff > 0 else 0,
        t / T_steps if T_steps > 0 else 0
    ])

def compute_reward(hedge, target_value, binary_prices, is_terminal):
    """Super-replication reward: strong shortfall penalty, weak excess penalty"""
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    realized = np.sum(hedge)
    
    shortfall = max(0, target_value - realized)
    excess = max(0, realized - target_value)
    
    normalized_shortfall = shortfall / max_terminal_payoff if max_terminal_payoff > 0 else shortfall
    normalized_excess = excess / max_terminal_payoff if max_terminal_payoff > 0 else excess
    
    # Adaptive penalty for high-value states
    if is_terminal and target_value > 30:
        penalty_weight = 200
    elif is_terminal:
        penalty_weight = 100
    else:
        penalty_weight = 150
    
    # SUPER-REPLICATION: Heavy shortfall penalty, light excess penalty
    reward = -(COST_WEIGHT * abs(cost) 
               + penalty_weight * SHORTFALL_MULTIPLIER * normalized_shortfall**2
               + penalty_weight * EXCESS_PENALTY * normalized_excess**2)
    
    reward = np.clip(reward, -10000, 0)
    
    return reward, cost, shortfall

# ============================================================
# TRAINING LOOP
# ============================================================
def train_universal_agent_with_lsmc():
    print("\n" + "="*60)
    print("ITERATIVE LSMC + UNIVERSAL AGENT TRAINING")
    print(f"MODE: SUPER-REPLICATION (N={N}-NOMIAL)")
    print("="*60)
    
    simulator = NnomialPathSimulator(S0, multipliers, probabilities, T_steps, dt)
    lsmc_estimator = LSMCEstimator(polynomial_degree=POLYNOMIAL_DEGREE, alpha=REGRESSION_ALPHA)
    
    state_dim = 3
    action_dim = N  # N binaries for N-nomial (will use 1 for terminals)
    
    agent = UniversalDDPGAgent(state_dim, action_dim, HIDDEN_DIM)
    
    best_total_shortfall = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'='*60}")
        
        # PHASE 1: LSMC
        print(f"\nPhase 1: Simulating {NUM_SIMULATIONS} paths and running LSMC...")
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        # PHASE 2: TRAIN AGENT
        print(f"\nPhase 2: Training universal agent...")
        episodes_this_iter = TOTAL_EPISODES // NUM_ITERATIONS
        
        reward_history = []
        
        for episode in range(episodes_this_iter):
            path_idx = np.random.randint(0, len(paths))
            time_idx = np.random.randint(0, T_steps + 1)
            
            sampled_state = paths[path_idx][time_idx]
            S = sampled_state['S']
            t = sampled_state['t']
            
            is_terminal = (t == T_steps)
            
            if is_terminal:
                target = sampled_state['payoff']
                n_binaries = 1
                prices = [np.exp(-r * dt)]
            else:
                target = sampled_state['value']
                n_binaries = N  # N-nomial has N binaries
                prices = [np.exp(-r * dt)] * N
            
            state = construct_state(S, t, target)
            
            noise_decay = max(0.1, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.8
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            if is_terminal:
                action_used = np.maximum(action[:1], 0)
                prices_used = prices[:1]
            else:
                action_used = action[:n_binaries]
                prices_used = prices[:n_binaries]
            
            reward, cost, shortfall = compute_reward(action_used, target, prices_used, is_terminal)
            
            agent.replay_buffer.push(state, action, reward, state, True)
            
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            reward_history.append(reward)
            
            if (episode + 1) % 5000 == 0:
                avg_reward = np.mean(reward_history[-1000:])
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.2f}")
        
        # PHASE 3: EVALUATE
        print(f"\nPhase 3: Evaluating agent performance...")
        
        total_shortfall = 0
        total_cost = 0
        num_states = 0
        
        for path in paths[:1000]:
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
                    n_binaries = N
                    prices = [np.exp(-r * dt)] * N
                
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
        
        # More lenient threshold for higher N
        success_threshold = 0.5 + (N - 2) * 0.2  # 0.5 for N=2, 0.7 for N=3, 0.9 for N=4
        if avg_shortfall < success_threshold:
            print(f"\n🎉 SUCCESS! Avg Shortfall < {success_threshold} at iteration {iteration + 1}")
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
print("\n" + "="*60)
print("FINAL EVALUATION - SUPER-REPLICATION")
print("="*60)

# Generate test states based on N
test_states = [(S0, 0)]  # Initial state

# t=1 states
for i in range(min(N, 4)):  # Show first 4 or N states, whichever is smaller
    test_states.append((S0 * multipliers[i], 1))

# t=2 states (if T>=2)
if T_steps >= 2:
    # Show some representative t=2 states
    test_states.extend([
        (S0 * multipliers[0] * multipliers[0], 2),  # Best case
        (S0 * multipliers[0] * multipliers[-1], 2), # Mixed
        (S0 * multipliers[-1] * multipliers[-1], 2) # Worst case
    ])

print("\nAgent Performance at Key States:")
print("-" * 80)
for S, t in test_states:
    is_terminal = (t == T_steps)
    
    if is_terminal:
        target = max(S - K, 0)
        n_binaries = 1
        prices = [np.exp(-r * dt)]
    else:
        target = lsmc_estimator.predict_continuation_value(S, t)
        n_binaries = N
        prices = [np.exp(-r * dt)] * N
    
    state = construct_state(S, t, target)
    action = agent.select_action(state, add_noise=False)
    
    if is_terminal:
        action_used = np.maximum(action[:1], 0)
        prices_used = prices[:1]
    else:
        action_used = action[:n_binaries]
        prices_used = prices[:n_binaries]
    
    _, cost, shortfall = compute_reward(action_used, target, prices_used, is_terminal)
    realized = np.sum(action_used)
    excess = max(0, realized - target)
    
    print(f"\nState: S=${S:.2f}, t={t}")
    print(f"  Target: ${target:.4f}")
    print(f"  Hedge: {action_used}")
    print(f"  Realized Value: ${realized:.4f}")
    print(f"  Cost: ${cost:.4f}")
    print(f"  Shortfall: {shortfall:.6f}")
    print(f"  Excess: {excess:.4f}")
    if SUPER_REPLICATION:
        status = "✓ COVERED" if shortfall < 0.01 else "✗ UNCOVERED"
        print(f"  Status: {status}")

print("\n" + "="*60)
print("SUMMARY:")
print(f"N={N}-nomial, T={T_steps}")
print(f"Super-Replication: {SUPER_REPLICATION}")
print(f"LSMC + Universal Agent Framework")
print("="*60)
print("DONE!")
print("="*60)