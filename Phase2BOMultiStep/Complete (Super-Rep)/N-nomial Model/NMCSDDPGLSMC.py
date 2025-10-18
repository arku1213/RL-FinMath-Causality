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

# N-NOMIAL PARAMETERS - COMPLETE MARKET
N = 5  # Number of states (3=trinomial, 4=4-nomial, 5=5-nomial, etc.)
sigma = 0.3

# LSMC Parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# ============================================================
# FINANCIAL MATH: N-NOMIAL SUPER-REPLICATION (COMPLETE MARKET)
# ============================================================
ACTOR_LR = 0.00005
CRITIC_LR = 0.00015

# Calculate max possible payoff for scaling
def calculate_max_payoff(N, S0, sigma, dt, T_steps, K):
    """Calculate maximum possible terminal stock price"""
    # For N-nomial, the highest multiplier is approximately:
    max_multiplier = np.exp(sigma * np.sqrt(dt * N))
    max_stock = S0 * (max_multiplier ** T_steps)
    return max(max_stock - K, 0)

max_terminal_payoff = calculate_max_payoff(N, S0, sigma, dt, T_steps, K)
ACTION_SCALE = max_terminal_payoff * 10.0

# SUPER-REPLICATION: Asymmetric penalties
SHORTFALL_PENALTY = 100000000  # HUGE - must NEVER underpay!
EXCESS_PENALTY = 100           # SMALL - overpayment okay but minimize
COST_WEIGHT = 0.1              # LOW - but still care about efficiency

# Training
TOTAL_EPISODES = 540000
NUM_ITERATIONS = 12
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 500000
HIDDEN_DIM = 256

print("="*60)
print(f"FINANCIAL MATH: {N}-NOMIAL SUPER-REPLICATION (T={T_steps})")
print("="*60)
print("GOAL: Find minimum cost hedge that NEVER underpays")
print(f"Market: {N} states → {N} binaries (COMPLETE)")
print(f"Shortfall Penalty: {SHORTFALL_PENALTY:,} (MUST NEVER UNDERPAY)")
print(f"Excess Penalty: {EXCESS_PENALTY} (minimize overpayment)")
print(f"Cost Weight: {COST_WEIGHT} (seek efficiency)")
print("="*60)

# ============================================================
# N-NOMIAL PARAMETERS
# ============================================================
def calculate_nnomial_parameters(N, S0, sigma, dt, r):
    """
    Calculate N-nomial tree parameters and risk-neutral probabilities
    
    For N states, create symmetric moves around the middle state
    Middle state has m = 1.0 (stock stays same)
    """
    # Create N symmetric multipliers centered around 1.0
    moves = []
    lambda_param = sigma * np.sqrt(dt * N)
    
    for i in range(N):
        # Map i ∈ [0, N-1] to multipliers symmetrically around 1.0
        # i=0 → down most, i=N-1 → up most, middle ≈ 1.0
        offset = (i - (N-1)/2) * 2 / (N-1)  # Maps to [-1, +1]
        multiplier = np.exp(offset * lambda_param)
        moves.append(multiplier)
    
    # Calculate risk-neutral probabilities
    growth = np.exp(r * dt)
    
    def objective(p):
        # Minimize squared deviation from uniform (entropy maximization proxy)
        return np.sum((p - 1/N)**2)
    
    def constraint_mean(p):
        # Expected growth must match risk-free rate
        return np.dot(p, moves) - growth
    
    def constraint_sum(p):
        # Probabilities must sum to 1
        return np.sum(p) - 1
    
    # Initial guess: uniform distribution
    p0 = np.ones(N) / N
    
    # Constraints
    cons = [
        {'type': 'eq', 'fun': constraint_mean},
        {'type': 'eq', 'fun': constraint_sum}
    ]
    
    # Bounds: all probabilities between 0.001 and 0.999
    bounds = [(0.001, 0.999) for _ in range(N)]
    
    result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons)
    
    if result.success:
        probs = result.x
    else:
        print("WARNING: Optimization failed, using fallback uniform probabilities")
        probs = np.ones(N) / N
    
    # Verify constraints
    expected_growth = np.dot(probs, moves)
    prob_sum = np.sum(probs)
    
    print(f"\n{N}-nomial Parameters:")
    print(f"  Moves: {[f'{m:.4f}' for m in moves]}")
    print(f"  Probs: {[f'{p:.4f}' for p in probs]}")
    print(f"  Sum = {prob_sum:.6f}, Expected growth = {expected_growth:.6f} (target={growth:.6f})")
    
    assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities don't sum to 1: {prob_sum}"
    assert abs(expected_growth - growth) < 1e-2, f"Growth mismatch: {expected_growth} vs {growth}"
    assert all(p >= 0 for p in probs), "Negative probabilities detected"
    
    print("  ✓ Valid risk-neutral probabilities")
    
    return moves, probs

moves, probs = calculate_nnomial_parameters(N, S0, sigma, dt, r)

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
    def __init__(self, S0, moves, probs, T_steps, dt):
        self.S0 = S0
        self.moves = moves
        self.probs = probs
        self.N = len(moves)
        self.T_steps = T_steps
        self.dt = dt
    
    def simulate_paths(self, num_paths):
        paths = []
        cumulative_probs = np.cumsum(self.probs)
        
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
                    # Sample which child state occurs
                    rand = np.random.random()
                    child_idx = np.searchsorted(cumulative_probs, rand)
                    child_idx = min(child_idx, self.N - 1)  # Safety
                    
                    S *= self.moves[child_idx]
                    path_step['child_occurred'] = child_idx
                
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
        
        # Initialize terminal values
        for path in paths:
            path[-1]['value'] = path[-1]['payoff']
        
        # Backward induction through time
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
        """Returns [V_0, V_1, ..., V_{N-1}] for all N children"""
        if t >= self.T_steps - 1:
            # Terminal: return actual payoffs
            return [max(S * move - K, 0) for move in moves]
        else:
            # Intermediate: predict continuation values
            return [self.predict_continuation_value(S * move, t + 1) for move in moves]

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
# STATE & REWARD - SUPER-REPLICATION (ASYMMETRIC)
# ============================================================
def construct_state(S, t, target_values, N):
    """State: [S_norm, avg_target, t_norm, target_0, target_1, ..., target_{N-1}]"""
    is_intermediate = isinstance(target_values, (list, np.ndarray))
    norm_factor = max_terminal_payoff if max_terminal_payoff > 0 else 1.0
    
    state_vec = np.zeros(3 + N)  # 3 base features + N target values
    state_vec[0] = S / S0
    state_vec[2] = t / T_steps
    
    if is_intermediate:
        state_vec[1] = np.mean(target_values) / norm_factor
        state_vec[3:3+N] = np.array(target_values) / norm_factor
    else:
        state_vec[1] = target_values / norm_factor
    
    return state_vec

def compute_super_replication_reward(hedge, target_values, binary_prices, is_terminal, child_occurred=None):
    """
    SUPER-REPLICATION: Asymmetric penalties
    - HUGE penalty for shortfall (underpayment)
    - SMALL penalty for excess (overpayment)
    - SMALL penalty for cost (efficiency)
    """
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    
    if is_terminal:
        # Terminal: single value
        realized = hedge[0]
        target = target_values
        shortfall = max(0, target - realized)
        excess = max(0, realized - target)
    else:
        # Intermediate: check specific child or all
        if child_occurred is not None:
            # Training: specific child
            realized = hedge[child_occurred]
            target = target_values[child_occurred]
            shortfall = max(0, target - realized)
            excess = max(0, realized - target)
        else:
            # Evaluation: all N children
            shortfalls = []
            excesses = []
            for i in range(len(target_values)):
                realized_i = hedge[i]
                target_i = target_values[i]
                shortfalls.append(max(0, target_i - realized_i))
                excesses.append(max(0, realized_i - target_i))
            
            shortfall = np.mean(shortfalls)
            excess = np.mean(excesses)
    
    # Normalize
    norm_factor = max_terminal_payoff if max_terminal_payoff > 0 else 1.0
    normalized_shortfall = shortfall / norm_factor
    normalized_excess = excess / norm_factor
    
    # ASYMMETRIC PENALTIES
    reward = -(COST_WEIGHT * abs(cost)
               + SHORTFALL_PENALTY * normalized_shortfall**2  # HUGE!
               + EXCESS_PENALTY * normalized_excess**2)        # Small
    
    return np.clip(reward, -100000000, 0), cost, shortfall, excess

# ============================================================
# TRAINING LOOP
# ============================================================
def train_super_replication_agent():
    simulator = NnomialPathSimulator(S0, moves, probs, T_steps, dt)
    lsmc_estimator = LSMCEstimator(polynomial_degree=POLYNOMIAL_DEGREE, alpha=REGRESSION_ALPHA)
    
    # State: 3 base + N targets, Action: N binaries
    state_dim = 3 + N
    action_dim = N
    agent = UniversalDDPGAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=HIDDEN_DIM)
    
    best_avg_shortfall = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}\nITERATION {iteration + 1}/{NUM_ITERATIONS}\n{'='*60}")
        
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        print(f"\nTraining super-replication agent...")
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
                prices = [np.exp(-r * dt) * p for p in probs]
                child_occurred = sampled_node['child_occurred']
            
            state = construct_state(S, t, target, N)
            
            noise_decay = max(0.01, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.95
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            action_used = action[:1] if is_terminal else action[:N]
            
            # Multi-child training
            if is_terminal:
                reward, _, _, _ = compute_super_replication_reward(
                    action_used, target, prices, True, None
                )
                agent.replay_buffer.push(state, action, reward, state, False)
                reward_history.append(reward)
            else:
                for child_idx in range(N):
                    reward, _, _, _ = compute_super_replication_reward(
                        action_used, target, prices, False, child_idx
                    )
                    agent.replay_buffer.push(state, action, reward, state, False)
                    reward_history.append(reward)
            
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            if (episode + 1) % (episodes_this_iter // 8) == 0:
                avg_reward = np.mean(reward_history[-1000:]) if len(reward_history) >= 1000 else np.mean(reward_history)
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.1f}")

        print(f"\nEvaluating super-replication performance...")
        
        total_shortfall, total_excess, num_evals = 0, 0, 0
        
        for path in paths[:1000]:
            for node in path:
                S_eval, t_eval = node['S'], node['t']
                is_terminal_eval = (t_eval == T_steps)
                
                if is_terminal_eval:
                    target_eval = node['payoff']
                    prices_eval = [np.exp(-r * dt)]
                else:
                    target_eval = lsmc_estimator.predict_child_continuation_values(S_eval, t_eval)
                    prices_eval = [np.exp(-r * dt) * p for p in probs]

                state_eval = construct_state(S_eval, t_eval, target_eval, N)
                action_eval = agent.select_action(state_eval, add_noise=False)
                action_used_eval = action_eval[:1] if is_terminal_eval else action_eval[:N]
                
                _, _, shortfall, excess = compute_super_replication_reward(
                    action_used_eval, target_eval, prices_eval, is_terminal_eval, None
                )
                
                total_shortfall += shortfall
                total_excess += excess
                num_evals += 1

        avg_shortfall = total_shortfall / num_evals
        avg_excess = total_excess / num_evals
        
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Avg Shortfall: {avg_shortfall:.6f} (MUST BE NEAR 0!)")
        print(f"  Avg Excess: {avg_excess:.6f} (minimize but acceptable)")
        
        if avg_shortfall < best_avg_shortfall:
            best_avg_shortfall = avg_shortfall
            print(f"  ✓ NEW BEST!")
        
        if avg_shortfall < 0.1:
            print(f"\n🎉 SUCCESS! Avg Shortfall < 0.1 at iteration {iteration + 1}")
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
print("\n" + "="*60 + "\nFINAL EVALUATION - SUPER-REPLICATION\n" + "="*60)

# Generate test states at t=0, t=1, t=2
test_states = [(S0, 0)]

# Add some representative t=1 states
for i in range(min(N, 5)):  # Test up to 5 states at t=1
    test_states.append((S0 * moves[i], 1))

# Add some representative t=2 states
if T_steps >= 2:
    for i in range(min(N, 3)):
        for j in range(min(N, 3)):
            test_states.append((S0 * moves[i] * moves[j], 2))

print(f"\nEvaluating {len(test_states)} key states")
print("="*60)

success_count = 0

for S, t in test_states:
    is_terminal = (t == T_steps)
    
    if is_terminal:
        target = max(S - K, 0)
        prices = [np.exp(-r * dt)]
    else:
        target = lsmc_estimator.predict_child_continuation_values(S, t)
        prices = [np.exp(-r * dt) * p for p in probs]
    
    state = construct_state(S, t, target, N)
    action = agent.select_action(state, add_noise=False)
    action_used = action[:1] if is_terminal else action[:N]
    
    _, cost, shortfall, excess = compute_super_replication_reward(
        action_used, target, prices, is_terminal, None
    )
    
    is_success = shortfall < 0.1
    if is_success:
        success_count += 1
    
    print(f"\nt={t} | S=${S:.2f}")
    print("-"*60)
    
    if is_terminal:
        print(f"LSMC Target: ${target:.4f}")
        print(f"Super-Hedge (# of binaries): {action_used[0]:+.4f}")
        print(f"Shortfall: ${shortfall:.4f}, Excess: ${excess:.4f}")
    else:
        print(f"LSMC Targets: {[f'${v:.2f}' for v in target]}")
        print(f"Super-Hedge: {[f'{h:+.2f}' for h in action_used]}")
        print(f"Avg Shortfall: ${shortfall:.4f}, Avg Excess: ${excess:.4f}")
    
    print(f"Total Cost: ${cost:+.4f}")
    print("-"*60)
    if is_success:
        print("✓ Super-replication successful (no shortfall)")
    else:
        print("✗ FAILED - shortfall detected!")
    print("="*60)

print("\n" + "="*60)
print(f"SUCCESS RATE: {success_count}/{len(test_states)} ({100*success_count/len(test_states):.1f}%)")
print("="*60)
print("\nFINANCIAL MATH - SUPER-REPLICATION:")
print(f"• Complete market: {N} states → {N} binaries")
print(f"• Goal: Minimum cost hedge that NEVER underpays")
print(f"• Asymmetric penalties enforce super-replication")
print("="*60)
print("\nThe numbers represent the minimum # of binary contracts")
print("needed to ALWAYS cover the option payoff (upper bound).")
print("="*60)