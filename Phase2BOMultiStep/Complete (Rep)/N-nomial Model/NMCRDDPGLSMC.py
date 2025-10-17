import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math
from scipy.optimize import linprog

# ============================================================
# CONFIGURATION
# ============================================================
S0 = 100.0
r = 0.05
K = 100.0
T_steps = 2
dt = 1.0

# N-NOMIAL PARAMETER - CHANGE THIS!
N = 4  # Number of states (2=binomial, 3=trinomial, 4=quadrinomial, 5=pentanomial, etc.)

# Stock price multipliers will be calculated automatically
sigma = 0.2  # Volatility parameter

# LSMC Parameters
NUM_SIMULATIONS = 10000
POLYNOMIAL_DEGREE = 3
REGRESSION_ALPHA = 0.1

# ============================================================
# AGGRESSIVE HYPERPARAMETERS FOR COMPLETE MARKET
# ============================================================
ACTOR_LR = 0.0001
CRITIC_LR = 0.0003

# Action scale based on N
max_stock_price = S0 * (np.exp(sigma * np.sqrt(2 * dt)) ** (T_steps))
max_terminal_payoff = max(max_stock_price - K, 0)
ACTION_SCALE = max_terminal_payoff * 3.0

# CRITICAL: Make accuracy MUCH more important than cost
REPLICATION_PENALTY = 10000
EXTREME_PENALTY_WEIGHT = 100
COST_WEIGHT = 0.0001

# Training schedule
TOTAL_EPISODES = 300000
NUM_ITERATIONS = 12
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 300000
HIDDEN_DIM = 256

print("="*60)
print(f"LSMC + DDPG: {N}-NOMIAL (N={N}) T={T_steps}")
print(f"MODE: COMPLETE MARKET - PERFECT REPLICATION")
print("="*60)
print(f"States per node: {N}")
print(f"Binaries per node: {N} (COMPLETE MARKET)")
print(f"Replication Penalty: {REPLICATION_PENALTY}")
print(f"Cost Weight: {COST_WEIGHT}")
print(f"Accuracy/Cost Ratio: {REPLICATION_PENALTY/COST_WEIGHT:,.0f}:1")
print("="*60)

# ============================================================
# N-NOMIAL PARAMETERS - AUTOMATIC CALCULATION
# ============================================================
def calculate_n_nomial_parameters(N, S0, r, sigma, dt):
    """
    Calculate stock price multipliers and risk-neutral probabilities
    for N-nomial tree
    """
    growth = np.exp(r * dt)
    
    # Create N multipliers symmetrically spaced in log-space
    if N == 2:
        # Binomial
        u = np.exp(sigma * np.sqrt(dt))
        multipliers = [u, 1/u]
    elif N == 3:
        # Trinomial (CRR style)
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = 1 / u
        multipliers = [u, 1.0, d]
    else:
        # General N-nomial: symmetric spacing
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = 1 / u
        
        # Linearly spaced exponents from positive to negative
        exponents = np.linspace((N-1)/2, -(N-1)/2, N)
        base = u
        multipliers = [base ** exp for exp in exponents]
    
    multipliers = np.array(multipliers)
    
    # Calculate risk-neutral probabilities using linear programming
    # We want: p_0*m_0 + p_1*m_1 + ... + p_{N-1}*m_{N-1} = growth
    #          p_0 + p_1 + ... + p_{N-1} = 1
    #          All p_i >= min_prob (for numerical stability)
    
    min_prob = 0.001  # Minimum probability for stability
    
    # Objective: minimize variance (or just find feasible solution)
    # We'll use linprog to find probabilities
    
    # For N unknowns with 2 equality constraints, we have N-2 degrees of freedom
    # We'll minimize sum of squared deviations from uniform
    # This is equivalent to: minimize sum((p_i - 1/N)^2)
    
    # Use a simple heuristic: solve the system with additional constraint of minimum entropy
    # For simplicity, we'll use least squares with constraints
    
    # Set up: A_eq @ p = b_eq, bounds
    A_eq = np.array([
        np.ones(N),      # sum(p_i) = 1
        multipliers      # sum(p_i * m_i) = growth
    ])
    b_eq = np.array([1.0, growth])
    
    # Bounds: each probability between min_prob and 1-min_prob*(N-1)
    bounds = [(min_prob, 1.0 - min_prob * (N-1)) for _ in range(N)]
    
    # Objective: minimize distance from uniform (for numerical stability)
    c = np.ones(N)  # We'll solve a feasibility problem
    
    # Solve using linprog
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        probabilities = result.x
    else:
        print("WARNING: Could not solve for probabilities. Using fallback.")
        # Fallback: approximate with uniform then adjust
        probabilities = np.ones(N) / N
        # Adjust to match growth constraint
        current_growth = np.sum(probabilities * multipliers)
        adjustment = (growth - current_growth) / np.sum(multipliers)
        probabilities = probabilities + adjustment * multipliers / np.sum(multipliers**2)
        probabilities = np.maximum(probabilities, min_prob)
        probabilities = probabilities / np.sum(probabilities)
    
    # Validate
    prob_sum = np.sum(probabilities)
    expected_growth = np.sum(probabilities * multipliers)
    
    print(f"\n{N}-nomial Probabilities:")
    for i, (m, p) in enumerate(zip(multipliers, probabilities)):
        print(f"  State {i}: multiplier={m:.6f}, probability={p:.6f}")
    print(f"  Sum of probabilities = {prob_sum:.6f}")
    print(f"  Expected growth = {expected_growth:.6f} (target: {growth:.6f})")
    
    # Check for issues
    if not (0.999 < prob_sum < 1.001):
        print(f"WARNING: Probabilities don't sum to 1! Sum = {prob_sum}")
    if not (0.99 * growth < expected_growth < 1.01 * growth):
        print(f"WARNING: Expected growth mismatch!")
    if np.any(probabilities < 0):
        print(f"WARNING: Negative probabilities found!")
    
    return multipliers, probabilities

multipliers, probabilities = calculate_n_nomial_parameters(N, S0, r, sigma, dt)

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
# N-NOMIAL PATH SIMULATION WITH CHILD TRACKING
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
                    'child_occurred': None
                }
                
                if t < self.T_steps:
                    # Random transition based on probabilities
                    child_occurred = np.random.choice(self.N, p=self.probabilities)
                    S *= self.multipliers[child_occurred]
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
        
        print("LSMC ESTIMATION COMPLETE\n" + "="*60)
        return paths
    
    def predict_continuation_value(self, S, t):
        if t not in self.continuation_models: 
            return 0.0
        X = create_polynomial_features([S], self.poly_degree)
        return self.continuation_models[t].predict(X)[0]
    
    def predict_child_continuation_values(self, S, t):
        """Returns continuation values for all N children"""
        if t >= self.T_steps - 1:
            # Terminal children
            return [max(S * m - K, 0) for m in multipliers]
        else:
            # Use LSMC model
            return [self.predict_continuation_value(S * m, t + 1) for m in multipliers]

# ============================================================
# UNIVERSAL DDPG AGENT (GENERALIZED FOR N ACTIONS)
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
# STATE & REWARD FOR N-NOMIAL
# ============================================================
def construct_state(S, t, target_values):
    """
    State vector: [S_norm, avg_target_norm, t_norm, target_0_norm, ..., target_{N-1}_norm]
    Dimension: 3 + N
    """
    is_intermediate = isinstance(target_values, (list, np.ndarray))
    norm_factor = max_terminal_payoff if max_terminal_payoff > 0 else 1.0
    
    state_vec = np.zeros(3 + N)
    state_vec[0] = S / S0
    state_vec[2] = t / T_steps
    
    if is_intermediate:
        state_vec[1] = np.mean(target_values) / norm_factor
        state_vec[3:3+N] = np.array(target_values) / norm_factor
    else:
        state_vec[1] = target_values / norm_factor
    
    return state_vec

def compute_reward(hedge, target_values, binary_prices, is_terminal, child_occurred=None):
    """
    PERFECT REPLICATION REWARD for N-nomial complete market
    """
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    
    if is_terminal:
        deviation = abs(hedge[0] - target_values)
        extreme_penalty = 0
        
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
            # Evaluation mode: check all N
            deviations = [abs(hedge[i] - target_values[i]) for i in range(N)]
            deviation = np.mean(deviations)
            penalty_multiplier = 1.0
        
        # Extreme position penalty
        all_targets_near_zero = all(abs(tv) < 0.1 for tv in target_values)
        if all_targets_near_zero:
            max_reasonable = 2.0
        else:
            max_reasonable = max([abs(tv) * 3.0 for tv in target_values] + [10.0])
        
        extreme_penalty = sum(max(0, abs(h) - max_reasonable)**2 for h in hedge)
    
    normalized_deviation = deviation / (max_terminal_payoff if max_terminal_payoff > 0 else 1.0)
    
    reward = -(COST_WEIGHT * abs(cost) 
               + penalty_multiplier * REPLICATION_PENALTY * normalized_deviation**2
               + EXTREME_PENALTY_WEIGHT * extreme_penalty)
    
    return np.clip(reward, -100000, 0), cost, deviation

# ============================================================
# TRAINING LOOP
# ============================================================
def train_universal_agent_with_lsmc():
    simulator = NnomialPathSimulator(S0, multipliers, probabilities, T_steps, dt)
    lsmc_estimator = LSMCEstimator(polynomial_degree=POLYNOMIAL_DEGREE, alpha=REGRESSION_ALPHA)
    
    # State dim: 3 + N, Action dim: N
    agent = UniversalDDPGAgent(state_dim=3+N, action_dim=N, hidden_dim=HIDDEN_DIM)
    
    best_avg_deviation = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}\nITERATION {iteration + 1}/{NUM_ITERATIONS}\n{'='*60}")
        
        paths = simulator.simulate_paths(NUM_SIMULATIONS)
        paths = lsmc_estimator.estimate_continuation_values(paths, r, dt)
        
        print(f"\nTraining agent for {N}-nomial complete market...")
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
                prices = [np.exp(-r * dt) * p for p in probabilities]
                child_occurred = sampled_node['child_occurred']
            
            state = construct_state(S, t, target)
            
            noise_decay = max(0.05, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.9
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            action_used = action[:1] if is_terminal else action[:N]
            
            reward, _, _ = compute_reward(action_used, target, prices, is_terminal, child_occurred)
            reward_history.append(reward)
            
            for child_idx in range(N):
                reward, _, _ = compute_reward(
                    action_used, target, prices, is_terminal, child_idx
                )
                agent.replay_buffer.push(state, action, reward, state, False)
            
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            if (episode + 1) % (episodes_this_iter // 8) == 0:
                avg_reward = np.mean(reward_history[-1000:]) if len(reward_history) >= 1000 else np.mean(reward_history)
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.1f}")

        print(f"\nEvaluating on paths...")
        
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
                    prices_eval = [np.exp(-r * dt) * p for p in probabilities]
                    child_eval = node['child_occurred']

                state_eval = construct_state(S_eval, t_eval, target_eval)
                action_eval = agent.select_action(state_eval, add_noise=False)
                action_used_eval = action_eval[:1] if is_terminal_eval else action_eval[:N]
                
                _, _, deviation = compute_reward(action_used_eval, target_eval, prices_eval, is_terminal_eval, child_eval)
                
                total_deviation += deviation
                num_evals += 1

        avg_deviation = total_deviation / num_evals
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Average Deviation: {avg_deviation:.6f}")
        
        if avg_deviation < best_avg_deviation:
            best_avg_deviation = avg_deviation
            print(f"  ✓ NEW BEST!")
        
        if avg_deviation < 1.0:
            print(f"\n🎉 SUCCESS! Avg Deviation < 1.0 at iteration {iteration + 1}")
            break
            
    print(f"\n{'='*60}\nTRAINING COMPLETE!\n{'='*60}")
    print(f"Best Average Deviation: {best_avg_deviation:.6f}")
    print(f"Target: < 1.0 for success")
    return agent, lsmc_estimator

# ============================================================
# MAIN EXECUTION
# ============================================================
agent, lsmc_estimator = train_universal_agent_with_lsmc()

# ============================================================
# TRAINING COMPLETE — CONSISTENT REPORT FORMAT
# ============================================================

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("Target was: < 1.0 for success\n")

# ============================================================
# FINAL EVALUATION — Unified Hedge Report
# ============================================================

print("FINAL EVALUATION")
print("="*60)

# pick representative evaluation states (you can change)
eval_states = [100.0, 120.0, 80.0]  # sample underlying prices
successes, total_nodes = 0, 0

for S_eval in eval_states:
    for t_eval in range(T_steps + 1):
        # --- Get LSMC target continuation values ---
        try:
            target_values = lsmc_estimator.predict_child_continuation_values(S_eval, t_eval)
        except Exception:
            # terminal node → direct payoff
            target_values = [max(S_eval - K, 0.0)]

        target_values = np.atleast_1d(target_values)

        # --- Agent hedge decision ---
        state_input = construct_state(S_eval, t_eval, target_values)
        hedge = agent.select_action(state_input, add_noise=False)
        hedge = np.atleast_1d(hedge)

        # --- Price each binary (risk-neutral) ---
        if hedge.shape[0] == 1:
            binary_prices = [np.exp(-r * dt)]
        else:
            # Prefer using the globally computed `probabilities` if its length matches or exceeds the action dim;
            # otherwise fall back to uniform probabilities.
            try:
                if len(probabilities) >= hedge.shape[0]:
                    p_list = probabilities[:hedge.shape[0]].tolist()
                else:
                    p_list = [1.0 / hedge.shape[0]] * hedge.shape[0]
            except NameError:
                # If `probabilities` is somehow not defined, use uniform probabilities.
                p_list = [1.0 / hedge.shape[0]] * hedge.shape[0]
            binary_prices = [np.exp(-r * dt) * float(p) for p in p_list]

        # --- Compute hedge cost & deviations ---
        cost = float(np.sum(np.array(hedge) * np.array(binary_prices)))
        deviations = np.abs(np.array(hedge) - np.array(target_values))
        avg_dev = float(np.mean(deviations))

        # --- Print structured result ---
        print(f"\nState: S=${S_eval:.2f}, t={t_eval}")
        print(f"  Targets: {[round(v, 4) for v in target_values]}")
        print(f"  Hedges:  {[round(h, 4) for h in hedge]}")
        print(f"  Binary Prices: {[round(b, 6) for b in binary_prices]}")
        print(f"  Cost: ${cost:.4f}")
        print(f"  Individual Devs: {[round(d, 4) for d in deviations]}")
        print(f"  Average Deviation: {avg_dev:.6f} {'✓' if avg_dev < 1.0 else '✗'}")

        total_nodes += 1
        if avg_dev < 1.0:
            successes += 1

# ============================================================
# SUMMARY
# ============================================================
success_rate = (successes / total_nodes * 100) if total_nodes > 0 else 0
print("\n" + "="*60)
print(f"SUCCESS RATE: {successes}/{total_nodes} ({success_rate:.1f}%)")
print("="*60 + "\n")
