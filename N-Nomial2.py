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
T_steps = 2  # ← CHANGE THIS: Time steps

# Super-replication mode
SUPER_REPLICATION = True
SHORTFALL_MULTIPLIER = 300 + T_steps * 150  # REDUCED from 500 + 250*T

# UNIVERSAL AGENT Hyperparameters
TOTAL_EPISODES = 150000  # INCREASED from 100k
NUM_ITERATIONS = 6  # INCREASED from 5
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0003
CRITIC_LR = 0.0008
BUFFER_SIZE = 200000
HIDDEN_DIM = 256

# Reward shaping
def get_penalty_weight(node_id, tree):
    """Adaptive penalty based on node type and payoff size"""
    if len(tree[node_id]['children']) > 0:
        return 150
    else:
        payoff = tree[node_id]['payoff']
        if payoff > 150:
            return 250
        elif payoff > 50:
            return 150
        elif payoff > 15:
            return 100
        else:
            return 10

COST_WEIGHT = 0.01

# ============================================================
# N-NOMIAL TREE PARAMETERS
# ============================================================
def calculate_n_nomial_parameters(N, S0, r, dt):
    """Calculate stock price multipliers and risk-neutral probabilities"""
    sigma = 0.2
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
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
    
    growth = np.exp(r * dt)
    c = np.zeros(N)
    A_eq = np.array([np.ones(N), multipliers])
    b_eq = np.array([1.0, growth])
    
    MIN_PROB = 0.03
    max_prob = 1.0 - MIN_PROB * (N - 1)
    bounds = [(MIN_PROB, max_prob) for _ in range(N)]
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        MIN_PROB = 0.01
        max_prob = 1.0 - MIN_PROB * (N - 1)
        bounds = [(MIN_PROB, max_prob) for _ in range(N)]
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            probabilities = np.ones(N) / N
        else:
            probabilities = result.x
    else:
        probabilities = result.x
    
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

# FIX 1: INCREASED ACTION_SCALE with more aggressive multiplier
max_stock_price = S0 * max(multipliers) ** T_steps
max_terminal_payoff = max(max_stock_price - K, 0)
CONTINUATION_MULTIPLIER = 5.0 + (T_steps * 1.5) + (N - 2) * 1.0  # INCREASED
ACTION_SCALE = max(300.0, max_terminal_payoff * CONTINUATION_MULTIPLIER)  # INCREASED base

print("="*60)
print(f"ITERATIVE UNIVERSAL AGENT: N={N}-NOMIAL, T={T_steps}")
print("="*60)
print(f"Super-Replication: {SUPER_REPLICATION}")
print(f"Total Training Episodes: {TOTAL_EPISODES}")
print(f"Iterations: {NUM_ITERATIONS}")
print(f"Episodes per Iteration: {TOTAL_EPISODES // NUM_ITERATIONS}")
print(f"Shortfall Multiplier: {SHORTFALL_MULTIPLIER}")
print(f"Continuation Multiplier: {CONTINUATION_MULTIPLIER:.1f}×")
print(f"Action Scale (Dynamic): ±{ACTION_SCALE:.1f}")
print(f"States per Node: {N}")
print("="*60)

# ============================================================
# N-NOMIAL TREE BUILDER
# ============================================================
def get_node_id_from_state(state_vector, t):
    """Map state vector to unique node_id"""
    nodes_before = 0
    for time in range(t):
        from math import comb
        nodes_before += comb(time + N - 1, N - 1)
    
    position = 0
    for candidate in generate_all_states_at_time(t, N):
        if candidate == state_vector:
            break
        position += 1
    
    return nodes_before + position

def generate_all_states_at_time(t, N):
    """Generate all possible state vectors at time t"""
    if N == 1:
        return [(t,)]
    
    states = []
    
    def generate_recursive(remaining, current_state, depth):
        if depth == N - 1:
            states.append(tuple(current_state + [remaining]))
            return
        for i in range(remaining + 1):
            generate_recursive(remaining - i, current_state + [i], depth + 1)
    
    generate_recursive(t, [], 0)
    return states

def build_n_nomial_tree(T_steps, S0, K, multipliers, N):
    """Build N-nomial tree with recombining"""
    tree = {}
    node_id = 0
    
    for t in range(T_steps + 1):
        states_at_t = generate_all_states_at_time(t, N)
        
        for state_vector in states_at_t:
            S = S0
            for i, count in enumerate(state_vector):
                S *= (multipliers[i] ** count)
            
            tree[node_id] = {
                'S': S,
                'time': t,
                'state': state_vector
            }
            
            if t < T_steps:
                children = []
                for next_state_idx in range(N):
                    child_state = list(state_vector)
                    child_state[next_state_idx] += 1
                    child_state = tuple(child_state)
                    child_id = get_node_id_from_state(child_state, t + 1)
                    children.append(child_id)
                tree[node_id]['children'] = children
            else:
                tree[node_id]['children'] = []
                tree[node_id]['payoff'] = max(S - K, 0)
            
            node_id += 1
    
    return tree

tree = build_n_nomial_tree(T_steps, S0, K, multipliers, N)

terminal_nodes = [n for n, data in tree.items() if len(data['children']) == 0]
max_payoff = max(tree[n]['payoff'] for n in terminal_nodes)

print(f"\nN={N}-nomial Tree with T={T_steps}")
print(f"Total nodes: {len(tree)}")
print(f"Non-terminal nodes: {len(tree) - len(terminal_nodes)}")
print(f"Terminal nodes: {len(terminal_nodes)}")
print(f"Max Payoff: ${max_payoff:.2f}")

# ============================================================
# UNIVERSAL AGENT NEURAL NETWORKS
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
        
        # FIX 5: Better initialization with slight positive bias
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)
        nn.init.uniform_(self.fc4.bias, 0.0, 0.01)  # CHANGED: Positive bias
    
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

# ============================================================
# OU NOISE
# ============================================================
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

# ============================================================
# REPLAY BUFFER
# ============================================================
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

# ============================================================
# UNIVERSAL DDPG AGENT
# ============================================================
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
# STATE CONSTRUCTION FOR UNIVERSAL AGENT
# ============================================================
def construct_state(node_id, target_payoff, tree):
    """
    Construct state for universal agent
    State includes:
    - Normalized target payoff
    - Normalized stock price
    - Normalized time
    """
    S = tree[node_id]['S']
    t = tree[node_id]['time']
    
    # Normalize features
    normalized_payoff = target_payoff / max_payoff if max_payoff > 0 else 0
    normalized_price = S / S0
    normalized_time = t / T_steps
    
    state = np.array([
        normalized_payoff,
        normalized_price,
        normalized_time
    ])
    
    return state

# ============================================================
# REWARD FUNCTION WITH FIXES
# ============================================================
def compute_reward(hedge, target_payoff, binary_prices, node_id, tree):
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    realized = np.sum(hedge)
    
    shortfall = max(0, target_payoff - realized)
    excess = max(0, realized - target_payoff)
    
    penalty_weight = get_penalty_weight(node_id, tree)
    
    terminal_nodes_list = [n for n, data in tree.items() if len(data['children']) == 0]
    max_payoff_global = max(tree[n]['payoff'] for n in terminal_nodes_list)
    
    normalized_shortfall = shortfall / max_payoff_global if max_payoff_global > 0 else shortfall
    normalized_excess = excess / max_payoff_global if max_payoff_global > 0 else excess
    
    if SUPER_REPLICATION:
        reward = -(COST_WEIGHT * abs(cost) 
                   + penalty_weight * SHORTFALL_MULTIPLIER * normalized_shortfall**2
                   + penalty_weight * 0.01 * normalized_excess**2)
    else:
        violation = max(0, target_payoff - realized)
        normalized_violation = violation / max_payoff_global if max_payoff_global > 0 else violation
        reward = -(COST_WEIGHT * abs(cost) + penalty_weight * normalized_violation**2)
    
    # FIX 2: Clip reward to prevent explosion
    reward = np.clip(reward, -10000, 0)
    
    violation = shortfall
    return reward, cost, violation

# ============================================================
# SOLVE TREE HELPER FUNCTION
# ============================================================
def solve_tree_with_agent(agent, tree, probabilities):
    """
    Use agent to solve tree via backward induction
    Returns results dict
    """
    results = {}
    
    for t in range(T_steps, -1, -1):
        nodes_at_time = [n for n, data in tree.items() if data['time'] == t]
        
        for node_id in nodes_at_time:
            node_data = tree[node_id]
            children = node_data['children']
            
            if len(children) == 0:
                target = node_data['payoff']
                n_binaries = 1
                prices = [np.exp(-r * dt)]
            else:
                child_values = [results[c]['cost'] + np.sum(results[c]['hedge']) for c in children]
                target = np.sum([probabilities[i] * child_values[i] for i in range(N)])
                n_binaries = N
                prices = [np.exp(-r * dt) for _ in range(N)]
            
            state = construct_state(node_id, target, tree)
            action = agent.select_action(state, add_noise=False)
            hedge = action[:n_binaries]
            
            # FIX 3: Ensure terminal nodes have non-negative hedges
            if len(children) == 0:
                hedge = np.maximum(hedge, 0)
            
            _, cost, violation = compute_reward(hedge, target, prices, node_id, tree)
            
            results[node_id] = {
                'hedge': hedge,
                'cost': cost,
                'violation': violation
            }
    
    return results

# ============================================================
# ITERATIVE UNIVERSAL AGENT TRAINING
# ============================================================
def train_universal_agent_iterative(tree, probabilities):
    """
    Train universal agent with iterative refinement:
    1. Initialize with placeholder values
    2. Train agent
    3. Solve tree to get real continuation values
    4. Retrain agent with updated values
    5. Repeat until convergence
    """
    print("\n" + "="*60)
    print("ITERATIVE UNIVERSAL AGENT TRAINING")
    print("="*60)
    
    all_nodes = list(tree.keys())
    non_terminal_nodes = [n for n in all_nodes if len(tree[n]['children']) > 0]
    
    print(f"Training on {len(non_terminal_nodes)} non-terminal nodes")
    print(f"Plus {len(terminal_nodes)} terminal nodes")
    print(f"Total training episodes: {TOTAL_EPISODES}")
    print(f"Iterations: {NUM_ITERATIONS}")
    
    state_dim = 3
    action_dim = N
    
    # Initialize agent
    agent = UniversalDDPGAgent(state_dim, action_dim, HIDDEN_DIM)
    
    # Initialize node targets with placeholders
    node_targets = {}
    for node_id in tree.keys():
        if len(tree[node_id]['children']) == 0:
            node_targets[node_id] = tree[node_id]['payoff']
        else:
            node_targets[node_id] = max_payoff * 0.5
    
    best_total_shortfall = float('inf')
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'='*60}")
        
        # Phase 1: Train agent on current targets
        episodes_this_iter = TOTAL_EPISODES // NUM_ITERATIONS
        warmup_episodes = episodes_this_iter // 5
        
        print(f"Training for {episodes_this_iter} episodes...")
        
        reward_history = []
        
        for episode in range(episodes_this_iter):
            # Sample node
            if episode < warmup_episodes:
                # Warmup: more terminals
                if random.random() < 0.7:
                    node_id = random.choice(terminal_nodes)
                else:
                    node_id = random.choice(non_terminal_nodes) if non_terminal_nodes else random.choice(terminal_nodes)
            else:
                node_id = random.choice(all_nodes)
            
            target_payoff = node_targets[node_id]
            
            is_terminal = len(tree[node_id]['children']) == 0
            if is_terminal:
                n_binaries = 1
                prices = [np.exp(-r * dt)]
            else:
                n_binaries = N
                prices = [np.exp(-r * dt) for _ in range(N)]
            
            state = construct_state(node_id, target_payoff, tree)
            
            noise_decay = max(0.1, 1.0 - episode / episodes_this_iter)
            add_noise = episode < episodes_this_iter * 0.8
            action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
            
            # FIX 3: Force non-negative hedges for terminals
            if is_terminal:
                action_used = np.maximum(action[:1], 0)
                prices_used = prices[:1]
            else:
                action_used = action[:n_binaries]
                prices_used = prices[:n_binaries]
            
            reward, cost, violation = compute_reward(action_used, target_payoff, prices_used, node_id, tree)
            
            agent.replay_buffer.push(state, action, reward, state, True)
            
            if len(agent.replay_buffer) >= BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            reward_history.append(reward)
            
            if (episode + 1) % 5000 == 0:
                avg_reward = np.mean(reward_history[-1000:])
                print(f"  Episode {episode+1}/{episodes_this_iter}: Avg Reward={avg_reward:.2f}")
        
        # Phase 2: Solve tree to get updated continuation values
        print(f"\nSolving tree to update continuation values...")
        temp_results = solve_tree_with_agent(agent, tree, probabilities)
        
        # Phase 3: Update node targets based on solution
        for node_id in tree.keys():
            if len(tree[node_id]['children']) > 0:
                children = tree[node_id]['children']
                child_values = [temp_results[c]['cost'] + np.sum(temp_results[c]['hedge']) for c in children]
                node_targets[node_id] = np.sum([probabilities[i] * child_values[i] for i in range(N)])
        
        # Report iteration results
        total_shortfall = sum(r['violation'] for r in temp_results.values())
        initial_cost = temp_results[0]['cost']
        
        print(f"\nIteration {iteration + 1} Results:")
        print(f"  Total Shortfall: {total_shortfall:.6f}")
        print(f"  Initial Cost: ${initial_cost:.4f}")
        
        if total_shortfall < best_total_shortfall:
            best_total_shortfall = total_shortfall
            print(f"  ✓ New best shortfall!")
        
        if total_shortfall < 1.0:
            print(f"\n🎉 SUCCESS! Shortfall < 1.0 at iteration {iteration + 1}")
            break
    
    print("\n" + "="*60)
    print("ITERATIVE TRAINING COMPLETE!")
    print(f"Best Total Shortfall: {best_total_shortfall:.6f}")
    print("="*60)
    
    return agent

# ============================================================
# MAIN EXECUTION
# ============================================================

# Train universal agent with iterative refinement
universal_agent = train_universal_agent_iterative(tree, probabilities)

# Final solve with trained agent
print("\n" + "="*60)
print("FINAL TREE SOLUTION")
print("="*60)
results = solve_tree_with_agent(universal_agent, tree, probabilities)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

total_violation = sum(r['violation'] for r in results.values())

for t in range(T_steps + 1):
    print(f"\nTime t={t}:")
    nodes_at_time = [n for n, data in tree.items() if data['time'] == t]
    
    for node in nodes_at_time[:10]:
        r = results[node]
        print(f"  Node {node} (S=${tree[node]['S']:.2f}):")
        print(f"    Hedge: {r['hedge']}")
        print(f"    Cost: ${r['cost']:.4f}, Shortfall: {r['violation']:.6f}")
    
    if len(nodes_at_time) > 10:
        print(f"  ... and {len(nodes_at_time) - 10} more nodes at this level")

print("\n" + "="*60)
print(f"Initial Cost (t=0): ${results[0]['cost']:.4f}")
print(f"Total Shortfall: {total_violation:.6f}")
if SUPER_REPLICATION:
    print(f"Super-Replication: {'SUCCESS' if total_violation < 1.0 else 'NEEDS TUNING'}")
print("="*60)