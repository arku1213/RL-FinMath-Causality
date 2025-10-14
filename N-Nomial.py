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

# Super-replication mode
SUPER_REPLICATION = True
SHORTFALL_MULTIPLIER = 500 + T_steps * 250  # Increased scaling: was 200, now 250

# DDPG Hyperparameters
EPISODES = 15000 + (T_steps - 2) * 5000  # Scales with T
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0005
CRITIC_LR = 0.001
BUFFER_SIZE = 100000
HIDDEN_DIM = 256
WARMUP_EPISODES = 500

# Reward shaping
def get_penalty_weight(node_id, tree):
    """Adaptive penalty based on node type and payoff size - strengthened for deep trees"""
    if len(tree[node_id]['children']) > 0:
        return 150  # Intermediate nodes (was 100, now 150)
    else:
        payoff = tree[node_id]['payoff']
        if payoff > 150:
            return 250  # Very high payoffs (was 200, now 250)
        elif payoff > 50:
            return 150  # High payoffs (was 100, now 150)
        elif payoff > 15:
            return 100  # Medium payoffs (was 75, now 100)
        else:
            return 10   # Low/zero payoffs

COST_WEIGHT = 0.01

# ============================================================
# N-NOMIAL TREE PARAMETERS
# ============================================================
def calculate_n_nomial_parameters(N, S0, r, dt):
    """
    Calculate stock price multipliers and risk-neutral probabilities
    for N-nomial tree using symmetric spacing
    
    For N states, we create symmetric moves around S0:
    - Highest state: U^k where k = (N-1)/2
    - Middle states: Between U and D
    - Lowest state: D^k where k = (N-1)/2
    """
    # Base up and down factors (similar to binomial)
    sigma = 0.2  # Volatility (can be parameter later)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Create N states symmetrically spaced
    # For N=4: [u^1.5, u^0.5, d^0.5, d^1.5] (example spacing)
    multipliers = []
    if N == 2:
        multipliers = [u, d]
    elif N == 3:
        multipliers = [u, 1.0, d]
    else:
        # General case: exponentially space between u and d
        for i in range(N):
            # Map i from [0, N-1] to exponent from positive to negative
            exponent = ((N - 1 - 2*i) / (N - 1))
            if exponent > 0:
                multipliers.append(u ** exponent)
            elif exponent < 0:
                multipliers.append(d ** abs(exponent))
            else:
                multipliers.append(1.0)
    
    # Calculate risk-neutral probabilities
    # We need: p_0*m_0 + p_1*m_1 + ... + p_{N-1}*m_{N-1} = exp(r*dt)
    # And: p_0 + p_1 + ... + p_{N-1} = 1
    # And: All p_i >= MIN_PROB (ensure all states are reachable!)
    
    # Use linear programming to find probabilities
    growth = np.exp(r * dt)
    
    # Objective: minimize variance (or just find feasible solution)
    c = np.zeros(N)  # Don't optimize anything, just find feasible
    
    # Equality constraints:
    # 1) Sum of probabilities = 1
    # 2) Expected growth = exp(r*dt)
    A_eq = np.array([
        np.ones(N),  # p_0 + p_1 + ... = 1
        multipliers  # p_0*m_0 + p_1*m_1 + ... = growth
    ])
    b_eq = np.array([1.0, growth])
    
    # Bounds: MIN_PROB <= p_i <= 1 - MIN_PROB*(N-1)
    # This ensures all states have positive probability!
    MIN_PROB = 0.03  # Reduced from 0.05 to 0.03 for N=5 flexibility
    max_prob = 1.0 - MIN_PROB * (N - 1)  # Leave room for other states
    bounds = [(MIN_PROB, max_prob) for _ in range(N)]
    
    # Solve
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        print(f"WARNING: Could not find risk-neutral probabilities with MIN_PROB={MIN_PROB}")
        print(f"Trying with smaller minimum probability...")
        # Retry with smaller minimum
        MIN_PROB = 0.01
        max_prob = 1.0 - MIN_PROB * (N - 1)
        bounds = [(MIN_PROB, max_prob) for _ in range(N)]
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            print(f"WARNING: Still could not find probabilities. Using uniform fallback.")
            probabilities = np.ones(N) / N
        else:
            probabilities = result.x
    else:
        probabilities = result.x
    
    # Validate
    prob_sum = np.sum(probabilities)
    expected_growth = np.sum(probabilities * multipliers)
    min_prob = np.min(probabilities)
    
    print(f"\nN={N}-nomial Parameters:")
    print(f"  Multipliers: {[f'{m:.4f}' for m in multipliers]}")
    print(f"  Probabilities: {[f'{p:.4f}' for p in probabilities]}")
    print(f"  Min probability: {min_prob:.4f} (ensures all states reachable)")
    print(f"  Sum of probabilities: {prob_sum:.6f}")
    print(f"  Expected growth: {expected_growth:.6f} (target: {growth:.6f})")
    print(f"  Growth error: {abs(expected_growth - growth):.6e}")
    
    return multipliers, probabilities

multipliers, probabilities = calculate_n_nomial_parameters(N, S0, r, dt)

# Dynamic Action Space
max_stock_price = S0 * max(multipliers) ** T_steps
max_terminal_payoff = max(max_stock_price - K, 0)
CONTINUATION_MULTIPLIER = 3.0 + (T_steps * 0.7) + (N - 2) * 0.3  # Scales with T and N
ACTION_SCALE = max(150.0, max_terminal_payoff * CONTINUATION_MULTIPLIER)

print("="*60)
print(f"N={N}-NOMIAL DDPG SUPER-REPLICATION FOR T={T_steps}")
print("="*60)
print(f"Super-Replication: {SUPER_REPLICATION}")
print(f"Shortfall Multiplier: {SHORTFALL_MULTIPLIER}× (scales with T)")
print(f"Max Terminal Payoff: ${max_terminal_payoff:.2f}")
print(f"Continuation Multiplier: {CONTINUATION_MULTIPLIER:.1f}× (scales with T & N)")
print(f"Action Scale (Dynamic): ±{ACTION_SCALE:.1f}")
print(f"Base Episodes: {EPISODES}")
print(f"States per Node: {N}")
print("="*60)

# ============================================================
# N-NOMIAL TREE BUILDER
# ============================================================
def get_node_id_from_state(state_vector, t):
    """
    Map state vector to unique node_id
    state_vector: tuple of (n_0, n_1, ..., n_{N-1}) where n_i = count of state i
    Sum of state_vector = t
    """
    # Count nodes before this level
    nodes_before = 0
    for time in range(t):
        # Number of states at time is C(time + N - 1, N - 1)
        from math import comb
        nodes_before += comb(time + N - 1, N - 1)
    
    # Position within this level (lexicographic ordering)
    # This is complex for general N, so we use a simpler mapping
    # We'll use a hash-based approach with ordered enumeration
    
    # For now, enumerate all possible states at time t
    position = 0
    for candidate in generate_all_states_at_time(t, N):
        if candidate == state_vector:
            break
        position += 1
    
    return nodes_before + position

def generate_all_states_at_time(t, N):
    """
    Generate all possible state vectors at time t for N states
    Each state vector (n_0, n_1, ..., n_{N-1}) where sum = t
    """
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
    """
    Build N-nomial tree with recombining
    State: (n_0, n_1, ..., n_{N-1}) where n_i = number of times state i occurred
    """
    tree = {}
    node_id = 0
    
    for t in range(T_steps + 1):
        states_at_t = generate_all_states_at_time(t, N)
        
        for state_vector in states_at_t:
            # Calculate stock price
            S = S0
            for i, count in enumerate(state_vector):
                S *= (multipliers[i] ** count)
            
            tree[node_id] = {
                'S': S,
                'time': t,
                'state': state_vector
            }
            
            if t < T_steps:
                # Generate N children (one for each possible next state)
                children = []
                for next_state_idx in range(N):
                    # Child state: increment count for next_state_idx
                    child_state = list(state_vector)
                    child_state[next_state_idx] += 1
                    child_state = tuple(child_state)
                    
                    # Find child node_id
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
print(f"Terminal nodes: {len(terminal_nodes)}")
print(f"Max Payoff: ${max_payoff:.2f}")
print("\nSample terminal payoffs:")
for node_id in terminal_nodes[:min(5, len(terminal_nodes))]:
    state = tree[node_id]['state']
    print(f"  Node {node_id} {state}: S=${tree[node_id]['S']:.2f}, Payoff=${tree[node_id]['payoff']:.2f}")

# ============================================================
# NEURAL NETWORKS (Same architecture as before)
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)
        nn.init.uniform_(self.fc4.bias, -0.003, 0.003)
    
    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        x = torch.tanh(self.fc4(x)) * ACTION_SCALE
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
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
# DDPG AGENT
# ============================================================
class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
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
# SUPER-REPLICATION REWARD
# ============================================================
def compute_reward(hedge, target_payoff, binary_prices, node_id, tree):
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    realized = np.sum(hedge)
    
    shortfall = max(0, target_payoff - realized)
    excess = max(0, realized - target_payoff)
    
    penalty_weight = get_penalty_weight(node_id, tree)
    
    terminal_nodes = [n for n, data in tree.items() if len(data['children']) == 0]
    max_payoff_global = max(tree[n]['payoff'] for n in terminal_nodes)
    
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
    
    violation = shortfall
    return reward, cost, violation

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_node(node_id, n_binaries, target_payoff, binary_prices, tree, base_episodes):
    state_dim = 1
    action_dim = n_binaries
    
    is_terminal = len(tree[node_id]['children']) == 0
    
    if is_terminal:
        if target_payoff > 50:
            episodes = int(base_episodes * 2)
            print(f"  → High-value terminal: using {episodes} episodes")
        else:
            episodes = base_episodes
    else:
        episodes = int(base_episodes * 1.5)
        print(f"  → Intermediate node: using {episodes} episodes")
    
    agent = DDPGAgent(state_dim, action_dim, HIDDEN_DIM)
    
    best_reward = -float('inf')
    best_hedge = None
    best_cost = None
    best_violation = None
    
    reward_history = []
    
    S = tree[node_id]['S']
    state = np.array([S / S0])
    
    print(f"\nTraining Node {node_id} (S=${S:.2f}, Target=${target_payoff:.2f})")
    print(f"Penalty Weight: {get_penalty_weight(node_id, tree)}")
    
    curriculum_steps = WARMUP_EPISODES
    
    for episode in range(episodes):
        if is_terminal:
            curriculum_target = target_payoff
        else:
            if episode < curriculum_steps:
                curriculum_target = target_payoff * (episode / curriculum_steps)
            else:
                curriculum_target = target_payoff
        
        noise_decay = max(0.1, 1.0 - episode / episodes)
        add_noise = episode < episodes * 0.8
        action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
        
        reward, cost, violation = compute_reward(action, curriculum_target, binary_prices, node_id, tree)
        
        done = True
        agent.replay_buffer.push(state, action, reward, state, done)
        
        if episode >= BATCH_SIZE:
            actor_loss, critic_loss = agent.update(BATCH_SIZE)
        
        reward_history.append(reward)
        
        actual_reward, actual_cost, actual_violation = compute_reward(action, target_payoff, binary_prices, node_id, tree)
        if actual_reward > best_reward:
            best_reward = actual_reward
            best_hedge = action.copy()
            best_cost = actual_cost
            best_violation = actual_violation
        
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(reward_history[-500:])
            print(f"Ep {episode+1}/{episodes}: Avg={avg_reward:.2f}, Best Shortfall={best_violation:.6f}")
    
    return best_hedge, best_cost, best_violation

# ============================================================
# BACKWARD INDUCTION
# ============================================================
print("\n" + "="*60)
print("STARTING BACKWARD INDUCTION")
print("="*60)

results = {}

for t in range(T_steps, -1, -1):
    print("\n" + "*"*60)
    print(f"TIME STEP t={t}")
    print("*"*60)
    
    nodes_at_time = [n for n, data in tree.items() if data['time'] == t]
    
    for node_id in nodes_at_time:
        node_data = tree[node_id]
        children = node_data['children']
        
        if len(children) == 0:
            target = node_data['payoff']
            n_binaries = 1
            prices = [np.exp(-r * dt)]
        else:
            # N children for N-nomial
            child_values = [results[c]['cost'] + np.sum(results[c]['hedge']) for c in children]
            
            # Use N-nomial probabilities
            target = np.sum([probabilities[i] * child_values[i] for i in range(N)])
            
            n_binaries = N
            prices = [np.exp(-r * dt) for _ in range(N)]
        
        hedge, cost, viol = train_node(node_id, n_binaries, target, prices, tree, EPISODES)
        results[node_id] = {'hedge': hedge, 'cost': cost, 'violation': viol}

print("\n" + "="*60)
print("BACKWARD INDUCTION COMPLETE!")
print("="*60)

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
    
    for node in nodes_at_time[:10]:  # Show first 10 nodes per level
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