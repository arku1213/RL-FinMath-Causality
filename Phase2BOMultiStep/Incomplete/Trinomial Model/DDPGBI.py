import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ============================================================
# CONFIGURATION - CHANGE T_steps HERE!
# ============================================================
S0 = 100.0
u = 1.2214
m = 1.0      # Middle state (stock stays same)
d = 0.8187
r = 0.05
K = 100.0
T_steps = 5  # ← CHANGE THIS: 2, 3, 4, 5, etc.
dt = 1.0

# Super-replication mode
SUPER_REPLICATION = True
# REVERSED adaptive multiplier: INCREASES for deeper trees (need stronger enforcement)
SHORTFALL_MULTIPLIER = 500 + T_steps * 200  # T=2:900, T=3:1100, T=4:1300, T=5:1500

# DDPG Hyperparameters
EPISODES = 15000  # Increased from 10000 for better convergence on deep trees
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0005
CRITIC_LR = 0.001
BUFFER_SIZE = 100000
HIDDEN_DIM = 256
WARMUP_EPISODES = 500

# Reward shaping (node-dependent)
def get_penalty_weight(node_id, tree):
    """Adaptive penalty based on node type and payoff size"""
    if len(tree[node_id]['children']) > 0:
        return 100  # Intermediate nodes
    else:
        # Terminal nodes: stronger penalty for larger payoffs
        payoff = tree[node_id]['payoff']
        if payoff > 50:
            return 100  # Very high payoffs
        elif payoff > 15:  # Lowered from 20 to catch medium payoffs like $22
            return 75   # Medium-high payoffs (was 50)
        else:
            return 10   # Low/zero payoffs

COST_WEIGHT = 0.01

# Dynamic Action Space - Trinomial grows faster than binomial
max_stock_price = S0 * (u ** T_steps)
max_terminal_payoff = max(max_stock_price - K, 0)

# Trinomial continuation multiplier scales more aggressively
CONTINUATION_MULTIPLIER = 3.0 + (T_steps * 0.7)
ACTION_SCALE = max(150.0, max_terminal_payoff * CONTINUATION_MULTIPLIER)

print("="*60)
print(f"TRINOMIAL DDPG SUPER-REPLICATION FOR T={T_steps}")
print("="*60)
print(f"Super-Replication: {SUPER_REPLICATION}")
print(f"Shortfall Multiplier: {SHORTFALL_MULTIPLIER}× (ADAPTIVE)")
print(f"  Formula: max(500, 2000 - T×300)")
print(f"Max Terminal Payoff: ${max_terminal_payoff:.2f}")
print(f"Continuation Multiplier: {CONTINUATION_MULTIPLIER:.1f}×")
print(f"Action Scale (Dynamic): ±{ACTION_SCALE:.1f}")
print(f"Base Episodes: {EPISODES}")
print(f"Penalty: 100 (high >$50), 75 (med >$15), 10 (low)")
print(f"States per Node: 3 (Up, Middle, Down)")
print("="*60)

# ============================================================
# TRINOMIAL PROBABILITIES
# ============================================================
def calculate_trinomial_probabilities(S0, u, m, d, r, dt):
    """
    Calculate risk-neutral probabilities for trinomial model
    With m=1 (middle state), using arbitrage-free conditions
    """
    growth = np.exp(r * dt)
    
    # For trinomial with m=1, we have 2 constraints:
    # 1) p_u * u + p_m * 1 + p_d * d = growth (expected return)
    # 2) p_u + p_m + p_d = 1 (probabilities sum to 1)
    
    # We need a third equation. Standard approach: minimize variance
    # or use symmetric distribution around risk-neutral measure
    
    # Using the standard trinomial formula (Cox-Ross-Rubinstein style):
    # For m=1, we can derive:
    
    # From constraint (1): p_u * u + p_m + p_d * d = growth
    # From constraint (2): p_u + p_m + p_d = 1
    
    # Solving these two equations with additional constraint:
    # Let's use: p_u * u^2 + p_m * 1 + p_d * d^2 = growth^2 + variance
    
    # Simpler approach: symmetric probabilities around middle
    # p_u = (growth - d) / (u - d) * alpha
    # p_d = (u - growth) / (u - d) * alpha
    # where alpha is chosen so probabilities are positive
    
    # Most straightforward for m=1:
    p_d = (growth - u) * (growth - m) / ((d - u) * (d - m))
    p_u = (growth - m) * (growth - d) / ((u - m) * (u - d))
    p_m = 1.0 - p_u - p_d
    
    # Validation
    expected_growth = p_u * u + p_m * m + p_d * d
    prob_sum = p_u + p_m + p_d
    
    print(f"\nProbability Calculation:")
    print(f"  p_u = {p_u:.6f}")
    print(f"  p_m = {p_m:.6f}")
    print(f"  p_d = {p_d:.6f}")
    print(f"  Sum = {prob_sum:.6f}")
    print(f"  Expected growth = {expected_growth:.6f}")
    print(f"  Target growth = {growth:.6f}")
    
    assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities sum to {prob_sum}, not 1"
    assert abs(expected_growth - growth) < 1e-4, f"Expected growth {expected_growth} != {growth}"
    
    # Check if probabilities are valid
    if p_u < 0 or p_m < 0 or p_d < 0:
        print(f"\nWARNING: Negative probabilities detected!")
        print(f"This trinomial parameterization may not be arbitrage-free.")
        print(f"Adjusting to ensure non-negative probabilities...")
        
        # Fallback: use simpler symmetric approach
        p_u = (growth - d) / (u - d) * 0.4
        p_d = (u - growth) / (u - d) * 0.4
        p_m = 1.0 - p_u - p_d
        
        print(f"Adjusted probabilities:")
        print(f"  p_u = {p_u:.6f}")
        print(f"  p_m = {p_m:.6f}")
        print(f"  p_d = {p_d:.6f}")
    
    return p_u, p_m, p_d

p_u, p_m, p_d = calculate_trinomial_probabilities(S0, u, m, d, r, dt)
print(f"\nRisk-Neutral Probabilities:")
print(f"  P(Up) = {p_u:.4f}")
print(f"  P(Middle) = {p_m:.4f}")
print(f"  P(Down) = {p_d:.4f}")
print(f"  Sum = {p_u + p_m + p_d:.4f}")

# ============================================================
# TRINOMIAL TREE WITH RECOMBINING
# ============================================================
def get_node_id_at_level(t, n_up, n_mid):
    """
    Calculate node_id for state (n_up, n_mid, n_down) at time t
    where n_down = t - n_up - n_mid
    """
    # Nodes before this level
    nodes_before = sum((i+1)*(i+2)//2 for i in range(t))
    
    # Position within this level (lexicographic: enumerate by n_up, then n_mid)
    position = 0
    for up in range(n_up):
        # For each smaller n_up, count all valid n_mid values
        position += (t - up) + 1
    # Add n_mid for current n_up
    position += n_mid
    
    return nodes_before + position

def build_trinomial_tree(T_steps, S0, u, m, d, K):
    """
    Build recombining trinomial tree
    State: (n_up, n_mid, n_down) where n_up + n_mid + n_down = t
    """
    tree = {}
    
    for t in range(T_steps + 1):
        for n_up in range(t + 1):
            for n_mid in range(t - n_up + 1):
                n_down = t - n_up - n_mid
                
                node_id = get_node_id_at_level(t, n_up, n_mid)
                
                # Stock price
                S = S0 * (u ** n_up) * (m ** n_mid) * (d ** n_down)
                
                tree[node_id] = {
                    'S': S,
                    'time': t,
                    'state': (n_up, n_mid, n_down)
                }
                
                if t < T_steps:
                    # 3 children: up, middle, down
                    child_up_id = get_node_id_at_level(t+1, n_up+1, n_mid)
                    child_mid_id = get_node_id_at_level(t+1, n_up, n_mid+1)
                    child_down_id = get_node_id_at_level(t+1, n_up, n_mid)
                    tree[node_id]['children'] = [child_up_id, child_mid_id, child_down_id]
                else:
                    tree[node_id]['children'] = []
                    tree[node_id]['payoff'] = max(S - K, 0)
    
    return tree

tree = build_trinomial_tree(T_steps, S0, u, m, d, K)

# Calculate max payoff for normalization
terminal_nodes = [n for n, data in tree.items() if len(data['children']) == 0]
max_payoff = max(tree[n]['payoff'] for n in terminal_nodes)

print(f"\nTrinomial Tree with T={T_steps}")
print(f"Total nodes: {len(tree)}")
print(f"Terminal nodes: {len(terminal_nodes)}")
print(f"Max Payoff: ${max_payoff:.2f}")
print("\nSample terminal payoffs:")
for node_id in terminal_nodes[:5]:
    state = tree[node_id]['state']
    print(f"  Node {node_id} {state}: S=${tree[node_id]['S']:.2f}, Payoff=${tree[node_id]['payoff']:.2f}")

# ============================================================
# NEURAL NETWORKS (Same as Binomial)
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
# SUPER-REPLICATION REWARD FUNCTION
# ============================================================
def compute_reward(hedge, target_payoff, binary_prices, node_id, tree):
    """
    Super-replication reward: heavily penalize shortfall
    """
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    realized = np.sum(hedge)
    
    shortfall = max(0, target_payoff - realized)
    excess = max(0, realized - target_payoff)
    
    penalty_weight = get_penalty_weight(node_id, tree)
    
    # Normalize by max_payoff
    normalized_shortfall = shortfall / max_payoff if max_payoff > 0 else shortfall
    normalized_excess = excess / max_payoff if max_payoff > 0 else excess
    
    if SUPER_REPLICATION:
        # Super-replication: 1000× penalty for shortfall, tiny penalty for excess
        reward = -(COST_WEIGHT * abs(cost) 
                   + penalty_weight * SHORTFALL_MULTIPLIER * normalized_shortfall**2
                   + penalty_weight * 0.01 * normalized_excess**2)
    else:
        # Standard replication
        violation = max(0, target_payoff - realized)
        normalized_violation = violation / max_payoff if max_payoff > 0 else violation
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
    print(f"Binary Prices: {binary_prices}")
    print(f"Penalty Weight: {get_penalty_weight(node_id, tree)}")
    if SUPER_REPLICATION:
        print(f"Mode: Super-Replication (shortfall penalty {SHORTFALL_MULTIPLIER}×)")
    
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
            print(f"Episode {episode+1}/{episodes}")
            print(f"  Avg Reward (last 500): {avg_reward:.2f}")
            print(f"  Best Reward: {best_reward:.2f}")
            print(f"  Best Hedge: {best_hedge}")
            print(f"  Best Cost: ${best_cost:.4f}, Shortfall: {best_violation:.6f}")
            print(f"  Noise Decay: {noise_decay:.3f}")
    
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
            # 3 children for trinomial
            child_values = [results[c]['cost'] + np.sum(results[c]['hedge']) for c in children]
            target = p_u * child_values[0] + p_m * child_values[1] + p_d * child_values[2]
            n_binaries = 3
            prices = [np.exp(-r * dt), np.exp(-r * dt), np.exp(-r * dt)]
        
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
total_shortfall = total_violation  # In super-replication mode, violation = shortfall

for t in range(T_steps + 1):
    print(f"\nTime t={t}:")
    nodes_at_time = [n for n, data in tree.items() if data['time'] == t]
    
    for node in nodes_at_time:
        r = results[node]
        print(f"  Node {node} (S=${tree[node]['S']:.2f}):")
        print(f"    Hedge: {r['hedge']}")
        print(f"    Cost: ${r['cost']:.4f}")
        print(f"    Shortfall: {r['violation']:.6f}")

print("\n" + "="*60)
print(f"Initial Cost (t=0): ${results[0]['cost']:.4f}")
print(f"Total Shortfall: {total_shortfall:.6f}")
if SUPER_REPLICATION:
    print(f"Super-Replication: {'SUCCESS' if total_shortfall < 0.1 else 'NEEDS TUNING'}")
print("="*60)