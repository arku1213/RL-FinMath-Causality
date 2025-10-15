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
d = 0.8187
r = 0.05
K = 100.0
T_steps = 2  # ← CHANGE THIS: 2, 3, 4, 5, etc.
dt = 1.0

# DDPG Hyperparameters
EPISODES = 10000  # Increased for T=3 complexity
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0005  # Reduced from 0.001
CRITIC_LR = 0.001
BUFFER_SIZE = 100000
HIDDEN_DIM = 256  # Increased from 128
WARMUP_EPISODES = 500  # More exploration before learning

# Reward shaping (node-dependent)
def get_penalty_weight(node_id, tree):
    """Adaptive penalty based on node type and payoff size"""
    if len(tree[node_id]['children']) > 0:
        return 100  # Intermediate nodes (increased from 50)
    else:
        # Terminal nodes: stronger penalty for larger payoffs
        payoff = tree[node_id]['payoff']
        if payoff > 50:
            return 100  # Very strong penalty for high-value nodes
        elif payoff > 20:
            return 50   # Strong penalty for medium-value nodes
        else:
            return 10   # Normal penalty for zero/low-value nodes

COST_WEIGHT = 0.01  # Small weight on cost to prioritize constraint satisfaction

# Dynamic Action Space Bounds - AUTO-ADJUSTS BASED ON T!
# Calculate max possible payoff and continuation value
max_stock_price = S0 * (u ** T_steps)
max_terminal_payoff = max(max_stock_price - K, 0)

# Continuation multiplier scales with tree depth
# Deeper trees have larger continuation values relative to terminal payoffs
# T=2: 3.0×, T=3: 3.5×, T=4: 4.0×, T=5: 4.5×, etc.
CONTINUATION_MULTIPLIER = 2.5 + (T_steps * 0.5)  # Scales linearly with T

ACTION_SCALE = max(150.0, max_terminal_payoff * CONTINUATION_MULTIPLIER)

print("="*60)
print(f"IMPROVED PyTorch DDPG FOR T={T_steps} BINOMIAL")
print("="*60)
print(f"Max Terminal Payoff: ${max_terminal_payoff:.2f}")
print(f"Continuation Multiplier: {CONTINUATION_MULTIPLIER:.1f}× (scales with T)")
print(f"Action Scale (Dynamic): ±{ACTION_SCALE:.1f}")
print(f"Penalty Weight: Adaptive (10-100 based on node type/payoff)")
print(f"Cost Weight: {COST_WEIGHT}")
print(f"Hidden Dim: {HIDDEN_DIM}")
print(f"Base Episodes: {EPISODES}")
print(f"Curriculum: Disabled for terminals, enabled for intermediates")
print("="*60)

# ============================================================
# BINOMIAL TREE - AUTO-GENERATES BASED ON T_steps
# ============================================================
q = (np.exp(r * dt) - d) / (u - d)

def build_binomial_tree(T_steps, S0, u, d, K):
    """
    Automatically build binomial tree for any T_steps
    Node numbering: 
      - Level 0: node 0
      - Level 1: nodes 1, 2
      - Level 2: nodes 3, 4, 5
      - Level 3: nodes 6, 7, 8, 9
      etc.
    """
    tree = {}
    node_id = 0
    
    # Build level by level
    for t in range(T_steps + 1):
        n_nodes_at_level = t + 1
        
        for i in range(n_nodes_at_level):
            # Calculate stock price: S0 * u^(i) * d^(t-i)
            n_ups = t - i
            n_downs = i
            S = S0 * (u ** n_ups) * (d ** n_downs)
            
            tree[node_id] = {'S': S, 'time': t}
            
            # Add children if not terminal
            if t < T_steps:
                # Children are in next level
                # Up child: same i, down child: i+1
                child_start = node_id + n_nodes_at_level
                tree[node_id]['children'] = [child_start, child_start + 1]
            else:
                # Terminal node
                tree[node_id]['children'] = []
                tree[node_id]['payoff'] = max(S - K, 0)
            
            node_id += 1
    
    return tree

tree = build_binomial_tree(T_steps, S0, u, d, K)

# Calculate max payoff for normalization
terminal_nodes = [n for n, data in tree.items() if len(data['children']) == 0]
max_payoff = max(tree[n]['payoff'] for n in terminal_nodes)

print(f"\nBinomial Tree with T={T_steps}")
print(f"Total nodes: {len(tree)}")
print(f"Terminal nodes: {terminal_nodes}")
print(f"Max Payoff: ${max_payoff:.2f}")
for node_id in terminal_nodes:
    print(f"  Node {node_id} (S=${tree[node_id]['S']:.2f}): Payoff=${tree[node_id]['payoff']:.2f}")

# ============================================================
# NEURAL NETWORKS
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Extra layer
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        # Better initialization for large outputs
        nn.init.uniform_(self.fc4.weight, -0.003, 0.003)
        nn.init.uniform_(self.fc4.bias, -0.003, 0.003)
    
    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        x = torch.tanh(self.fc4(x)) * ACTION_SCALE  # CRITICAL: Scale to ±60
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
# ORNSTEIN-UHLENBECK NOISE (with decay)
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
        return self.state * decay_factor  # Decay noise over time

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
        
        # Critic update
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
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()

# ============================================================
# ENVIRONMENT FOR ONE NODE
# ============================================================
def compute_reward(hedge, target_payoff, binary_prices, node_id, tree, normalized=True):
    """
    Improved reward with normalization and node-dependent penalty
    """
    hedge = np.atleast_1d(hedge)
    cost = np.sum(hedge * binary_prices)
    realized = np.sum(hedge)
    violation = max(0, target_payoff - realized)
    
    # Get node-specific penalty weight
    penalty_weight = get_penalty_weight(node_id, tree)
    
    # Get max_payoff from tree
    terminal_nodes = [n for n, data in tree.items() if len(data['children']) == 0]
    max_payoff = max(tree[n]['payoff'] for n in terminal_nodes)
    
    # Normalize violation by max_payoff to keep scale reasonable
    if normalized:
        normalized_violation = violation / max_payoff if max_payoff > 0 else violation
        reward = -(COST_WEIGHT * abs(cost) + penalty_weight * normalized_violation**2)
    else:
        reward = -(COST_WEIGHT * abs(cost) + penalty_weight * violation**2)
    
    return reward, cost, violation

def train_node(node_id, n_binaries, target_payoff, binary_prices, tree, base_episodes):
    """
    Train DDPG agent for a single node with curriculum learning
    """
    state_dim = 1  # Just the stock price (could add time, but keeping simple)
    action_dim = n_binaries
    
    is_terminal = len(tree[node_id]['children']) == 0
    
    # Determine episode count based on node type and payoff
    if is_terminal:
        if target_payoff > 50:
            episodes = int(base_episodes * 2)  # 20,000 for high-value terminals
            print(f"  → High-value terminal node: using {episodes} episodes")
        else:
            episodes = base_episodes
    else:
        episodes = int(base_episodes * 1.5)  # 15,000 for intermediates
        print(f"  → Intermediate node: using {episodes} episodes")
    
    agent = DDPGAgent(state_dim, action_dim, HIDDEN_DIM)
    
    best_reward = -float('inf')
    best_hedge = None
    best_cost = None
    best_violation = None
    
    reward_history = []
    
    S = tree[node_id]['S']
    state = np.array([S / S0])  # Normalize state
    
    print(f"\nTraining Node {node_id} (S=${S:.2f}, Target=${target_payoff:.2f})")
    print(f"Binary Prices: {binary_prices}")
    print(f"Penalty Weight: {get_penalty_weight(node_id, tree)}")
    
    # Curriculum: Only for intermediate nodes, disabled for terminals
    curriculum_steps = WARMUP_EPISODES
    
    for episode in range(episodes):
        # Curriculum learning: ONLY for intermediate nodes
        if is_terminal:
            curriculum_target = target_payoff  # No curriculum for terminals!
        else:
            if episode < curriculum_steps:
                curriculum_target = target_payoff * (episode / curriculum_steps)
            else:
                curriculum_target = target_payoff
        
        # Noise decay: reduce exploration over time
        noise_decay = max(0.1, 1.0 - episode / episodes)
        
        # Select action
        add_noise = episode < episodes * 0.8  # Stop noise in final 20%
        action = agent.select_action(state, add_noise=add_noise, noise_decay=noise_decay)
        
        # Compute reward (pass tree for adaptive penalty)
        reward, cost, violation = compute_reward(action, curriculum_target, binary_prices, node_id, tree)
        
        # Store transition
        done = True  # Single-step episode
        agent.replay_buffer.push(state, action, reward, state, done)
        
        # Update networks
        if episode >= BATCH_SIZE:
            actor_loss, critic_loss = agent.update(BATCH_SIZE)
        
        reward_history.append(reward)
        
        # Track best solution (using ACTUAL target, not curriculum)
        actual_reward, actual_cost, actual_violation = compute_reward(action, target_payoff, binary_prices, node_id, tree)
        if actual_reward > best_reward:
            best_reward = actual_reward
            best_hedge = action.copy()
            best_cost = actual_cost
            best_violation = actual_violation
        
        # Logging
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(reward_history[-500:])
            print(f"Episode {episode+1}/{episodes}")
            print(f"  Avg Reward (last 500): {avg_reward:.2f}")
            print(f"  Best Reward: {best_reward:.2f}")
            print(f"  Best Hedge: {best_hedge}")
            print(f"  Best Cost: ${best_cost:.4f}, Violation: {best_violation:.6f}")
            print(f"  Noise Decay: {noise_decay:.3f}")
    
    return best_hedge, best_cost, best_violation

# ============================================================
# BACKWARD INDUCTION - AUTO-ADAPTS TO ANY T_steps
# ============================================================
print("\n" + "="*60)
print("STARTING BACKWARD INDUCTION")
print("="*60)

results = {}

# Work backwards from terminal time to t=0
for t in range(T_steps, -1, -1):
    print("\n" + "*"*60)
    print(f"TIME STEP t={t}")
    print("*"*60)
    
    # Find all nodes at this time level
    nodes_at_time = [n for n, data in tree.items() if data['time'] == t]
    
    for node_id in nodes_at_time:
        node_data = tree[node_id]
        children = node_data['children']
        
        if len(children) == 0:
            # Terminal node - single binary that pays at this node
            target = node_data['payoff']
            n_binaries = 1
            prices = [np.exp(-r * dt)]
        else:
            # Intermediate node - compute continuation value
            child_payoffs = [results[c]['cost'] + np.sum(results[c]['hedge']) for c in children]
            target = q * child_payoffs[0] + (1-q) * child_payoffs[1]
            
            # Number of binaries = number of children (2 for binomial)
            n_binaries = len(children)
            
            # Each binary has same discount factor (one time step)
            prices = [np.exp(-r * dt) for _ in range(n_binaries)]
        
        hedge, cost, viol = train_node(node_id, n_binaries, target, prices, tree, EPISODES)
        results[node_id] = {'hedge': hedge, 'cost': cost, 'violation': viol}

print("\n" + "="*60)
print("BACKWARD INDUCTION COMPLETE!")
print("="*60)

# ============================================================
# SUMMARY - AUTO-ADAPTS TO ANY T_steps
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

total_violation = sum(r['violation'] for r in results.values())

for t in range(T_steps + 1):
    print(f"\nTime t={t}:")
    nodes_at_time = [n for n, data in tree.items() if data['time'] == t]
    
    for node in nodes_at_time:
        r = results[node]
        print(f"  Node {node} (S=${tree[node]['S']:.2f}):")
        print(f"    Hedge: {r['hedge']}")
        print(f"    Cost: ${r['cost']:.4f}")
        print(f"    Violation: {r['violation']:.6f}")

print("\n" + "="*60)
print(f"Initial Cost (t=0): ${results[0]['cost']:.4f}")
print(f"Total Violation: {total_violation:.6f}")
print("="*60)