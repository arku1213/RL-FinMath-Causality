import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from scipy.optimize import minimize, LinearConstraint

# Binomial model parameters
S0 = 100
K = 100
r = 0.05
T = 1.0
u = 1.2
d = 0.8
dt = T / 2

# Risk-neutral probability
p = (np.exp(r * dt) - d) / (u - d)

# Stock prices and payoffs
prices = {
    'T2_UU': S0 * u * u,
    'T2_UD': S0 * u * d,
    'T2_DU': S0 * d * u,
    'T2_DD': S0 * d * d
}

payoffs = {
    'UU': max(prices['T2_UU'] - K, 0),
    'UD': max(prices['T2_UD'] - K, 0),
    'DU': max(prices['T2_DU'] - K, 0),
    'DD': max(prices['T2_DD'] - K, 0)
}

# Path probabilities
path_probs = {
    'UU': p * p,
    'UD': p * (1-p),
    'DU': (1-p) * p,
    'DD': (1-p) * (1-p)
}

# Theoretical price
theoretical_price = np.exp(-r * T) * sum(path_probs[path] * payoffs[path] for path in ['UU', 'UD', 'DU', 'DD'])

# Binary prices
binary_prices = {
    'b1': p * np.exp(-r * dt),
    'b2': (1-p) * np.exp(-r * dt),
    'b3': p * np.exp(-r * dt),
    'b4': (1-p) * np.exp(-r * dt),
    'b5': p * np.exp(-r * dt),
    'b6': (1-p) * np.exp(-r * dt),
}

print("="*60)
print("PROJECTION-BASED A2C FOR T=2 BINOMIAL")
print("="*60)
print(f"S0={S0}, K={K}, r={r}, T={T}, u={u}, d={d}")
print(f"Risk-neutral p={p:.4f}")
print(f"\nPayoffs: UU=${payoffs['UU']:.2f}, UD=${payoffs['UD']:.2f}, DU=${payoffs['DU']:.2f}, DD=${payoffs['DD']:.2f}")
print(f"Theoretical Price: ${theoretical_price:.2f}")
print("="*60 + "\n")

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(torch.clamp(self.actor_log_std, -20, 2))
        value = self.critic(shared_features)
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            # Map to [0, 50]
            action = torch.tanh(action_mean) * 25 + 25
            return action, None, value
        
        dist = Normal(action_mean, action_std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        
        # Map to [0, 50]
        action = torch.tanh(action_raw) * 25 + 25
        
        return action, log_prob, value, entropy

# Projection functions
def project_to_feasible(proposed_binaries, stage, existing_binaries=None):
    """
    Project proposed binaries onto feasible region using optimization.
    
    Stage 0 (T=0): Choose b1, b2
    Stage 1 (T=1): Choose b3, b4 (if UP) or b5, b6 (if DOWN), knowing b1, b2
    
    Constraints: For each terminal scenario, portfolio_value >= payoff
    """
    
    if stage == 0:
        # At T=0: Choose b1, b2 to minimize cost while satisfying future constraints
        # We don't know which path will be taken, so we optimize for expected feasibility
        
        def objective(x):
            # Minimize: cost + penalty for infeasibility
            b1, b2 = x
            cost = b1 * binary_prices['b1'] + b2 * binary_prices['b2']
            
            # Penalty for being far from proposed
            deviation = (b1 - proposed_binaries[0])**2 + (b2 - proposed_binaries[1])**2
            
            return cost + 0.1 * deviation
        
        # Bounds: [0, 50]
        bounds = [(0, 50), (0, 50)]
        
        # Initial guess: proposed values
        x0 = proposed_binaries
        
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return result.x
    
    else:
        # At T=1: Choose b3, b4 or b5, b6 knowing b1, b2
        b1 = existing_binaries.get('b1', 0)
        b2 = existing_binaries.get('b2', 0)
        
        if 'UP' in stage:  # We're at UP node, choosing b3, b4
            def objective(x):
                b3, b4 = x
                # Cost for this decision
                cost = b3 * binary_prices['b3'] + b4 * binary_prices['b4']
                # Deviation from proposed
                deviation = (b3 - proposed_binaries[0])**2 + (b4 - proposed_binaries[1])**2
                return cost + 0.1 * deviation
            
            def constraint_uu(x):
                b3, b4 = x
                return b1 + b3 - payoffs['UU']  # >= 0
            
            def constraint_ud(x):
                b3, b4 = x
                return b1 + b4 - payoffs['UD']  # >= 0
            
            constraints = [
                {'type': 'ineq', 'fun': constraint_uu},
                {'type': 'ineq', 'fun': constraint_ud}
            ]
            
        else:  # We're at DOWN node, choosing b5, b6
            def objective(x):
                b5, b6 = x
                cost = b5 * binary_prices['b5'] + b6 * binary_prices['b6']
                deviation = (b5 - proposed_binaries[0])**2 + (b6 - proposed_binaries[1])**2
                return cost + 0.1 * deviation
            
            def constraint_du(x):
                b5, b6 = x
                return b2 + b5 - payoffs['DU']  # >= 0
            
            def constraint_dd(x):
                b5, b6 = x
                return b2 + b6 - payoffs['DD']  # >= 0
            
            constraints = [
                {'type': 'ineq', 'fun': constraint_du},
                {'type': 'ineq', 'fun': constraint_dd}
            ]
        
        bounds = [(0, 50), (0, 50)]
        x0 = proposed_binaries
        
        result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
        
        if not result.success:
            # If optimization fails, use a simple heuristic
            if 'UP' in stage:
                # For UU path, need at least payoffs['UU'] - b1
                b3_min = max(0, payoffs['UU'] - b1)
                b4_min = max(0, payoffs['UD'] - b1)
                return np.array([b3_min, b4_min])
            else:
                b5_min = max(0, payoffs['DU'] - b2)
                b6_min = max(0, payoffs['DD'] - b2)
                return np.array([b5_min, b6_min])
        
        return result.x

# Environment with projection
class ProjectionBinomialEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.t = 0
        self.path = []
        self.binaries = {}
        self.proposed_actions = []
        self.projected_actions = []
        return self._get_state()
    
    def _get_state(self):
        if self.t == 0:
            return np.array([0.0, S0/100, 0.0, 2.0], dtype=np.float32)
        elif self.t == 1:
            path_ind = 1.0 if self.path[-1] == 'U' else -1.0
            price = S0 * u if self.path[-1] == 'U' else S0 * d
            return np.array([1.0, price/100, path_ind, 1.0], dtype=np.float32)
        else:
            return np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def step(self, proposed_action):
        if isinstance(proposed_action, torch.Tensor):
            proposed_action = proposed_action.detach().cpu().numpy()
        proposed_action = np.array(proposed_action).flatten()
        
        self.proposed_actions.append(proposed_action.copy())
        
        if self.t == 0:
            # Project onto feasible region
            projected = project_to_feasible(proposed_action, stage=0)
            self.projected_actions.append(projected)
            
            self.binaries['b1'] = float(projected[0])
            self.binaries['b2'] = float(projected[1])
            
            # Simulate next move
            next_move = 'U' if np.random.rand() < p else 'D'
            self.path.append(next_move)
            self.t = 1
            
            return self._get_state(), 0.0, False, {}
        
        elif self.t == 1:
            # Project with knowledge of b1, b2
            stage_label = 'UP' if self.path[-1] == 'U' else 'DOWN'
            projected = project_to_feasible(proposed_action, stage=stage_label, 
                                           existing_binaries=self.binaries)
            self.projected_actions.append(projected)
            
            if self.path[-1] == 'U':
                self.binaries['b3'] = float(projected[0])
                self.binaries['b4'] = float(projected[1])
            else:
                self.binaries['b5'] = float(projected[0])
                self.binaries['b6'] = float(projected[1])
            
            # Simulate second move
            next_move = 'U' if np.random.rand() < p else 'D'
            self.path.append(next_move)
            self.t = 2
            
            reward = self._calculate_reward()
            return self._get_state(), reward, True, {}
    
    def _calculate_reward(self):
        # Calculate cost
        cost = sum(self.binaries.get(f'b{i}', 0) * binary_prices[f'b{i}'] 
                   for i in range(1, 7))
        
        # Check violations (should be 0 due to projection)
        violations = []
        for scenario in ['UU', 'UD', 'DU', 'DD']:
            pv = self._get_portfolio_value(scenario)
            target = payoffs[scenario]
            if pv < target - 0.01:
                violations.append(target - pv)
        
        # Reward: minimize cost (constraints automatically satisfied)
        cost_penalty = abs(cost - theoretical_price)
        
        # Bonus for being close to theoretical
        bonus = 0
        if 0.95 * theoretical_price <= cost <= 1.05 * theoretical_price:
            bonus = 100
        
        # Penalty for large deviations from proposed actions (encourages learning)
        projection_penalty = 0
        for proposed, projected in zip(self.proposed_actions, self.projected_actions):
            projection_penalty += np.sum((proposed - projected)**2)
        
        reward = -cost_penalty + bonus - 0.01 * projection_penalty
        
        return reward
    
    def _get_portfolio_value(self, scenario):
        value = 0
        if scenario[0] == 'U':
            value += self.binaries.get('b1', 0)
        else:
            value += self.binaries.get('b2', 0)
        
        if scenario == 'UU':
            value += self.binaries.get('b3', 0)
        elif scenario == 'UD':
            value += self.binaries.get('b4', 0)
        elif scenario == 'DU':
            value += self.binaries.get('b5', 0)
        elif scenario == 'DD':
            value += self.binaries.get('b6', 0)
        
        return value
    
    def get_metrics(self):
        cost = sum(self.binaries.get(f'b{i}', 0) * binary_prices[f'b{i}'] 
                   for i in range(1, 7))
        
        violations = []
        for scenario in ['UU', 'UD', 'DU', 'DD']:
            pv = self._get_portfolio_value(scenario)
            target = payoffs[scenario]
            if pv < target - 0.01:
                violations.append(scenario)
        
        return {
            'cost': cost,
            'violations': violations,
            'binaries': dict(self.binaries)
        }

# Training
def train_projection_a2c(num_episodes=10000, gamma=0.99, lr=1e-4):
    env = ProjectionBinomialEnv()
    state_dim = 4
    action_dim = 2
    
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    episode_rewards = []
    episode_costs = []
    episode_violations = []
    
    print("TRAINING PROJECTION-BASED A2C")
    print("="*60 + "\n")
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        entropies = []
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value, entropy = model.get_action(state_tensor)
            
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = next_state
        
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).unsqueeze(1)
        
        # Calculate advantages
        values_tensor = torch.cat(values)
        advantages = returns - values_tensor.detach()
        
        # Losses
        log_probs_tensor = torch.cat(log_probs)
        entropies_tensor = torch.cat(entropies)
        
        actor_loss = -(log_probs_tensor * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -entropies_tensor.mean()
        
        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # Track metrics
        metrics = env.get_metrics()
        episode_rewards.append(sum(rewards))
        episode_costs.append(metrics['cost'])
        episode_violations.append(1 if metrics['violations'] else 0)
        
        if episode % 1000 == 0:
            window = 100
            avg_reward = np.mean(episode_rewards[-window:])
            avg_cost = np.mean(episode_costs[-window:])
            viol_rate = np.mean(episode_violations[-window:]) * 100
            
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Cost: ${avg_cost:.2f} (Target: ${theoretical_price:.2f})")
            print(f"  Violation Rate: {viol_rate:.1f}%")
            print(f"  Sample: {metrics['binaries']}")
            print()
    
    return model

# Train
model = train_projection_a2c()

# Evaluate
print("="*60)
print("FINAL EVALUATION")
print("="*60 + "\n")

env = ProjectionBinomialEnv()
eval_metrics = []

for _ in range(100):
    state = env.reset()
    done = False
    
    with torch.no_grad():
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _, _ = model.get_action(state_tensor, deterministic=True)
            state, reward, done, _ = env.step(action)
    
    eval_metrics.append(env.get_metrics())

avg_cost = np.mean([m['cost'] for m in eval_metrics])
viol_rate = np.mean([1 if m['violations'] else 0 for m in eval_metrics]) * 100
costs = [m['cost'] for m in eval_metrics]

print(f"Results over 100 episodes:")
print(f"  Average Cost: ${avg_cost:.2f}")
print(f"  Theoretical: ${theoretical_price:.2f}")
print(f"  Cost Error: {abs(avg_cost - theoretical_price)/theoretical_price * 100:.2f}%")
print(f"  Cost Std Dev: ${np.std(costs):.2f}")
print(f"  Violation Rate: {viol_rate:.1f}%")
print(f"  Within ±5% of theoretical: {sum(1 for c in costs if 0.95*theoretical_price <= c <= 1.05*theoretical_price)}%")
print(f"\nSample solutions:")
for i in range(3):
    print(f"  {i+1}. {eval_metrics[i]['binaries']}")
    print(f"     Cost: ${eval_metrics[i]['cost']:.2f}, Violations: {eval_metrics[i]['violations']}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)