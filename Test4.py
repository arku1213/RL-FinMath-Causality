import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from math import comb

# ----------------------------
# Environment and CEM
# ----------------------------

class BinomialEnvironment:
    """Environment for hybrid approach: minimal backward + RL forward"""
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T_steps=2, dt=1.0, option_type='call'):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_steps = T_steps
        self.dt = dt
        self.option_type = option_type
        
        # CRR model
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1 / self.u
        # Risk-neutral probability
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Build tree
        self.build_tree()
        self.calculate_terminal_payoffs()
        
        # Target values at T-1 (to be computed in Stage 1)
        self.t_minus_1_targets = None
        self.t_minus_1_solutions = None

    def build_tree(self):
        """Build all nodes"""
        self.nodes = {}
        for t in range(self.T_steps + 1):
            for n_ups in range(t + 1):
                price = self.S0 * (self.u ** n_ups) * (self.d ** (t - n_ups))
                self.nodes[(t, n_ups)] = price

    def calculate_terminal_payoffs(self):
        """Calculate terminal payoffs"""
        self.terminal_payoffs = {}
        for n_ups in range(self.T_steps + 1):
            S_T = self.nodes[(self.T_steps, n_ups)]
            if self.option_type == 'call':
                payoff = max(S_T - self.K, 0)
            else:
                payoff = max(self.K - S_T, 0)
            self.terminal_payoffs[n_ups] = payoff


class CEM1Step:
    """CEM for solving 1-step problems with improved stability"""
    def __init__(self, population_size=400, elite_frac=0.1, n_iterations=200,
                 penalty_mult=500, penalty_max_mult=1000): 
        self.population_size = population_size
        self.n_elite = max(1, int(population_size * elite_frac))
        self.n_iterations = n_iterations
        self.penalty_mult = penalty_mult
        self.penalty_max_mult = penalty_max_mult

    def solve(self, target_up, target_down, price_up, price_down):
        """Solve 1-step binomial to replicate targets"""
        # Initialize sensibly
        mean = np.array([max(target_up, 0.0), max(target_down, 0.0)], dtype=float)
        std = np.array([max(abs(target_up) * 0.5, 1.0), max(abs(target_down) * 0.5, 1.0)], dtype=float)
        
        best_solution = np.array([0.0, 0.0])
        best_cost = float('inf')
        best_error = float('inf')
        
        for iteration in range(self.n_iterations):
            population = np.random.normal(mean, std, (self.population_size, 2))
            
            # Clip to non-negative if both targets are non-negative
            if target_up >= 0 and target_down >= 0:
                population = np.maximum(population, 0.0)
            
            rewards = np.empty(self.population_size)
            
            for idx, sample in enumerate(population):
                b_up, b_down = sample
                cost = b_up * price_up + b_down * price_down
                
                # Replication errors
                error_up = abs(b_up - target_up)
                error_down = abs(b_down - target_down)
                total_error = error_up + error_down
                
                # Penalty
                penalty = self.penalty_mult * (error_up**2 + error_down**2) + \
                          self.penalty_max_mult * max(error_up, error_down)
                
                reward = -cost - penalty
                rewards[idx] = reward
                
                # Track best solution (prefer smaller error, then lower cost)
                if total_error < best_error - 1e-12 or (abs(total_error - best_error) < 1e-12 and cost < best_cost):
                    best_error = total_error
                    best_cost = cost
                    best_solution = sample.copy()
            
            # Select elites and update distribution
            elite_indices = np.argsort(rewards)[-self.n_elite:]
            elite_samples = population[elite_indices]
            
            mean = np.mean(elite_samples, axis=0)
            std = np.std(elite_samples, axis=0) + 1e-6
            std = np.maximum(std, 0.05)
        
        return best_solution, best_cost, best_error

def stage1_solve_t_minus_1(env, cem_params=None):
    """
    STAGE 1: Minimal Backward Induction
    Solve all nodes at T-1 using CEM to get target values.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: Solving T-1 Nodes (Minimal Backward)")
    print(f"{'='*60}\n")

    if cem_params is None:
        # Use new default for CEM
        cem = CEM1Step(penalty_mult=500)
    else:
        cem = CEM1Step(**cem_params)
        
    t_minus_1_solutions = {}

    t = env.T_steps - 1

    for n_ups in range(t + 1):
        S_t = env.nodes[(t, n_ups)]
        
        # Targets are terminal payoffs
        target_up = env.terminal_payoffs[n_ups + 1]
        target_down = env.terminal_payoffs[n_ups]
        
        # Binary prices (1-step, risk-neutral)
        price_up = env.p * np.exp(-env.r * env.dt)
        price_down = (1 - env.p) * np.exp(-env.r * env.dt)
        
        print(f"Node ({t}, {n_ups}) @ S={S_t:.2f}")
        print(f"  Targets: Up=${target_up:.4f}, Down=${target_down:.4f}")
        
        solution, cost, error = cem.solve(target_up, target_down, price_up, price_down)
        
        b_up, b_down = solution
        
        t_minus_1_solutions[n_ups] = {
            'b_up': float(b_up),
            'b_down': float(b_down),
            'cost': float(cost),
            'error': float(error),
            'target_value': float(cost)
        }
        
        print(f"  Solution: b_up={b_up:.4f}, b_down={b_down:.4f}")
        print(f"  Cost (Target Value): ${cost:.6f}, Error: {error:.8f}\n")

    # Store in environment
    env.t_minus_1_targets = {n_ups: sol['target_value'] 
                             for n_ups, sol in t_minus_1_solutions.items()}
    env.t_minus_1_solutions = t_minus_1_solutions

    print(f"Stage 1 Complete!")
    print(f"Target values at T-1: {[f'${v:.4f}' for v in env.t_minus_1_targets.values()]}\n")

    return t_minus_1_solutions

# ----------------------------
# RL Environment
# ----------------------------

class DynamicHedgingEnv:
    """RL Environment for Stage 2: Learn to reach T-1 targets"""
    def __init__(self, base_env): 
        self.base_env = base_env
        self.T_steps = base_env.T_steps

        # State/action dimensions
        self.state_dim = 6
        self.action_dim = 2

    def reset(self):
        """Start at T=0"""
        self.current_time = 0
        self.current_n_ups = 0
        self.portfolio_value = 0.0
        self.total_cost = 0.0
        self.actions_taken = []

        return self._get_state()

    def _get_state(self):
        """State representation (normalized and stable)"""
        S_t = self.base_env.nodes[(self.current_time, self.current_n_ups)]
        time_to_t_minus_1 = (self.T_steps - 1) - self.current_time

        reachable_targets = []
        remaining = max(0, (self.T_steps - 1) - self.current_time)
        
        for add_ups in range(remaining + 1):
            future_n_ups = self.current_n_ups + add_ups
            if future_n_ups in self.base_env.t_minus_1_targets:
                prob = comb(remaining, add_ups) * (self.base_env.p ** add_ups) * ((1-self.base_env.p) ** (remaining - add_ups))
                reachable_targets.append((prob, self.base_env.t_minus_1_targets[future_n_ups]))

        expected_target = sum(p * v for p, v in reachable_targets) if len(reachable_targets) > 0 else 0.0

        n_ups_norm = self.current_n_ups / (self.current_time + 1)

        # Normalize state features by S0
        state = np.array([
            self.current_time / max(1, self.T_steps),
            S_t / self.base_env.S0,
            n_ups_norm,
            self.portfolio_value / self.base_env.S0,
            expected_target / self.base_env.S0,
            time_to_t_minus_1 / max(1, self.T_steps)
        ], dtype=np.float32)

        return state

    def step(self, action):
        """Take action (purchase binaries) and simulate price move"""
        # Action is [b_down, b_up]
        action = np.array(action, dtype=float)
        action = np.clip(action, 0.0, None)

        # Binary prices
        price_up = self.base_env.p * np.exp(-self.base_env.r * self.base_env.dt)
        price_down = (1 - self.base_env.p) * np.exp(-self.base_env.r * self.base_env.dt)

        # Cost of action
        step_cost = float(action[0] * price_down + action[1] * price_up)
        self.total_cost += step_cost

        # Store action
        self.actions_taken.append({
            'time': self.current_time,
            'n_ups': self.current_n_ups,
            'action': action.copy()
        })

        # Simulate move
        if np.random.rand() < self.base_env.p:
            self.current_n_ups += 1
            move_idx = 1 # action[1] (b_up) pays
        else:
            self.current_n_ups += 0
            move_idx = 0 # action[0] (b_down) pays

        self.portfolio_value += float(action[move_idx])
        self.current_time += 1

        # Check if we've reached T-1 (end of RL phase)
        done = (self.current_time >= self.T_steps - 1)

        if done:
            target_value = self.base_env.t_minus_1_targets[self.current_n_ups]
            error = abs(self.portfolio_value - target_value)

            # New Reward Function: reward = -abs(error)/S0 - 0.1 * abs(cost)/S0
            normalized_error = error / self.base_env.S0
            normalized_cost = self.total_cost / self.base_env.S0
            
            reward = -normalized_error - 0.1 * normalized_cost

            next_state = None
        else:
            # Intermediate: just negative normalized cost
            reward = -(step_cost / self.base_env.S0)
            next_state = self._get_state()
            error = 0.0

        info = {
            'cost': self.total_cost,
            'portfolio': self.portfolio_value,
            'target_error': error
        }

        return next_state, float(reward), bool(done), info


# ----------------------------
# SAC Agent
# ----------------------------

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=2.0):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Apply clipping to the standard deviation for entropy control
        std = torch.clamp(std, 0.05, 0.3) 

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action # Apply new max_action

        # Log prob correction for tanh
        log_prob = normal.log_prob(x_t)
        # Prevent numerical issues
        log_prob = log_prob - torch.log(self.max_action * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))
    
class SAC:
    def __init__(self, state_dim, action_dim, max_action=2.0,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=1e-4):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)

        if evaluate:
            # Deterministic for evaluation
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.actor.max_action
        else:
            # Stochastic for training
            action, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        # --- Critic Update ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            # Clipped Double-Q trick + Entropy term
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Actor and Alpha Update ---
        new_action, log_prob = self.actor.sample(state)
        q1 = self.critic1(state, new_action)
        q2 = self.critic2(state, new_action)
        q = torch.min(q1, q2)

        # Actor loss (maximize expected Q - alpha * entropy)
        actor_loss = (self.log_alpha.exp() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha loss (minimize difference from target entropy)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft Target Update ---
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ----------------------------
# Replay buffer
# ----------------------------

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

# ----------------------------
# Stage 2 training with improvements
# ----------------------------

# REMOVED: checkpoint_dir from function signature
def stage2_train_sac(base_env, n_episodes=None, 
                     early_stop_thresh=1e-3, early_stop_patience=1000): 
    
    WARMUP_EPISODES = 500
    MIN_CHECKPOINT_EPISODE = 5000 
    REQUIRED_CHECKPOINT_ERROR = 0.01 
    
    if n_episodes is None:
        n_episodes = 30000 * max(1, (base_env.T_steps - 1))
    
    best_actor_state_dict = None


    print(f"\n{'='*60}")
    print(f"STAGE 2: Training SAC (Forward to T-1)")
    print(f"Episodes: {n_episodes:,}")
    print(f"{'='*60}\n")

    env = DynamicHedgingEnv(base_env)
    
    agent = SAC(env.state_dim, env.action_dim, max_action=2.0,
                actor_lr=3e-4, critic_lr=3e-4, alpha_lr=1e-4) 
    replay_buffer = ReplayBuffer()

    best_error = float('inf')
    best_cost = float('inf')
    best_episode = -1

    recent_errors = deque(maxlen=200)
    recent_costs = deque(maxlen=200)

    print_freq = max(100, n_episodes // 20)

    consecutive_good = 0

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        
        # Determine if we are in the warm-up phase
        is_warmup = episode <= WARMUP_EPISODES
        
        # Simulate an episode
        while True:
            # Action selection: random for warm-up, SAC policy otherwise
            if is_warmup:
                # Random non-negative action (up to max_action)
                action = np.random.uniform(0.0, agent.actor.max_action, env.action_dim) 
            else:
                action = agent.select_action(state)
                action = np.clip(action, 0.0, None) # Safety clip
                
            next_state, reward, done, info = env.step(action)
            
            replay_buffer.push(state, action, reward, 
                              next_state if next_state is not None else state, done)
            
            # SAC updates start after warm-up
            if not is_warmup and len(replay_buffer) > 2000:
                # do a few updates per step to improve sample efficiency
                for _ in range(2):
                    agent.update(replay_buffer)
            
            if done:
                break
            
            state = next_state
        
        recent_errors.append(info['target_error'])
        recent_costs.append(info['cost'])
        
        # Checkpoint logic
        if episode >= MIN_CHECKPOINT_EPISODE and info['target_error'] < REQUIRED_CHECKPOINT_ERROR:
            # Only save if this model is better AND it meets the error threshold
            if info['target_error'] < best_error:
                best_error = info['target_error']
                best_cost = info['cost']
                best_episode = episode
                # Checkpoint actor state dictionary
                best_actor_state_dict = agent.actor.state_dict().copy()
                consecutive_good = 0
            else:
                if info['target_error'] < early_stop_thresh:
                    consecutive_good += 1
                else:
                    consecutive_good = 0
        
        # Early stopping condition
        if consecutive_good >= early_stop_patience:
            print(f"Early stopping at episode {episode} (consistent low error for {early_stop_patience} episodes)")
            break
        
        if episode % print_freq == 0:
            avg_error = float(np.mean(recent_errors)) if len(recent_errors) > 0 else float('nan')
            avg_cost = float(np.mean(recent_costs)) if len(recent_costs) > 0 else float('nan')
            progress = 100.0 * episode / n_episodes
            
            status = "(Warmup)" if is_warmup else ""

            print(f"Episode {episode:6d} ({progress:5.1f}%) {status} | Avg Cost: ${avg_cost:8.4f} | "
                  f"Avg T-1 Error: {avg_error:8.6f}")
            if best_episode != -1:
                print(f"                          | Best so far (ep {best_episode}): Cost=${best_cost:.4f}, Error={best_error:.6f}")

    print(f"\n{'='*60}")
    print("STAGE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"Best T-1 Target Error: {best_error:.6f} (episode {best_episode})")
    print(f"Best Cost: ${best_cost:.4f}")

    # Load best actor weights into agent (if saved)
    if best_actor_state_dict is not None:
        agent.actor.load_state_dict(best_actor_state_dict)

    return agent, env


# ----------------------------
# Full evaluation
# ----------------------------

def full_evaluation(base_env, agent, rl_env):
    print(f"\n{'='*60}")
    print("FULL STRATEGY EVALUATION (ALL PATHS)")
    print(f"{'='*60}\n")

    # Generate all possible paths from T=0 to Terminal
    def generate_all_paths(t, n_ups, path):
        if t == base_env.T_steps:
            return [path]
        
        paths = []
        # Down
        paths.extend(generate_all_paths(t+1, n_ups, path + [(t+1, n_ups, 'down')]))
        # Up  
        paths.extend(generate_all_paths(t+1, n_ups+1, path + [(t+1, n_ups+1, 'up')]))
        return paths

    all_paths = generate_all_paths(0, 0, [(0, 0, 'start')])

    print(f"Total paths to evaluate: {len(all_paths)}")

    path_results = []

    for path_idx, path in enumerate(all_paths):
        # Simulate this specific path with RL policy
        portfolio_value = 0.0
        total_cost = 0.0
        
        # Phase 1: T=0 to T-1 (RL decisions)
        for step_idx in range(base_env.T_steps - 1): # loop from t=0 up to t=T-2
            t, n_ups, _ = path[step_idx]
            
            # --- Recreate State at (t, n_ups) ---
            S_t = base_env.nodes[(t, n_ups)]
            time_to_t_minus_1 = (base_env.T_steps - 1) - t
            
            # Calculate expected target value
            reachable_targets = []
            remaining = max(0, (base_env.T_steps - 1) - t)
            for add_ups in range(remaining + 1):
                future_n_ups = n_ups + add_ups
                if future_n_ups in base_env.t_minus_1_targets:
                    prob = comb(remaining, add_ups) * (base_env.p ** add_ups) * ((1-base_env.p) ** (remaining - add_ups))
                    reachable_targets.append((prob, base_env.t_minus_1_targets[future_n_ups]))
            
            expected_target = sum(p * v for p, v in reachable_targets) if len(reachable_targets) > 0 else 0.0
            n_ups_norm = n_ups / (t + 1) # Normalization of n_ups
            
            # Normalized by S0
            state = np.array([
                t / max(1, base_env.T_steps),
                S_t / base_env.S0,
                n_ups_norm,
                portfolio_value / base_env.S0, 
                expected_target / base_env.S0,
                time_to_t_minus_1 / max(1, base_env.T_steps)
            ], dtype=np.float32)
            
            # Get RL action (deterministic)
            action = agent.select_action(state, evaluate=True)
            action = np.clip(action, 0.0, None)
            
            # --- Apply action ---
            price_up = base_env.p * np.exp(-base_env.r * base_env.dt)
            price_down = (1 - base_env.p) * np.exp(-base_env.r * base_env.dt)
            
            step_cost = float(action[0] * price_down + action[1] * price_up)
            total_cost += step_cost
            
            # Determine which binary pays based on path's next step
            next_t, next_n_ups, move = path[step_idx + 1]
            if move == 'up':
                portfolio_value += float(action[1])
            else:
                portfolio_value += float(action[0])
        
        # At T-1: Check portfolio value vs. target
        t_minus_1_t, t_minus_1_n_ups, _ = path[base_env.T_steps - 1]
        
        t_minus_1_error = abs(portfolio_value - base_env.t_minus_1_targets[t_minus_1_n_ups])
        
        # Phase 2: T-1 to Terminal (Apply CEM solution)
        t_minus_1_sol = base_env.t_minus_1_solutions[t_minus_1_n_ups]
        total_cost += t_minus_1_sol['cost']
        
        # Apply T-1 binaries
        terminal_t, terminal_n_ups, move = path[base_env.T_steps]
        if move == 'up':
            portfolio_value += t_minus_1_sol['b_up']
        else:
            portfolio_value += t_minus_1_sol['b_down']
        
        # Check final replication
        terminal_payoff = base_env.terminal_payoffs[terminal_n_ups]
        final_error = portfolio_value - terminal_payoff # final portfolio - target payoff
        
        path_results.append({
            'path': path,
            'portfolio': portfolio_value,
            'target': terminal_payoff,
            'error': final_error,
            'cost': total_cost,
            't_minus_1_error': t_minus_1_error
        })

    # Report results
    print(f"\nPath-by-Path Results:")
    for i, result in enumerate(path_results):
        path_str = '→'.join([f"({t},{n})" for t, n, _ in result['path']])
        status = "✓" if abs(result['error']) < 0.01 else "❌"
        print(f"  Path {i+1}: {path_str}")
        print(f"    Portfolio=${result['portfolio']:.6f}, Target=${result['target']:.6f}, "
              f"Error={result['error']:+.8f} {status}")
        print(f"    T-1 Error: {result['t_minus_1_error']:.6f}")

    # Aggregate statistics
    errors = [abs(r['error']) for r in path_results]
    costs = [r['cost'] for r in path_results]
    t_minus_1_errors = [r['t_minus_1_error'] for r in path_results]

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Total Absolute Error: {sum(errors):.10f}")
    print(f"Max Error: {max(errors):.10f}")
    print(f"Mean Error: {np.mean(errors):.10f}")
    print(f"\nT-1 Target Errors:")
    print(f"  Max: {max(t_minus_1_errors):.10f}")
    print(f"  Mean: {np.mean(t_minus_1_errors):.10f}")
    print(f"\nCost (all paths should be same): ${np.mean(costs):.8f} ± ${np.std(costs):.8f}")

    # Theoretical
    theoretical = sum(base_env.terminal_payoffs[i] * comb(base_env.T_steps, i) * (base_env.p ** i) * ((1-base_env.p) ** (base_env.T_steps - i)) *
                     np.exp(-base_env.r * base_env.T_steps * base_env.dt)
                     for i in range(base_env.T_steps + 1))

    avg_cost = np.mean(costs)
    print(f"\nTheoretical Price: ${theoretical:.8f}")
    print(f"Cost Error: ${abs(avg_cost - theoretical):.8f} ({100*abs(avg_cost - theoretical)/theoretical:.4f}%)")

    # Check for violations
    violations = sum(1 for e in errors if e > 0.01)
    print(f"\nViolations (error > 0.01): {violations}/{len(errors)} ({100*violations/len(errors):.1f}%)")


# ----------------------------
# Run example (T=2)
# ----------------------------

if __name__ == "__main__":
    print("="*60)
    print("HYBRID APPROACH: MINIMAL BACKWARD + SAC FORWARD (FINAL CLEANUP)")
    print("="*60)

    # Initial setup for a T=2 example
    base_env = BinomialEnvironment(T_steps=2, option_type='call')

    # Stage 1: Solve T-1 with gentler CEM penalties
    t1_solutions = stage1_solve_t_minus_1(base_env, cem_params=None)

    # Stage 2: Train SAC with improved defaults and parameters
    # The redundant checkpoint_dir argument is removed from the call
    agent, rl_env = stage2_train_sac(base_env, n_episodes=30000,
                                     early_stop_thresh=1e-3, early_stop_patience=1000)

    # Full evaluation on ALL paths
    full_evaluation(base_env, agent, rl_env)

    print("\n" + "="*60)
    print("Notes: All 'os' related code has been removed.")
    print("="*60)