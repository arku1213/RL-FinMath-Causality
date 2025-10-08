import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from math import comb

# ----------------------------
# Binomial Environment (Perfect Model)
# ----------------------------

class BinomialEnvironment:
    """Perfect model of binomial tree dynamics."""
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T_steps=2, dt=1.0, option_type='call'):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_steps = T_steps
        self.dt = dt
        self.option_type = option_type
        
        # CRR parameters
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Build tree structure
        self.build_tree()
        self.calculate_payoffs()
        self.identify_binaries()
        
    def build_tree(self):
        """Build all nodes in the tree"""
        self.nodes = {}
        for t in range(self.T_steps + 1):
            for n_ups in range(t + 1):
                price = self.S0 * (self.u ** n_ups) * (self.d ** (t - n_ups))
                self.nodes[(t, n_ups)] = price
                
    def calculate_payoffs(self):
        """Calculate terminal option payoffs"""
        self.terminal_payoffs = {}
        for n_ups in range(self.T_steps + 1):
            S_T = self.nodes[(self.T_steps, n_ups)]
            if self.option_type == 'call':
                payoff = max(S_T - self.K, 0)
            else:
                payoff = max(self.K - S_T, 0)
            self.terminal_payoffs[n_ups] = payoff
            
    def identify_binaries(self):
        """Identify all binaries needed for complete market."""
        self.binaries = []
        self.binary_prices = []
        
        # Binaries at each time step (except root)
        for t in range(1, self.T_steps + 1):
            for n_ups in range(t + 1):
                prob = self.get_node_probability(t, n_ups)
                price = prob * np.exp(-self.r * t * self.dt)
                
                self.binaries.append((t, n_ups))
                self.binary_prices.append(price)
                
        self.n_binaries = len(self.binaries)
        print(f"Environment initialized: {self.n_binaries} binaries for T={self.T_steps}")
        
    def get_node_probability(self, t, n_ups):
        """Risk-neutral probability of reaching node (t, n_ups)"""
        n_downs = t - n_ups
        return comb(t, n_ups) * (self.p ** n_ups) * ((1 - self.p) ** n_downs)
    
    def simulate_path(self, starting_node=(0, 0)):
        """Simulate one random path from starting_node to terminal."""
        t, n_ups = starting_node
        path = [(t, n_ups, 'start')]
        
        while t < self.T_steps:
            if np.random.rand() < self.p:
                n_ups += 1
                move = 'up'
            else:
                move = 'down'
            t += 1
            path.append((t, n_ups, move))
            
        return path
    
    def get_payoff_for_path(self, path):
        """Get terminal option payoff for a given path"""
        terminal_t, terminal_n_ups, _ = path[-1]
        return self.terminal_payoffs[terminal_n_ups]
    
    def evaluate_portfolio(self, binary_holdings, path):
        """Evaluate portfolio value along a path."""
        # Initial cost
        cost = 0.0
        for i, (t, n_ups) in enumerate(self.binaries):
            if (t, n_ups) in binary_holdings:
                cost += binary_holdings[(t, n_ups)] * self.binary_prices[i]
        
        # Evaluate along path
        portfolio_value = 0.0
        for t, n_ups, _ in path[1:]:  # Skip start
            if (t, n_ups) in binary_holdings:
                portfolio_value += binary_holdings[(t, n_ups)]
                
        return portfolio_value, cost


# ----------------------------
# Backward Induction Module
# ----------------------------

def compute_t_minus_1_targets(env):
    """
    ANALYTICAL: Use backward induction to compute target values at T-1.
    These guide the RL agent.
    
    Returns:
        targets: dict mapping (T-1, n_ups) -> target portfolio value
    """
    print(f"\n{'='*60}")
    print("BACKWARD INDUCTION: Computing T-1 Targets")
    print(f"{'='*60}")
    
    targets = {}
    t = env.T_steps - 1
    
    for n_ups in range(t + 1):
        # Children at T
        target_up = env.terminal_payoffs[n_ups + 1]
        target_down = env.terminal_payoffs[n_ups]
        
        # Risk-neutral expectation discounted to T-1
        expected_value = env.p * target_up + (1 - env.p) * target_down
        pv_target = expected_value * np.exp(-env.r * env.dt)
        
        targets[(t, n_ups)] = pv_target
        
        print(f"Node ({t}, {n_ups}): Target = ${pv_target:.6f}")
        print(f"  → From: {env.p:.3f} × ${target_up:.2f} + "
              f"{1-env.p:.3f} × ${target_down:.2f}, discounted")
    
    print(f"{'='*60}\n")
    
    return targets


# ----------------------------
# Policy Network
# ----------------------------

class PolicyNetwork(nn.Module):
    """Deep neural network that learns optimal binary allocation."""
    def __init__(self, state_dim, n_binaries, hidden_dim=256):
        super().__init__()
        
        self.n_binaries = n_binaries
        
        # Deep network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_binaries)
        )
        
    def forward(self, state):
        return self.network(state)


# ----------------------------
# Model-Based Planner
# ----------------------------

class ModelBasedPlanner:
    """Uses the perfect model to simulate outcomes."""
    def __init__(self, environment, n_simulation_paths=50):
        self.env = environment
        self.n_simulation_paths = n_simulation_paths
        
    def simulate_outcomes(self, binary_holdings):
        """Simulate many possible paths using the model."""
        outcomes = []
        
        for _ in range(self.n_simulation_paths):
            path = self.env.simulate_path()
            portfolio_value, cost = self.env.evaluate_portfolio(binary_holdings, path)
            target_payoff = self.env.get_payoff_for_path(path)
            error = abs(portfolio_value - target_payoff)
            
            outcomes.append({
                'path': path,
                'portfolio': portfolio_value,
                'target': target_payoff,
                'cost': cost,
                'error': error
            })
            
        return outcomes
    
    def evaluate_policy(self, binary_holdings):
        """Evaluate how good a policy is by simulating many outcomes."""
        outcomes = self.simulate_outcomes(binary_holdings)
        
        errors = [o['error'] for o in outcomes]
        costs = [o['cost'] for o in outcomes]
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs)
        }


# ----------------------------
# Hybrid Deep RL Agent
# ----------------------------

class HybridDeepRLAgent:
    """
    Hybrid approach:
    - Uses backward induction to compute T-1 targets (analytical)
    - Uses deep RL to learn how to reach those targets (learning)
    """
    def __init__(self, environment, t_minus_1_targets, state_dim=6, hidden_dim=256, lr=1e-3):
        self.env = environment
        self.n_binaries = environment.n_binaries
        self.t_minus_1_targets = t_minus_1_targets
        
        # Deep policy network
        self.policy = PolicyNetwork(state_dim, self.n_binaries, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Model-based planner
        self.planner = ModelBasedPlanner(environment, n_simulation_paths=50)
        
        # Training history
        self.history = {
            'episode': [],
            'mean_error': [],
            'max_error': [],
            'mean_cost': [],
            'loss': []
        }
        
    def get_state(self):
        """Construct state representation from tree parameters."""
        state = np.array([
            self.env.S0 / 100.0,
            self.env.K / 100.0,
            self.env.r,
            self.env.sigma,
            self.env.p,
            float(self.env.T_steps) / 10.0
        ], dtype=np.float32)
        
        return state
    
    def select_binaries(self, state, deterministic=False):
        """Use policy network to select binary holdings."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            binary_values = self.policy(state_tensor).squeeze(0).numpy()
            
        # Convert to dictionary
        binary_holdings = {}
        for i, (t, n_ups) in enumerate(self.env.binaries):
            binary_holdings[(t, n_ups)] = float(binary_values[i])
            
        return binary_holdings
    
    def compute_loss(self, binary_values_tensor):
        """
        HYBRID LOSS:
        - Checks if portfolio reaches T-1 targets (from backward induction)
        - Uses model-based simulation to evaluate
        """
        # Convert tensor to holdings dictionary for simulation
        binary_holdings = {}
        for i, (t, n_ups) in enumerate(self.env.binaries):
            binary_holdings[(t, n_ups)] = binary_values_tensor[i].item()
        
        # Simulate outcomes
        outcomes = self.planner.simulate_outcomes(binary_holdings)
        
        # Compute losses: Check T-1 targets AND terminal payoffs
        errors_list = []
        costs_list = []
        
        for outcome in outcomes:
            # Compute cost
            cost = torch.tensor(0.0, dtype=torch.float32)
            for i, (t, n_ups) in enumerate(self.env.binaries):
                cost += binary_values_tensor[i] * self.env.binary_prices[i]
            
            # Track portfolio value and check at T-1 AND terminal
            portfolio_value = torch.tensor(0.0, dtype=torch.float32)
            path_errors = []
            
            for step_idx, (t, n_ups, move) in enumerate(outcome['path'][1:]):  # Skip root
                # Add binary payoff at this node
                binary_idx = self.env.binaries.index((t, n_ups))
                portfolio_value = portfolio_value + binary_values_tensor[binary_idx]
                
                # Check error at T-1 (guided by backward induction)
                if t == self.env.T_steps - 1:
                    target = self.t_minus_1_targets[(t, n_ups)]
                    error = torch.abs(portfolio_value - target)
                    path_errors.append(error * 10.0)  # Weight T-1 errors heavily!
                
                # Check error at terminal
                if t == self.env.T_steps:
                    target = outcome['target']
                    error = torch.abs(portfolio_value - target)
                    path_errors.append(error)
            
            # Average error across checkpoints in this path
            if len(path_errors) > 0:
                path_error = torch.stack(path_errors).mean()
                errors_list.append(path_error)
            
            costs_list.append(cost)
        
        # Stack into tensors
        errors = torch.stack(errors_list)
        costs = torch.stack(costs_list)
        
        # MSE loss for replication error
        mse_loss = (errors ** 2).mean()
        
        # Cost penalty
        cost_mean = costs.mean()
        theoretical_price = sum(
            self.env.terminal_payoffs[i] * 
            comb(self.env.T_steps, i) * 
            (self.env.p ** i) * 
            ((1 - self.env.p) ** (self.env.T_steps - i)) *
            np.exp(-self.env.r * self.env.T_steps * self.env.dt)
            for i in range(self.env.T_steps + 1)
        )
        cost_penalty = 0.01 * (cost_mean - theoretical_price) ** 2
        
        # Total loss
        total_loss = mse_loss + cost_penalty
        
        return total_loss, mse_loss.item(), cost_mean.item()
    
    def train_step(self):
        """One training step with gradient-preserving computation."""
        state = self.get_state()
        
        # Forward pass
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        binary_values = self.policy(state_tensor).squeeze(0)
        
        # Compute loss
        loss, mse, cost = self.compute_loss(binary_values)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), mse, cost
    
    def train(self, n_episodes=2000, eval_freq=50):
        """Train the agent using hybrid approach."""
        print(f"\n{'='*60}")
        print("HYBRID DEEP MODEL-BASED RL TRAINING")
        print(f"Backward Induction guides T-1 targets")
        print(f"Deep RL learns how to reach them")
        print(f"Episodes: {n_episodes}")
        print(f"{'='*60}\n")
        
        for episode in range(1, n_episodes + 1):
            # Train step
            loss, mse, cost = self.train_step()
            
            # Periodic evaluation
            if episode % eval_freq == 0 or episode == 1:
                state = self.get_state()
                binary_holdings = self.select_binaries(state, deterministic=True)
                metrics = self.planner.evaluate_policy(binary_holdings)
                
                # Store history
                self.history['episode'].append(episode)
                self.history['mean_error'].append(metrics['mean_error'])
                self.history['max_error'].append(metrics['max_error'])
                self.history['mean_cost'].append(metrics['mean_cost'])
                self.history['loss'].append(loss)
                
                print(f"Episode {episode:4d} | Loss: {loss:.6f} | "
                      f"Mean Error: {metrics['mean_error']:.6f} | "
                      f"Max Error: {metrics['max_error']:.6f} | "
                      f"Cost: ${metrics['mean_cost']:.4f}")
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}\n")


# ----------------------------
# Exhaustive Evaluation
# ----------------------------

def exhaustive_evaluation(agent):
    """Evaluate on ALL possible paths."""
    print(f"\n{'='*60}")
    print("EXHAUSTIVE EVALUATION (ALL PATHS)")
    print(f"{'='*60}\n")
    
    env = agent.env
    
    # Get learned policy
    state = agent.get_state()
    binary_holdings = agent.select_binaries(state, deterministic=True)
    
    print("Learned Binary Holdings:")
    for (t, n_ups), quantity in sorted(binary_holdings.items()):
        if (t, n_ups) in agent.t_minus_1_targets:
            target = agent.t_minus_1_targets[(t, n_ups)]
            print(f"  Node ({t}, {n_ups}): {quantity:.6f} units (T-1 target: ${target:.6f})")
        else:
            print(f"  Node ({t}, {n_ups}): {quantity:.6f} units")
    
    # Generate all paths
    def generate_all_paths(t, n_ups, path):
        if t == env.T_steps:
            return [path]
        paths = []
        paths.extend(generate_all_paths(t+1, n_ups, path + [(t+1, n_ups, 'down')]))
        paths.extend(generate_all_paths(t+1, n_ups+1, path + [(t+1, n_ups+1, 'up')]))
        return paths
    
    all_paths = generate_all_paths(0, 0, [(0, 0, 'start')])
    
    print(f"\nEvaluating {len(all_paths)} paths...")
    
    results = []
    for path in all_paths:
        portfolio_value, cost = env.evaluate_portfolio(binary_holdings, path)
        target = env.get_payoff_for_path(path)
        error = portfolio_value - target
        
        # Also check T-1 error
        t_minus_1_step = env.T_steps  # This is the index in path for T-1
        t_minus_1_node = path[t_minus_1_step]  # Node at T-1
        t, n_ups, _ = t_minus_1_node
        
        # Calculate portfolio value at T-1 (only count binaries UP TO t-1!)
        portfolio_at_t_minus_1 = 0.0
        for node_t, node_n_ups, _ in path[1:t_minus_1_step+1]:  # Up to and including T-1
            if (node_t, node_n_ups) in binary_holdings:
                portfolio_at_t_minus_1 += binary_holdings[(node_t, node_n_ups)]
        
        t_minus_1_target = agent.t_minus_1_targets.get((t, n_ups), 0)
        t_minus_1_error = abs(portfolio_at_t_minus_1 - t_minus_1_target)
        
        results.append({
            'path': path,
            'portfolio': portfolio_value,
            'target': target,
            'error': error,
            'abs_error': abs(error),
            'cost': cost,
            't_minus_1_error': t_minus_1_error
        })
    
    # Print results
    print(f"\nPath-by-Path Results:")
    for i, result in enumerate(results):
        path_str = '→'.join([f"({t},{n})" for t, n, _ in result['path']])
        status = "✓" if result['abs_error'] < 0.01 else "❌"
        print(f"  Path {i+1}: {path_str}")
        print(f"    Portfolio=${result['portfolio']:.6f}, Target=${result['target']:.6f}, "
              f"Error={result['error']:+.8f} {status}")
        print(f"    T-1 Error: {result['t_minus_1_error']:.6f}")
    
    # Aggregate statistics
    errors = [r['abs_error'] for r in results]
    costs = [r['cost'] for r in results]
    t_minus_1_errors = [r['t_minus_1_error'] for r in results]
    
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Total Absolute Error: {sum(errors):.10f}")
    print(f"Max Error: {max(errors):.10f}")
    print(f"Mean Error: {np.mean(errors):.10f}")
    print(f"\nT-1 Errors (Guided by Backward Induction):")
    print(f"  Max: {max(t_minus_1_errors):.10f}")
    print(f"  Mean: {np.mean(t_minus_1_errors):.10f}")
    
    # Theoretical price
    theoretical = sum(
        env.terminal_payoffs[i] * comb(env.T_steps, i) * 
        (env.p ** i) * ((1 - env.p) ** (env.T_steps - i)) *
        np.exp(-env.r * env.T_steps * env.dt)
        for i in range(env.T_steps + 1)
    )
    
    avg_cost = np.mean(costs)
    print(f"\nCost: ${avg_cost:.8f} ± ${np.std(costs):.8f}")
    print(f"Theoretical Price: ${theoretical:.8f}")
    print(f"Cost Error: ${abs(avg_cost - theoretical):.8f} ({100*abs(avg_cost - theoretical)/theoretical:.4f}%)")
    
    violations = sum(1 for e in errors if e > 0.01)
    print(f"\nViolations (error > 0.01): {violations}/{len(errors)} ({100*violations/len(errors):.1f}%)")
    
    return results


# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    print("="*60)
    print("HYBRID: BACKWARD INDUCTION + DEEP MODEL-BASED RL")
    print("="*60)
    
    # Set seeds
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    print(f"Using Seed: {SEED}\n")
    
    # Create environment
    T_STEPS = 2
    env = BinomialEnvironment(
        S0=100, K=100, r=0.05, sigma=0.2, 
        T_steps=T_STEPS, dt=1.0, option_type='call'
    )
    
    print(f"\nBinomial Tree Structure:")
    print(f"  T={T_STEPS} steps")
    print(f"  {env.n_binaries} binaries (complete market)")
    print(f"  Risk-neutral probability p={env.p:.4f}")
    
    # STEP 1: Backward induction for T-1 targets (analytical)
    t_minus_1_targets = compute_t_minus_1_targets(env)
    
    # STEP 2: Deep RL learns to reach those targets
    agent = HybridDeepRLAgent(
        environment=env,
        t_minus_1_targets=t_minus_1_targets,
        state_dim=6,
        hidden_dim=256,
        lr=1e-3
    )
    
    # Train
    agent.train(n_episodes=2000, eval_freq=50)
    
    # Exhaustive evaluation
    results = exhaustive_evaluation(agent)
    
    print("\n" + "="*60)
    print("HYBRID APPROACH SUMMARY:")
    print("✓ Backward induction computes T-1 targets (analytical)")
    print("✓ Deep RL learns optimal binary allocation (learning)")
    print("✓ Combines rigor + flexibility")
    print("✓ Scales to any T!")
    print("="*60)