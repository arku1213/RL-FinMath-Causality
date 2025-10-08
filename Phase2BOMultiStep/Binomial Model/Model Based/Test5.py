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
# Policy Network (Larger, with more capacity)
# ----------------------------

class PolicyNetwork(nn.Module):
    """Deep neural network that learns optimal binary allocation."""
    def __init__(self, state_dim, n_binaries, hidden_dim=512):
        super().__init__()
        
        self.n_binaries = n_binaries
        
        # Deeper network with more capacity
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, n_binaries)
        )
        
    def forward(self, state):
        return self.network(state)


# ----------------------------
# Model-Based Planner
# ----------------------------

class ModelBasedPlanner:
    """Uses the perfect model to simulate outcomes."""
    def __init__(self, environment, n_simulation_paths=100):
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
# Pure Deep Model-Based RL Agent
# ----------------------------

class PureDeepRLAgent:
    """Pure deep RL - learns from terminal payoffs only, no backward induction."""
    def __init__(self, environment, state_dim=6, hidden_dim=512, lr=3e-4):
        self.env = environment
        self.n_binaries = environment.n_binaries
        
        # Deeper policy network
        self.policy = PolicyNetwork(state_dim, self.n_binaries, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=200
        )
        
        # Model-based planner (more simulations)
        self.planner = ModelBasedPlanner(environment, n_simulation_paths=100)
        
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
        Compute loss - ONLY checks terminal payoffs (pure deep RL).
        No backward induction, no intermediate targets.
        """
        # Convert tensor to holdings dictionary for simulation
        binary_holdings = {}
        for i, (t, n_ups) in enumerate(self.env.binaries):
            binary_holdings[(t, n_ups)] = binary_values_tensor[i].item()
        
        # Simulate outcomes
        outcomes = self.planner.simulate_outcomes(binary_holdings)
        
        # Compute losses - ONLY terminal errors
        errors_list = []
        costs_list = []
        
        for outcome in outcomes:
            # Recompute portfolio value with gradients
            portfolio_value = torch.tensor(0.0, dtype=torch.float32)
            cost = torch.tensor(0.0, dtype=torch.float32)
            
            for i, (t, n_ups) in enumerate(self.env.binaries):
                # Cost contribution
                cost += binary_values_tensor[i] * self.env.binary_prices[i]
                
                # Portfolio value contribution (if node is in path)
                if (t, n_ups) in [(node[0], node[1]) for node in outcome['path'][1:]]:
                    portfolio_value = portfolio_value + binary_values_tensor[i]
            
            # Error for this path (terminal only!)
            target = outcome['target']
            error = torch.abs(portfolio_value - target)
            
            errors_list.append(error)
            costs_list.append(cost)
        
        # Stack into tensors
        errors = torch.stack(errors_list)
        costs = torch.stack(costs_list)
        
        # Loss components
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
        cost_penalty = 0.1 * (cost_mean - theoretical_price) ** 2  # Increased weight
        
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
    
    def train(self, n_episodes=5000, eval_freq=100):
        """Train the agent using pure deep model-based RL."""
        print(f"\n{'='*60}")
        print("PURE DEEP MODEL-BASED RL TRAINING")
        print("NO backward induction - learns from terminal payoffs only!")
        print(f"Episodes: {n_episodes}")
        print(f"Simulating {self.planner.n_simulation_paths} paths per update")
        print(f"{'='*60}\n")
        
        for episode in range(1, n_episodes + 1):
            # Train step
            loss, mse, cost = self.train_step()
            
            # Update learning rate based on loss
            if episode % 50 == 0:
                self.scheduler.step(loss)
            
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
                
                print(f"Episode {episode:5d} | Loss: {loss:.6f} | "
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
        
        results.append({
            'path': path,
            'portfolio': portfolio_value,
            'target': target,
            'error': error,
            'abs_error': abs(error),
            'cost': cost
        })
    
    # Print results
    print(f"\nPath-by-Path Results:")
    for i, result in enumerate(results):
        path_str = '→'.join([f"({t},{n})" for t, n, _ in result['path']])
        status = "✓" if result['abs_error'] < 0.01 else "❌"
        print(f"  Path {i+1}: {path_str}")
        print(f"    Portfolio=${result['portfolio']:.6f}, Target=${result['target']:.6f}, "
              f"Error={result['error']:+.8f} {status}")
    
    # Aggregate statistics
    errors = [r['abs_error'] for r in results]
    costs = [r['cost'] for r in results]
    
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Total Absolute Error: {sum(errors):.10f}")
    print(f"Max Error: {max(errors):.10f}")
    print(f"Mean Error: {np.mean(errors):.10f}")
    
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
    print("PURE DEEP MODEL-BASED RL (No Backward Induction)")
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
    
    # Create agent
    agent = PureDeepRLAgent(
        environment=env,
        state_dim=6,
        hidden_dim=512,
        lr=3e-4
    )
    
    # Train using pure deep model-based RL
    agent.train(n_episodes=5000, eval_freq=100)
    
    # Exhaustive evaluation
    results = exhaustive_evaluation(agent)
    
    print("\n" + "="*60)
    print("PURE DEEP RL APPROACH:")
    print("- NO backward induction used")
    print("- NO intermediate targets")
    print("- Learns ONLY from terminal payoff errors")
    print("- Network must discover structure on its own")
    print("="*60)