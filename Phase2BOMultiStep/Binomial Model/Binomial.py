import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class BinomialEnvironment:
    """
    Binomial tree with node-paying binaries (immediate payouts).
    No root binary. Binaries at t=1,2,...,T pay immediately upon reaching that node.
    """
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
        self.p = (np.exp(r * dt) - self.d) / (self.u - self.d)
        
        # Build tree
        self.build_tree()
        self.calculate_terminal_payoffs()
        self.calculate_binary_prices()
        
    def build_tree(self):
        """Build recombining binomial tree - no root binary"""
        self.nodes = {}  # (t, n_ups) -> stock_price
        self.binary_nodes = []  # All nodes with binaries (excluding root)
        
        for t in range(self.T_steps + 1):
            for n_ups in range(t + 1):
                price = self.S0 * (self.u ** n_ups) * (self.d ** (t - n_ups))
                self.nodes[(t, n_ups)] = price
                
                # Binaries at all nodes except root (t=0)
                if t > 0:
                    self.binary_nodes.append((t, n_ups))
        
        self.n_binaries = len(self.binary_nodes)
        
        print(f"\n{'='*60}")
        print(f"Binomial Tree Structure (T={self.T_steps})")
        print(f"{'='*60}")
        print(f"Total binaries: {self.n_binaries}")
        
        for t in range(1, self.T_steps + 1):
            count = sum(1 for node_t, _ in self.binary_nodes if node_t == t)
            print(f"  Time t={t}: {count} binaries")
        
    def calculate_terminal_payoffs(self):
        """Calculate option payoff at terminal nodes"""
        self.terminal_payoffs = []
        
        for n_ups in range(self.T_steps + 1):
            S_T = self.nodes[(self.T_steps, n_ups)]
            if self.option_type == 'call':
                payoff = max(S_T - self.K, 0)
            else:
                payoff = max(self.K - S_T, 0)
            self.terminal_payoffs.append(payoff)
        
        print(f"\nTerminal Payoffs ({self.option_type}):")
        for n_ups, payoff in enumerate(self.terminal_payoffs):
            S_T = self.nodes[(self.T_steps, n_ups)]
            print(f"  n_ups={n_ups} (S={S_T:.2f}): ${payoff:.2f}")
    
    def calculate_binary_prices(self):
        """Calculate risk-neutral price for each binary"""
        self.binary_prices = []
        
        for t, n_ups in self.binary_nodes:
            # Probability of reaching (t, n_ups)
            from math import comb
            n_downs = t - n_ups
            prob = comb(t, n_ups) * (self.p ** n_ups) * ((1 - self.p) ** n_downs)
            
            # Discounted price (paid at t=0, pays at time t)
            price = np.exp(-self.r * t * self.dt) * prob
            self.binary_prices.append(price)
        
        print(f"\nBinary Prices (Risk-Neutral):")
        for i, (t, n_ups) in enumerate(self.binary_nodes):
            S = self.nodes[(t, n_ups)]
            print(f"  Binary_{i+1} b({t},{n_ups}) @ S={S:.2f}: ${self.binary_prices[i]:.6f}")
    
    def evaluate_strategy(self, binary_positions):
        """
        Evaluate strategy on all possible paths.
        Each path accumulates binary payoffs as it passes through nodes.
        """
        # Total initial cost
        cost = sum(binary_positions[i] * self.binary_prices[i] for i in range(self.n_binaries))
        
        # Generate all possible paths
        paths = []
        for terminal_n_ups in range(self.T_steps + 1):
            path = [(0, 0)]  # Start at root
            
            # Build path to terminal
            current_n_ups = 0
            for t in range(1, self.T_steps + 1):
                # Determine if this step was up or down
                if current_n_ups < terminal_n_ups:
                    # Need more ups
                    current_n_ups += 1
                # else: down (current_n_ups stays same)
                
                path.append((t, current_n_ups))
            
            paths.append(path)
        
        # Evaluate each path
        errors = []
        portfolio_values = []
        
        for path_idx, path in enumerate(paths):
            # Sum binary payoffs along this path
            portfolio_value = 0
            
            for t, n_ups in path:
                if t > 0:  # Skip root
                    # Find this node in binary_nodes
                    try:
                        node_idx = self.binary_nodes.index((t, n_ups))
                        portfolio_value += binary_positions[node_idx]
                    except ValueError:
                        pass  # Node not in binary list
            
            target_payoff = self.terminal_payoffs[path_idx]
            error = portfolio_value - target_payoff
            
            portfolio_values.append(portfolio_value)
            errors.append(error)
        
        errors = np.array(errors)
        total_abs_error = np.sum(np.abs(errors))
        max_error = np.max(np.abs(errors))
        mse_error = np.mean(errors ** 2)
        
        return cost, total_abs_error, max_error, mse_error, errors, portfolio_values


class PPOAgent:
    """PPO for learning optimal binary positions"""
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=lr
        )
        
    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean = self.actor(state_t)
            action_std = torch.ones_like(action_mean) * 1.0
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        
        return action.squeeze(0).numpy(), log_prob.item()
    
    def update(self, memory):
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.FloatTensor(np.array(memory['actions']))
        log_probs_old = torch.FloatTensor(memory['log_probs'])
        returns = torch.FloatTensor(memory['returns'])
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        for _ in range(10):
            action_mean = self.actor(states)
            values = self.critic(states).squeeze()
            
            action_std = torch.ones_like(action_mean) * 1.0
            dist = Normal(action_mean, action_std)
            
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            
            ratios = torch.exp(log_probs - log_probs_old)
            advantages = returns - values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = ((returns - values) ** 2).mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 
                max_norm=0.5
            )
            self.optimizer.step()


def train_hedging(T_steps=2, n_episodes=10000, option_type='call'):
    """Train agent to find optimal binary strategy"""
    
    env = BinomialEnvironment(S0=100, K=100, r=0.05, sigma=0.2, 
                              T_steps=T_steps, option_type=option_type)
    
    state_dim = 5
    action_dim = env.n_binaries
    
    agent = PPOAgent(state_dim, action_dim, lr=3e-4)
    
    print(f"\n{'='*60}")
    print(f"Training PPO for Binomial Hedging (Node-Paying Binaries)")
    print(f"Action space: {action_dim} binary positions")
    print(f"{'='*60}\n")
    
    memory = {'states': [], 'actions': [], 'log_probs': [], 'returns': []}
    best_cost = float('inf')
    best_positions = None
    best_error = float('inf')
    
    for episode in range(n_episodes):
        state = np.array([env.K/env.S0, env.r, env.sigma, T_steps, 1 if option_type=='call' else 0])
        
        action, log_prob = agent.select_action(state)
        
        cost, total_abs_error, max_error, mse_error, errors, portfolio_values = env.evaluate_strategy(action)
        
        # Reward: minimize cost + heavily penalize errors
        penalty = 10000 * mse_error + 5000 * max_error
        reward = -cost - penalty
        
        # Track best solution (prioritize low error, then low cost)
        if total_abs_error < best_error or (total_abs_error < 1.0 and cost < best_cost):
            best_error = total_abs_error
            best_cost = cost
            best_positions = action.copy()
        
        memory['states'].append(state)
        memory['actions'].append(action)
        memory['log_probs'].append(log_prob)
        memory['returns'].append(reward)
        
        if episode % 20 == 0 and episode > 0:
            agent.update(memory)
            memory = {'states': [], 'actions': [], 'log_probs': [], 'returns': []}
        
        if episode % 1000 == 0:
            print(f"Ep {episode:5d} | Cost: {cost:8.4f} | TotalAbsErr: {total_abs_error:7.4f} | MaxErr: {max_error:7.4f}")
            if best_positions is not None:
                print(f"          | Best so far: Cost=${best_cost:.4f}, TotalAbsErr={best_error:.4f}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    if best_positions is not None:
        cost, total_abs_error, max_error, mse_error, errors, portfolio_values = env.evaluate_strategy(best_positions)
        
        print(f"*** BEST REPLICATION STRATEGY ***")
        for i in range(env.n_binaries):
            print(f"  b{i+1} = {best_positions[i]:.8f}")
        
        print(f"  Initial cost: {cost:.8f}")
        
        print(f"  Cost breakdown:")
        for i, (t, n_ups) in enumerate(env.binary_nodes):
            contribution = best_positions[i] * env.binary_prices[i]
            print(f"    Binary_{i+1}: {contribution:.8f} ({best_positions[i]:.4f} units @ {env.binary_prices[i]:.4f})")
        
        print(f"  Replication verification:")
        for path_idx in range(len(errors)):
            portfolio = portfolio_values[path_idx]
            payoff = env.terminal_payoffs[path_idx]
            error = errors[path_idx]
            status = "✓" if abs(error) < 0.01 else "❌"
            print(f"    Scenario {path_idx+1}: Portfolio={portfolio:.8f}, Payoff={payoff:.8f}, Error={error:+.10f} {status}")
        
        print(f"  Total absolute error: {total_abs_error:.10f}")
        print(f"  Max absolute error: {max_error:.10f}")
        print(f"  MSE: {mse_error:.10f}")
    else:
        print("\nNo solution found")
    
    return agent, env, best_positions


if __name__ == "__main__":
    print("="*60)
    print("BINOMIAL HEDGING WITH NODE-PAYING BINARIES")
    print("No stock, no bond - only binaries!")
    print("Binaries pay immediately upon reaching the node")
    print("="*60)
    
    agent, env, best_pos = train_hedging(T_steps=2, n_episodes=10000, option_type='call')
    