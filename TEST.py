import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    """Neural network that outputs mean and log_std for delta and B"""
    def __init__(self, state_dim=2, hidden_dim=64, S0=100):
        super().__init__()
        self.S0 = S0  # Need S0 to compute hedge price constraint
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Output: [delta_mean, delta_log_std, B_mean, B_log_std]
        self.output = nn.Linear(hidden_dim, 4)
        
    def forward(self, state):
        features = self.network(state)
        output = self.output(features)
        
        # Split output into means and log_stds
        # Delta constrained to [0, 1]
        delta_mean = torch.sigmoid(output[:, 0])  # Delta in [0, 1]
        delta_log_std = output[:, 1].clamp(-3, -0.5)  # Tighter std for better convergence
        
        # B_mean - allow full range including negative values
        B_mean = torch.tanh(output[:, 2]) * 60  # B in roughly [-60, 60]
        B_log_std = output[:, 3].clamp(-3, -0.5)  # Tighter std
        
        return delta_mean, delta_log_std, B_mean, B_log_std
    
    def sample_action(self, state):
        """Sample delta and B from the policy distribution with constraint: 0 <= delta*S0 + B <= 50"""
        delta_mean, delta_log_std, B_mean, B_log_std = self.forward(state)
        
        delta_std = delta_log_std.exp()
        B_std = B_log_std.exp()
        
        # Sample from Gaussian distributions
        delta_dist = torch.distributions.Normal(delta_mean, delta_std)
        B_dist = torch.distributions.Normal(B_mean, B_std)
        
        delta = delta_dist.sample()
        B = B_dist.sample()
        
        # Clip delta to [0, 1] first
        delta = torch.clamp(delta, 0, 1)
        
        # Now enforce constraint: 0 <= delta*S0 + B <= 50
        # This means: -delta*S0 <= B <= 50 - delta*S0
        B_min = -delta * self.S0
        B_max = 50 - delta * self.S0
        B = torch.clamp(B, B_min, B_max)
        
        # Calculate log probabilities (before clamping for unbiased gradient)
        log_prob = delta_dist.log_prob(delta) + B_dist.log_prob(B)
        
        return delta, B, log_prob


class BinomialOptionEnvironment:
    """One-step binomial model for option pricing"""
    def __init__(self, S0=100, K=100, r=0.05, T=1.0, u=1.2, d=0.8):
        self.S0 = S0  # Initial stock price
        self.K = K    # Strike price
        self.r = r    # Risk-free rate
        self.T = T    # Time to maturity
        self.u = u    # Up factor
        self.d = d    # Down factor
        
        # Calculate risk-neutral probability
        self.q = (np.exp(r * T) - d) / (u - d)
        
        # Calculate stock prices at maturity
        self.Su = S0 * u
        self.Sd = S0 * d
        
        # Option payoffs (call option)
        self.Cu = max(self.Su - K, 0)
        self.Cd = max(self.Sd - K, 0)
        
        # Theoretical option price
        self.C0 = np.exp(-r * T) * (self.q * self.Cu + (1 - self.q) * self.Cd)
        
    def get_state(self):
        """Return state: [S0, K] (no step_type since we evaluate both)"""
        return np.array([self.S0, self.K], dtype=np.float32)
    
    def evaluate_hedge_stratified(self, delta, B):
        """
        Evaluate a hedge (delta, B) on BOTH up and down scenarios.
        
        P&L = Portfolio Value at T - Option Payoff at T
        
        For perfect hedge: P&L = 0 in both scenarios
        """
        pnls = []
        
        # Test on BOTH up and down moves
        for step_type in [1, 0]:  # 1 = up, 0 = down
            # Determine final stock price and option payoff
            if step_type == 1:  # Up move
                ST = self.Su
                CT = self.Cu
            else:  # Down move
                ST = self.Sd
                CT = self.Cd
            
            # Portfolio value at maturity
            portfolio_value = delta * ST + B * np.exp(self.r * self.T)
            
            # P&L = Portfolio value - Option payoff
            pnl = portfolio_value - CT
            
            pnls.append(pnl)
        
        pnl_up, pnl_down = pnls
        
        # Reward: negative MSE of P&L (mean squared error)
        # Divide by 100 to scale rewards to more reasonable magnitudes
        reward = -(pnl_up ** 2 + pnl_down ** 2) / 100.0
        
        return reward, pnl_up, pnl_down


class REINFORCEAgent:
    """REINFORCE algorithm with stratified reward evaluation"""
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.policy_net = PolicyNetwork(S0=env.S0)  # Pass S0 to network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # For tracking
        self.episode_rewards = []
        self.episode_pnls_up = []
        self.episode_pnls_down = []
        
        # Batch storage for more stable updates
        self.batch_log_probs = []
        self.batch_rewards = []
        
    def train_episode(self):
        """
        Train one episode:
        1. Sample a hedge (delta, B) from the policy
        2. Evaluate it on BOTH up and down scenarios (stratified)
        3. Get a single reward that reflects performance on both
        4. Store in batch for update
        """
        # Get state (same for both scenarios)
        state = self.env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Sample action from policy
        delta, B, log_prob = self.policy_net.sample_action(state_tensor)
        
        # Evaluate this hedge on BOTH up and down scenarios (stratified)
        reward, pnl_up, pnl_down = self.env.evaluate_hedge_stratified(
            delta.item(), 
            B.item()
        )
        
        # Store in batch
        self.batch_log_probs.append(log_prob)
        self.batch_rewards.append(reward)
        
        # Update policy if batch is full
        if len(self.batch_rewards) >= self.batch_size:
            self.update_policy()
        
        # Track metrics
        return reward, pnl_up, pnl_down
    
    def update_policy(self):
        """Update policy using batch of experiences"""
        if len(self.batch_rewards) == 0:
            return
        
        # Convert to tensors
        log_probs = torch.stack(self.batch_log_probs)
        rewards = torch.FloatTensor(self.batch_rewards)
        
        # Normalize rewards for stability (crucial for REINFORCE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Policy gradient
        policy_loss = -(log_probs * rewards).mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear batch
        self.batch_log_probs = []
        self.batch_rewards = []
    
    def train(self, n_episodes=1000, print_every=100):
        """Train the agent"""
        print("Starting training...")
        print(f"Theoretical option price: {self.env.C0:.4f}")
        print(f"Up payoff: {self.env.Cu:.4f}, Down payoff: {self.env.Cd:.4f}")
        print(f"Risk-neutral probability: {self.env.q:.4f}")
        print()
        
        for episode in range(n_episodes):
            reward, pnl_up, pnl_down = self.train_episode()
            
            self.episode_rewards.append(reward)
            self.episode_pnls_up.append(abs(pnl_up))
            self.episode_pnls_down.append(abs(pnl_down))
            
            if (episode + 1) % print_every == 0:
                recent_reward = np.mean(self.episode_rewards[-print_every:])
                recent_pnl_up = np.mean(self.episode_pnls_up[-print_every:])
                recent_pnl_down = np.mean(self.episode_pnls_down[-print_every:])
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"  Avg Reward: {recent_reward:.4f}")
                print(f"  Avg |P&L| Up: {recent_pnl_up:.4f}")
                print(f"  Avg |P&L| Down: {recent_pnl_down:.4f}")
                self.evaluate()
                print()
        
        # Final update for any remaining batch
        self.update_policy()
    
    def evaluate(self):
        """Evaluate current policy"""
        self.policy_net.eval()
        
        state = self.env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            delta, B, _ = self.policy_net.sample_action(state_tensor)
        
        reward, pnl_up, pnl_down = self.env.evaluate_hedge_stratified(
            delta.item(), 
            B.item()
        )
        
        hedge_price = delta.item() * self.env.S0 + B.item()
        
        print(f"  Current hedge: δ={delta.item():.4f}, B={B.item():.4f}")
        print(f"  Hedge price (δ*S0 + B): {hedge_price:.4f}")
        print(f"    Up scenario:   P&L={pnl_up:.4f}")
        print(f"    Down scenario: P&L={pnl_down:.4f}")
        print(f"    Total reward: {reward:.4f}")
        
        self.policy_net.train()
        return delta.item(), B.item(), pnl_up, pnl_down
    
    def plot_training(self):
        """Display training statistics"""
        print("\n" + "="*50)
        print("TRAINING STATISTICS")
        print("="*50)
        
        if len(self.episode_rewards) == 0:
            print("No training data available")
            return
        
        # Calculate statistics
        window = 50
        total_episodes = len(self.episode_rewards)
        
        print(f"Total Episodes: {total_episodes}")
        print(f"\nRewards:")
        print(f"  Final 50 episodes avg: {np.mean(self.episode_rewards[-window:]):.4f}")
        print(f"  Best reward: {np.max(self.episode_rewards):.4f}")
        print(f"  Worst reward: {np.min(self.episode_rewards):.4f}")
        
        print(f"\nHedging Error (|P&L|):")
        print(f"  Up moves - Final 50 avg: {np.mean(self.episode_pnls_up[-window:]):.4f}")
        print(f"  Down moves - Final 50 avg: {np.mean(self.episode_pnls_down[-window:]):.4f}")
        print(f"  Combined avg: {(np.mean(self.episode_pnls_up[-window:]) + np.mean(self.episode_pnls_down[-window:]))/2:.4f}")
        
        # Show trend
        if total_episodes >= 100:
            first_quarter_up = np.mean(self.episode_pnls_up[:total_episodes//4])
            last_quarter_up = np.mean(self.episode_pnls_up[-total_episodes//4:])
            improvement_up = ((first_quarter_up - last_quarter_up) / first_quarter_up) * 100
            
            first_quarter_down = np.mean(self.episode_pnls_down[:total_episodes//4])
            last_quarter_down = np.mean(self.episode_pnls_down[-total_episodes//4:])
            improvement_down = ((first_quarter_down - last_quarter_down) / first_quarter_down) * 100
            
            print(f"\nImprovement:")
            print(f"  Up scenarios: {improvement_up:.1f}% reduction in hedging error")
            print(f"  Down scenarios: {improvement_down:.1f}% reduction in hedging error")
        
        print("="*50)


# Example usage
if __name__ == "__main__":
    # Create environment
    env = BinomialOptionEnvironment(
        S0=100,    # Initial stock price
        K=100,     # Strike price
        r=0.05,    # Risk-free rate
        T=1.0,     # Time to maturity
        u=1.2,     # Up factor
        d=0.8      # Down factor
    )
    
    # Create and train agent
    agent = REINFORCEAgent(env, learning_rate=1e-4, batch_size=64)
    agent.train(n_episodes=10000, print_every=1000)
    
    # Show results
    agent.plot_training()
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    delta_learned, B_learned, pnl_up, pnl_down = agent.evaluate()
    
    # Calculate theoretical hedge
    q = env.q
    delta_theory = (env.Cu - env.Cd) / (env.Su - env.Sd)
    B_theory = np.exp(-env.r * env.T) * (env.u * env.Cd - env.d * env.Cu) / (env.u - env.d)
    hedge_price_theory = delta_theory * env.S0 + B_theory
    
    print("\nTheoretical Perfect Hedge:")
    print(f"  δ_theory = {delta_theory:.4f}")
    print(f"  B_theory = {B_theory:.4f}")
    print(f"  Hedge price (δ*S0 + B): {hedge_price_theory:.4f}")
    
    print("\nComparison:")
    print(f"  Delta difference: {abs(delta_learned - delta_theory):.4f}")
    print(f"  B difference: {abs(B_learned - B_theory):.4f}")
    
    print("\nConstraint Check:")
    hedge_price_learned = delta_learned * env.S0 + B_learned
    print(f"  Learned hedge price: {hedge_price_learned:.4f}")
    print(f"  Constraint: 0 <= hedge price <= 50")
    print(f"  Satisfied: {0 <= hedge_price_learned <= 50}")