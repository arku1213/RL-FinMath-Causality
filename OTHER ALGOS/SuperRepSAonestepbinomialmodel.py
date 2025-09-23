# -------------------------
# Super-Replication via Simulated Annealing
# -------------------------

import random
import numpy as np

# -------------------------
# Environment: 1-step binomial
# -------------------------

class OneStepBinomialEnv:
    def __init__(self,
                 S0=100.0,
                 K=110.0,
                 up_price=140.0,
                 down_price=80.0,
                 probability=0.5,
                 seed=0):
        self.S0 = float(S0)
        self.K = float(K)
        self.up_price = float(up_price)
        self.down_price = float(down_price)
        self.probability = float(probability)
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self):
        self.state = np.array([self.S0, 1.0], dtype=np.float32)
        return self.state
    
    # Super-replication step
    def step(self, delta, B):
        """
        delta: hedge ratio (float)
        B: bank position (float)
        """
        # simulate stochastic stock move
        is_up = self.rng.random() < self.probability
        S_T = self.up_price if is_up else self.down_price
        
        # European call payoff
        payoff = max(S_T - self.K, 0.0)
        
        # portfolio value
        portfolio = delta * S_T + B
        
        # super-replication error
        shortfall = max(0.0, payoff - portfolio)
        reward = -shortfall  # reward = 0 if portfolio >= payoff, negative otherwise
        
        next_state = np.array([S_T, 0.0], dtype=np.float32)
        done = True
        
        info = {
            'is_up': is_up,
            'S_T': S_T,
            'payoff': payoff,
            'portfolio': portfolio,
            'shortfall': shortfall
        }
        
        return next_state, reward, done, info

# -------------------------
# Simulated Annealing for super-replication
# -------------------------

def simulated_annealing(env, 
                        max_iter=5000, 
                        initial_temp=10.0, 
                        alpha=0.995, 
                        delta_bounds=(-1, 1), 
                        B_bounds=(-50, 50)):
    """
    env: instance of OneStepBinomialEnv
    max_iter: number of SA iterations
    initial_temp: starting temperature
    alpha: cooling rate (temperature multiplier per iteration)
    delta_bounds: allowed range for hedge ratio
    B_bounds: allowed range for bank position
    """
    
    # Initialize delta and B randomly within bounds
    delta = random.uniform(*delta_bounds)
    B = random.uniform(*B_bounds)
    
    # Evaluate initial portfolio
    _, reward, _, _ = env.step(delta, B)
    
    best_delta, best_B = delta, B
    best_reward = reward
    temp = initial_temp
    
    for i in range(max_iter):
        # propose new candidate by small random perturbation
        delta_new = np.clip(delta + random.uniform(-0.1, 0.1), *delta_bounds)
        B_new = np.clip(B + random.uniform(-1.0, 1.0), *B_bounds)
        
        # evaluate new candidate
        _, reward_new, _, _ = env.step(delta_new, B_new)
        
        # calculate improvement
        delta_reward = reward_new - reward
        
        # accept new solution if better, or probabilistically if worse
        if delta_reward >= 0 or random.random() < np.exp(delta_reward / temp):
            delta, B, reward = delta_new, B_new, reward_new
            
            # update best found so far
            if reward_new > best_reward:
                best_delta, best_B, best_reward = delta_new, B_new, reward_new
        
        # cool down
        temp *= alpha
        
        # optional: print progress every 500 iterations
        if (i + 1) % 500 == 0:
            print(f"Iteration {i+1:5d} | best_reward {best_reward:.4f} | "
                  f"Δ={best_delta:.3f}, B={best_B:.3f} | temp={temp:.4f}")
    
    return best_delta, best_B, best_reward

# -------------------------
# Run example
# -------------------------
if __name__ == "__main__":
    env = OneStepBinomialEnv(S0=100.0, K=110.0, up_price=140.0, down_price=80.0, probability=0.5, seed=42)
    
    best_delta, best_B, best_reward = simulated_annealing(env,
                                                          max_iter=5000,
                                                          initial_temp=10.0,
                                                          alpha=0.995)
    
    # implied fair price
    fair_price = best_delta * env.S0 + best_B
    
    print("\n--- Simulated Annealing Super-Replication ---")
    print(f"Best Δ = {best_delta:.4f}")
    print(f"Best B = {best_B:.4f}")
    print(f"Implied fair price X = Δ * S0 + B = {fair_price:.4f}")
    print(f"Best reward (negative shortfall) = {best_reward:.6f}")
