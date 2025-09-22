# -------------------------
# Simulated Annealing for 1-step binomial hedging
# -------------------------

import numpy as np
import random

# We'll reuse the same environment
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
        self.t = 0
        self.state = np.array([self.S0, 1.0], dtype=np.float32)
        return self.state

    def step(self, delta, B):
        """Single-step environment evaluation"""
        delta = float(delta)
        B = float(B)

        is_up = self.rng.random() < self.probability
        S_T = self.up_price if is_up else self.down_price
        payoff = max(S_T - self.K, 0.0)
        portfolio = delta * S_T + B
        err = portfolio - payoff
        reward = -(err ** 2)  # L2 error for pure replication

        next_state = np.array([S_T, 0.0], dtype=np.float32)
        done = True

        info = {
            "is_up": is_up,
            "S_T": S_T,
            "payoff": payoff,
            "portfolio": portfolio,
            "err": err
        }
        return next_state, reward, done, info

# -------------------------
# Simulated Annealing optimizer
# -------------------------

def simulated_annealing_hedge(env,
                              delta_range=(-1, 1),
                              B_range=(-50, 50),
                              initial_temp=10.0,
                              final_temp=0.01,
                              alpha=0.99,  # cooling rate
                              max_iter=10000,
                              seed=42):
    """
    env: OneStepBinomialEnv
    delta_range: tuple (min, max) for hedge ratio
    B_range: tuple (min, max) for bank borrowing
    initial_temp: starting temperature
    final_temp: minimum temperature to stop
    alpha: cooling rate
    max_iter: max number of iterations
    """
    random.seed(seed)
    np.random.seed(seed)

    # initialize delta and B randomly
    delta = random.uniform(*delta_range)
    B = random.uniform(*B_range)

    # evaluate initial solution
    _, reward, _, _ = env.step(delta, B)
    current_loss = -reward  # since reward = -err^2
    best_delta, best_B = delta, B
    best_loss = current_loss

    temp = initial_temp

    for i in range(max_iter):
        # perturb delta and B slightly
        new_delta = delta + random.uniform(-0.05, 0.05)
        new_B = B + random.uniform(-1.0, 1.0)

        # clip to allowed ranges
        new_delta = max(min(new_delta, delta_range[1]), delta_range[0])
        new_B = max(min(new_B, B_range[1]), B_range[0])

        # evaluate new solution
        _, reward, _, _ = env.step(new_delta, new_B)
        new_loss = -reward

        # acceptance probability
        if new_loss < current_loss:
            accept = True  # better solution
        else:
            # worse solution can be accepted with probability exp(-ΔE / T)
            prob = np.exp(-(new_loss - current_loss) / temp)
            accept = random.random() < prob

        if accept:
            delta, B = new_delta, new_B
            current_loss = new_loss
            # update best
            if current_loss < best_loss:
                best_delta, best_B = delta, B
                best_loss = current_loss

        # cool down
        temp *= alpha
        if temp < final_temp:
            break

        # optional: print progress every 1000 iterations
        if (i + 1) % 1000 == 0:
            print(f"Iter {i+1}, Temp {temp:.4f}, Best Loss {best_loss:.4f}, Δ {best_delta:.4f}, B {best_B:.4f}")

    return best_delta, best_B, best_loss

# -------------------------
# Quick run / test
# -------------------------
if __name__ == "__main__":
    env = OneStepBinomialEnv()
    best_delta, best_B, best_loss = simulated_annealing_hedge(env)
    print("\n--- Simulated Annealing Result ---")
    print(f"Δ = {best_delta:.4f}, B = {best_B:.4f}, implied fair price = {best_delta * env.S0 + best_B:.4f}")
    print(f"L2 replication error = {best_loss:.6f}")
