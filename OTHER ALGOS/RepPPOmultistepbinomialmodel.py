# Multi-period PPO for Option Replication
# -------------------------------------------------
# This script extends your original 1-step hedging PPO code to support n-periods.
# If n=1, it reduces exactly to your original setup.
# -------------------------------------------------

# import packages
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

# -------------------------
# Environment Setup - Multi-step Binomial Tree
# -------------------------
class MultiStepBinomialEnv:
    def __init__(self,
                 n=2,                # number of periods
                 S0=100.0,           # initial stock price
                 K=110.0,            # strike price of the option
                 up_price=140.0,     # upper stock price factor
                 down_price=80.0,    # lower stock price factor
                 probability=0.5,    # probability of moving up each step
                 market_option_price=None,   # if provided, used for arbitrage bonus
                 seed=0):
        # Store parameters
        self.n = int(n)
        self.S0 = float(S0)
        self.K = float(K)
        self.up_price = float(up_price)
        self.down_price = float(down_price)
        self.probability = float(probability)
        self.market_option_price = None if market_option_price is None else float(market_option_price)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        """
        Reset environment to initial state.
        Returns: state = [stock price, normalized time-to-maturity]
        """
        self.t = 0
        self.S = self.S0
        self.portfolio = 0.0  # track portfolio value over time
        state = np.array([self.S, 1.0], dtype=np.float32)
        return state

    def step(self, action):
        """
        Step the environment forward by 1 period.
        action: [Δ, B]
        Returns: next_state, reward, done, info
        """
        Δ, B = float(action[0]), float(action[1])

        # simulate stock movement
        is_up = self.rng.random() < self.probability
        self.S = self.up_price if is_up else self.down_price

        # update time
        self.t += 1
        done = (self.t == self.n)

        # if final step, compute payoff and portfolio error
        if done:
            payoff = max(self.S - self.K, 0.0)
            portfolio = Δ * self.S + B
            err = portfolio - payoff
            reward = -(err ** 2)  # L2 error at maturity only

            arbitrage_bonus = 0.0
            if (self.market_option_price is not None):
                fair_price_estimate = Δ * self.S0 + B
                if abs(err) <= 1e-6:
                    profit_est = fair_price_estimate - self.market_option_price
                    if profit_est > 0:
                        arbitrage_bonus = profit_est
            reward += arbitrage_bonus
        else:
            payoff = None
            portfolio = Δ * self.S + B
            err = None
            reward = 0.0  # intermediate steps yield no reward

        # next state encodes current stock price and remaining time fraction
        next_state = np.array([self.S, 1 - self.t / self.n], dtype=np.float32)

        info = {
            'is_up': is_up,
            'S': self.S,
            'payoff': payoff,
            'portfolio': portfolio,
            'err': err,
        }
        return next_state, reward, done, info

# -------------------------
# Policy and Value networks
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_Δ = nn.Linear(hidden, 1)
        self.mean_B = nn.Linear(hidden, 1)
        self.log_std_Δ = nn.Parameter(torch.tensor(-0.5))
        self.log_std_B = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x):
        h = self.net(x)
        mean_Δ = self.mean_Δ(h).squeeze(-1)
        mean_B = self.mean_B(h).squeeze(-1)
        std_Δ = torch.exp(self.log_std_Δ)
        std_B = torch.exp(self.log_std_B)
        return mean_Δ, std_Δ, mean_B, std_B

    def get_action_and_value(self, x, action=None):
        mean_Δ, std_Δ, mean_B, std_B = self.forward(x)
        dist_Δ = torch.distributions.Normal(mean_Δ, std_Δ)
        dist_B = torch.distributions.Normal(mean_B, std_B)
        if action is None:
            Δ = dist_Δ.sample()
            B = dist_B.sample()
        else:
            Δ, B = action[:, 0], action[:, 1]
        log_prob = dist_Δ.log_prob(Δ) + dist_B.log_prob(B)
        entropy = dist_Δ.entropy() + dist_B.entropy()
        action = torch.stack([Δ, B], dim=-1)
        return action, log_prob, entropy

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# -------------------------
# PPO Training routine
# -------------------------
def train_multi_step_ppo(
    n=1,                 # number of periods
    episodes=800000,     # training episodes (will scale with n)
    batch_size=128,
    lr_policy=1e-4,
    lr_value=1e-4,
    gamma=1.0,
    clip_ratio=0.1,
    ppo_epochs=15,
    target_kl=0.02,
    entropy_coef=0.01,
    value_coef=1.0,
    max_grad_norm=1.0,
    seed=0,
    verbose=True,
    use_market_price=False
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # scale episodes by n to maintain training stability
    episodes *= n
    batch_size *= n

    env = MultiStepBinomialEnv(n=n, S0=100.0, K=110.0,
                               up_price=140.0, down_price=80.0,
                               probability=0.5,
                               market_option_price=(8.0 if use_market_price else None),
                               seed=seed)

    obs_dim = 2
    policy = PolicyNetwork(obs_dim)
    value = ValueNetwork(obs_dim)
    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_value = optim.Adam(value.parameters(), lr=lr_value)

    episode_data = []

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        while not done:
            s_tensor = torch.tensor(s, dtype=torch.float32)
            with torch.no_grad():
                action, old_log_prob, _ = policy.get_action_and_value(s_tensor.unsqueeze(0))
                old_value = value(s_tensor)
            action_np = action.squeeze(0).numpy()
            next_s, reward, done, info = env.step(action_np)
            episode_data.append({
                's': s,
                'a': action_np,
                'old_log_prob': old_log_prob,
                'old_value': old_value,
                'r': reward,
                'next_s': next_s,
                'info': info
            })
            s = next_s

        if (ep % batch_size) == 0:
            states = torch.tensor(np.array([d['s'] for d in episode_data]), dtype=torch.float32)
            actions = torch.tensor(np.array([d['a'] for d in episode_data]), dtype=torch.float32)
            old_log_probs = torch.stack([d['old_log_prob'] for d in episode_data])
            old_values = torch.stack([d['old_value'] for d in episode_data])
            rewards = torch.tensor(np.array([d['r'] for d in episode_data]), dtype=torch.float32)

            # multi-step: returns = discounted sum of rewards
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = returns - old_values.detach()
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for ppo_epoch in range(ppo_epochs):
                _, new_log_probs, entropies = policy.get_action_and_value(states, actions)
                new_values = value(states)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy_coef * entropies.mean()
                total_policy_loss = policy_loss + entropy_loss

                value_pred_clipped = old_values + torch.clamp(new_values - old_values, -clip_ratio, clip_ratio)
                value_loss1 = functional.mse_loss(new_values, returns)
                value_loss2 = functional.mse_loss(value_pred_clipped, returns)
                value_loss = value_coef * torch.max(value_loss1, value_loss2)

                opt_policy.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt_policy.step()

                opt_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value.parameters(), max_grad_norm)
                opt_value.step()

                with torch.no_grad():
                    kl_div = (old_log_probs - new_log_probs).mean().item()
                    if kl_div > target_kl:
                        break

            if verbose:
                mean_reward = rewards.mean().item()
                mean_Δ = actions[:, 0].mean().item()
                std_Δ = actions[:, 0].std().item()
                mean_B = actions[:, 1].mean().item()
                std_B = actions[:, 1].std().item()
                mean_abs_advantage = torch.abs(advantages).mean().item()
                fair_price_est = mean_Δ * env.S0 + mean_B
                print(f"ep {ep:5d} | reward {mean_reward:.2f} | Δ {mean_Δ:.3f}±{std_Δ:.3f} | "
                      f"B {mean_B:.3f}±{std_B:.3f} | price {fair_price_est:.2f} | "
                      f"|adv| {mean_abs_advantage:.2f} | KL {kl_div:.4f} | epochs {ppo_epoch+1}")

            episode_data = []

    return policy, value, env

# -------------------------
# Quick run / unit test
# -------------------------
if __name__ == "__main__":
    pol, val, env = train_multi_step_ppo(n=2, episodes=200000, batch_size=256, seed=42, verbose=True)

    s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32)
    mean_Δ, std_Δ, mean_B, std_B = pol(s0)
    mean_Δ = mean_Δ.item()
    std_Δ = std_Δ.item()
    mean_B = mean_B.item()
    std_B = std_B.item()
    fair_price_est = mean_Δ * env.S0 + mean_B

    print("\n--- Final policy at S0 ---")
    print(f"Δ (mean) = {mean_Δ:.4f}, std = {std_Δ:.4f}")
    print(f"B (mean) = {mean_B:.4f}, std = {std_B:.4f}")
    print(f"Implied fair price X = Δ * S0 + B = {fair_price_est:.4f}")
