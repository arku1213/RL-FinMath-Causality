# import packages
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------
# Environment: 1-step binomial
# -------------------------
class OneStepBinomialEnv:
    def __init__(self,
                 S0=100.0, #initial stock price
                 K=110.0, # strike price of the option
                 up_price=140.0, # upper stock price of binomial model
                 down_price=80.0, # lower stock price of binomial model
                 probability=0.5, # probability of moving in either direction
                 B= -40.0, # borrowed money from bank
                 market_option_price=None,   # if provided, used for arbitrage bonus
                 seed=0):
        # converting into float such that Pytorch understands
        self.S0 = float(S0)
        self.K = float(K)
        self.up_price = float(up_price)
        self.down_price = float(down_price)
        self.probability = float(probability)
        self.B = float(B)
        self.market_option_price = None if market_option_price is None else float(market_option_price)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        # state contains S0 and time_to_maturity (1). Initializes the state back to reset state,
        # such that the agent can start a new episode with the inital parameters
        self.t = 0
        self.state = np.array([self.S0, 1.0], dtype=np.float32)
        return self.state

    def step(self, action_delta):
        """
        action_delta: float (hedge ratio chosen by agent)
        returns: next_state, reward, done, info
        """
        delta = float(action_delta)
        # sample outcome
        is_up = self.rng.random() < self.probability
        S_T = self.up_price if is_up else self.down_price

        # option payoff (call)
        payoff = max(S_T - self.K, 0.0)

        # portfolio at maturity (we assume B fixed and no interest)
        portfolio = delta * S_T + self.B

        # replication error
        err = portfolio - payoff
        # terminal reward: negative squared error (we want to minimize it)
        reward = - (err ** 2)

        # optional arbitrage bonus:
        # If agent replicated payoff very closely, and market price < implied fair price,
        # then agent could buy at market price and hedge, realizing (fair_price - market_price).
        # This is a simplified proxy for realized arbitrage profit in the 1-step perfect replication case.
        arbitrage_bonus = 0.0
        if (self.market_option_price is not None):
            fair_price_estimate = delta * self.S0 + self.B
            # treat replication as successful if absolute error tiny
            if abs(err) <= 1e-6:
                profit_est = fair_price_estimate - self.market_option_price
                if profit_est > 0:
                    arbitrage_bonus = profit_est
        reward = reward + arbitrage_bonus

        next_state = np.array([S_T, 0.0], dtype=np.float32)  # terminal
        done = True
        info = {
            'is_up': is_up,
            'S_T': S_T,
            'payoff': payoff,
            'portfolio': portfolio,
            'err': err,
            'arbitrage_bonus': arbitrage_bonus,
            'fair_price_estimate': (delta * self.S0 + self.B)
        }
        return next_state, reward, done, info

# -------------------------
# Policy and Value networks
# -------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 1)       # output mean of Gaussian
        # we'll parameterize log_std as a learnable scalar
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h).squeeze(-1)
        std = torch.exp(self.log_std)
        return mean, std


class ValueNet(nn.Module):
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
# Training routine
# -------------------------
def train_one_step_binomial(
    episodes=2000,
    batch_size=512,
    lr_policy=1e-4,
    lr_value=5e-4,
    gamma=1.0,
    seed=0,
    verbose=True,
    use_market_price=False
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = OneStepBinomialEnv(S0=100.0, K=110.0, up_price=140.0, down_price=80.0,
                             probability=0.5, B=-40.0,
                             market_option_price=(8.0 if use_market_price else None),
                             seed=seed)

    obs_dim = 2  # [S_t, time_to_maturity]
    policy = PolicyNet(obs_dim)
    value = ValueNet(obs_dim)
    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_value = optim.Adam(value.parameters(), lr=lr_value)

    # storage for batch
    episode_data = []

    for ep in range(1, episodes + 1):
        s = env.reset()
        s_tensor = torch.tensor(s, dtype=torch.float32)
        mean, std = policy(s_tensor)
        # sample action (continuous)
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()
        logp = dist.log_prob(a)
        action = a.item()

        # step env
        next_s, reward, done, info = env.step(action)

        # store
        episode_data.append({
            's': s,
            'a': action,
            'logp': logp,   # keep gradient
            'r': reward,
            'next_s': next_s,
            'info': info
        })

        # update every batch_size episodes
        if (ep % batch_size) == 0:
            # build tensors
            states = torch.tensor(np.array([d['s'] for d in episode_data]), dtype=torch.float32)
            actions = torch.tensor(np.array([d['a'] for d in episode_data]), dtype=torch.float32)
            logps = torch.stack([d['logp'] for d in episode_data])
            rewards = torch.tensor(np.array([d['r'] for d in episode_data]), dtype=torch.float32)

            # compute returns (gamma=1, single-step episodes)
            returns = rewards

            # baseline (value) predictions
            values = value(states)

            # advantage for policy gradient (detach to avoid backprop through value network)
            advantage = (returns - values).detach()

            # policy loss
            policy_loss = -(logps * advantage).mean()
            opt_policy.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            opt_policy.step()

            # value loss
            value_loss = F.mse_loss(values, returns)
            opt_value.zero_grad()
            value_loss.backward()
            opt_value.step()

            # logging
            mean_reward = rewards.mean().item()
            mean_delta = actions.mean().item()
            # implied fair price estimate using average delta (B=0)
            fair_price_est = mean_delta * env.S0 + env.B
            if verbose:
                print(f"ep {ep:5d} | batch mean reward {mean_reward:.4f} | mean Δ {mean_delta:.4f} | fair_price ≈ {fair_price_est:.4f}")

            # reset episode_data
            episode_data = []

    # return trained components and environment for inspection
    return policy, value, env

# -------------------------
# Quick run / unit test
# -------------------------
if __name__ == "__main__":
    # train for a small experiment
    pol, val, env = train_one_step_binomial(
        episodes=10000,
        batch_size=128,
        seed=42,
        verbose=True,
        use_market_price=False
    )

    # inspect policy at S0
    s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32)
    mean, std = pol(s0)
    mean = mean.item()
    std = std.item()
    fair_price_est = mean * env.S0 + env.B

    print("\n--- Final policy at S0 ---")
    print(f"Δ (mean) = {mean:.4f}, std = {std:.4f}")
    print(f"Implied fair price X = Δ * S0 + B = {fair_price_est:.4f}")

    # run many sims to check replication error
    N = 1000
    errs = []
    deltas = []
    for _ in range(N):
        # sample action deterministically as mean (evaluation)
        delta = mean
        _, _, _, info = env.step(delta)
        errs.append(info['err'])
        deltas.append(delta)

    errs = np.array(errs)
    print(f"Mean abs replication error over {N} sims: {np.mean(np.abs(errs)):.6f}")