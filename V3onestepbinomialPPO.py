# FOCUSED ON REPLICATION WITH LEARNING Δ AND B

# import packages
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

# -------------------------
# Environment: 1-step binomial
# -------------------------
class OneStepBinomialEnv:
    def __init__(self,
                 S0=100.0, # initial stock price
                 K=110.0, # strike price of the option
                 up_price=140.0, # upper stock price of binomial model
                 down_price=80.0, # lower stock price of binomial model
                 probability=0.5, # probability of moving in either direction
                 market_option_price=None,   # if provided, used for arbitrage bonus
                 seed=0):
        # converting into float such that Pytorch understands
        self.S0 = float(S0)
        self.K = float(K)
        self.up_price = float(up_price)
        self.down_price = float(down_price)
        self.probability = float(probability)
        self.market_option_price = None if market_option_price is None else float(market_option_price)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        # state contains S0 and time_to_maturity (1). Initializes the state back to reset state,
        # such that the agent can start a new episode with the inital parameters
        self.t = 0
        self.state = np.array([self.S0, 1.0], dtype=np.float32)
        return self.state

    def step(self, action):
        """
        action: [delta, B] where
            delta = hedge ratio
            B = bank account position
        returns: next_state (new environment state after step), reward (scalar),
                 done (boolean indicating episode is over), info (dict with extra info)
        """

        delta, B = float(action[0]), float(action[1])
        
        # stochastic outcome, determines up or down movement
        is_up = self.rng.random() < self.probability
        S_T = self.up_price if is_up else self.down_price

        # standard European call option payoff, profit if stock above strike, 0 otherwise
        payoff = max(S_T - self.K, 0.0)

        # portfolio at maturity
        portfolio = delta * S_T + B

        # replication error
        err = portfolio - payoff

        reward = -abs(err)

        # optional arbitrage bonus:
        arbitrage_bonus = 0.0
        if (self.market_option_price is not None):
            fair_price_estimate = delta * self.S0 + B
            # treat replication as successful if absolute error tiny
            if abs(err) <= 1e-6:
                profit_est = fair_price_estimate - self.market_option_price
                if profit_est > 0:
                    arbitrage_bonus = profit_est
        reward = reward + arbitrage_bonus

        # stock price at maturity + time-to-maturity 0 (terminal)
        next_state = np.array([S_T, 0.0], dtype=np.float32)  
        done = True

        # info dict for debugging
        info = {
            'is_up': is_up,
            'S_T': S_T,
            'payoff': payoff,
            'portfolio': portfolio,
            'err': err,
            'arbitrage_bonus': arbitrage_bonus,
            'fair_price_estimate': (delta * self.S0 + B)
        }
        return next_state, reward, done, info

# -------------------------
# Policy and Value networks
# -------------------------

# neural network class for the policy (the agent’s decision rule for choosing hedge ratio and B)
class PolicyNet(nn.Module):
    # obs_dim: dimension of the observation space (2 for [S_t, time_to_maturity])
    # hidden: number of neurons in the neural network
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        # simple 2-layer feedforward network with tanh activations, tanh makes output between -1 and 1
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # separate heads for hedge ratio Δ and bank account B
        self.mean_delta = nn.Linear(hidden, 1)
        self.mean_B = nn.Linear(hidden, 1)

        # learnable log stds for both outputs
        self.log_std_delta = nn.Parameter(torch.tensor(-0.5))
        self.log_std_B = nn.Parameter(torch.tensor(-0.5))

    # Constructs two Gaussian distributions: one for Δ, one for B
    def forward(self, x):
        h = self.net(x)
        mean_delta = self.mean_delta(h).squeeze(-1)
        mean_B = self.mean_B(h).squeeze(-1)
        std_delta = torch.exp(self.log_std_delta)
        std_B = torch.exp(self.log_std_B)
        return mean_delta, std_delta, mean_B, std_B

    def get_action_and_value(self, x, action=None):
        """PPO-style method that returns [delta, B], log_prob, and entropy"""
        mean_delta, std_delta, mean_B, std_B = self.forward(x)
        dist_delta = torch.distributions.Normal(mean_delta, std_delta)
        dist_B = torch.distributions.Normal(mean_B, std_B)

        if action is None:
            delta = dist_delta.sample()
            B = dist_B.sample()
        else:
            delta, B = action[:, 0], action[:, 1]

        log_prob = dist_delta.log_prob(delta) + dist_B.log_prob(B)
        entropy = dist_delta.entropy() + dist_B.entropy()

        action = torch.stack([delta, B], dim=-1)
        return action, log_prob, entropy


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
# PPO Training routine
# -------------------------
def train_one_step_binomial_ppo(
    episodes=2000,          # Number of interactions with the environment, each episode = one option hedging attempt
    batch_size=64,          # Number of episodes before update
    lr_policy=3e-3,         # Updates how hedge ratio and B are chosen
    lr_value=3e-3,          # Updates how well the critic predicts payoff error
    gamma=1.0,              # Discount factor (how much future rewards count)
    clip_ratio=0.2,         # Controls how much the new policy can deviate from the old policy
    ppo_epochs=10,          
    target_kl=0.02,         
    entropy_coef=0.1,       
    value_coef=1.0,         # Weight for the value loss in the total update (balanced)
    max_grad_norm=1.0,      
    seed=0,
    verbose=True,
    use_market_price=False  # If true - adds an arbitrage incentive (bonus reward if replication is exact and market option price is exploitable)
    # If false - pure replication focus     
):
    # seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # environment setup
    env = OneStepBinomialEnv(S0=100.0, K=110.0, up_price=140.0, down_price=80.0,
                             probability=0.5,
                             market_option_price=(8.0 if use_market_price else None),
                             seed=seed)

    obs_dim = 2  # [S_t, time_to_maturity]

    # Policy network outputs hedge ratio Δ and bank position B
    policy = PolicyNet(obs_dim)

    # Value network estimates expected payoff error.
    value = ValueNet(obs_dim)
    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_value = optim.Adam(value.parameters(), lr=lr_value)

    # storage for batch
    episode_data = []

    # reset environment to inital state, sample hedge ratio and B, store log-prob
    for ep in range(1, episodes + 1):
        s = env.reset()
        s_tensor = torch.tensor(s, dtype=torch.float32)
        
        # Sample action using PPO-style method
        with torch.no_grad():
            action, old_log_prob, _ = policy.get_action_and_value(s_tensor.unsqueeze(0))
            old_value = value(s_tensor)

        action_np = action.squeeze(0).numpy()  # [delta, B]

        # step env
        next_s, reward, done, info = env.step(action_np)

        # store (including old_log_prob and old_value for PPO)
        episode_data.append({
            's': s,
            'a': action_np,
            'old_log_prob': old_log_prob,
            'old_value': old_value,
            'r': reward,
            'next_s': next_s,
            'info': info
        })

        # PPO update every batch_size episodes
        if (ep % batch_size) == 0:
            # build tensors
            states = torch.tensor(np.array([d['s'] for d in episode_data]), dtype=torch.float32)
            actions = torch.tensor(np.array([d['a'] for d in episode_data]), dtype=torch.float32)
            old_log_probs = torch.stack([d['old_log_prob'] for d in episode_data])
            old_values = torch.stack([d['old_value'] for d in episode_data])
            rewards = torch.tensor(np.array([d['r'] for d in episode_data]), dtype=torch.float32)

            # compute returns and advantages (gamma=1, single-step episodes)
            returns = rewards
            advantages = returns - old_values.detach()
            
            # normalize advantages (PPO improvement)
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update loop
            for ppo_epoch in range(ppo_epochs):
                # get new policy outputs
                _, new_log_probs, entropies = policy.get_action_and_value(states, actions)
                new_values = value(states)

                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Encourages policy to move in direction of advantage but prevents giant updates
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # entropy bonus (encourages exploration)
                entropy_loss = -entropy_coef * entropies.mean()

                # total policy loss
                total_policy_loss = policy_loss + entropy_loss

                # value loss (clipped for stability)
                value_pred_clipped = old_values + torch.clamp(
                    new_values - old_values, -clip_ratio, clip_ratio
                )
                value_loss1 = functional.mse_loss(new_values, returns)
                value_loss2 = functional.mse_loss(value_pred_clipped, returns)
                value_loss = value_coef * torch.max(value_loss1, value_loss2)

                # policy update
                opt_policy.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt_policy.step()

                # value update
                opt_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value.parameters(), max_grad_norm)
                opt_value.step()

                # early stopping based on KL divergence
                with torch.no_grad():
                    kl_div = (old_log_probs - new_log_probs).mean().item()
                    if kl_div > target_kl:
                        break

            # Tracks how hedge ratio and bank position evolve over training
            mean_reward = rewards.mean().item()
            mean_delta = actions[:, 0].mean().item()
            std_delta = actions[:, 0].std().item()
            mean_B = actions[:, 1].mean().item()
            std_B = actions[:, 1].std().item()
            mean_abs_advantage = torch.abs(advantages).mean().item()
            fair_price_est = mean_delta * env.S0 + mean_B
            if verbose:
                print(f"ep {ep:5d} | reward {mean_reward:.2f} | Δ {mean_delta:.3f}±{std_delta:.3f} | "
                      f"B {mean_B:.3f}±{std_B:.3f} | price {fair_price_est:.2f} | "
                      f"|adv| {mean_abs_advantage:.2f} | KL {kl_div:.4f} | epochs {ppo_epoch+1}")

            # reset episode_data
            episode_data = []

    # return trained components and environment for inspection
    return policy, value, env

# -------------------------
# Quick run / unit test
# -------------------------
if __name__ == "__main__":
    # train for a small experiment
    pol, val, env = train_one_step_binomial_ppo(
        episodes= 800000,
        batch_size= 128,
        seed=42,
        verbose=True,
        use_market_price=False
    )

    # inspect policy at S0
    s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32)
    mean_delta, std_delta, mean_B, std_B = pol(s0)
    mean_delta = mean_delta.item()
    std_delta = std_delta.item()
    mean_B = mean_B.item()
    std_B = std_B.item()
    fair_price_est = mean_delta * env.S0 + mean_B

    print("\n--- Final policy at S0 ---")
    print(f"Δ (mean) = {mean_delta:.4f}, std = {std_delta:.4f}")
    print(f"B (mean) = {mean_B:.4f}, std = {std_B:.4f}")
    print(f"Implied fair price X = Δ * S0 + B = {fair_price_est:.4f}")

    # run many sims to check replication error
    N = 1000
    errs = []
    for _ in range(N):
        # sample action deterministically as mean (evaluation)
        delta = mean_delta
        B = mean_B
        _, _, _, info = env.step([delta, B])
        errs.append(info['err'])

    errs = np.array(errs)
    print(f"Mean abs replication error over {N} sims: {np.mean(np.abs(errs)):.6f}")
