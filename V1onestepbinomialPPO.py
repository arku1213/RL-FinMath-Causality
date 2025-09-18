# FOCUSED ON REPLICATION

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
        action_delta: hedge ratio
        returns: next_state (new environment state after step), reward (scalar),
                 done (boolean indicating episode is over), info (dict with extra info)
        """

        # float for hedge ratio
        delta = float(action_delta)
        
        # stochastic outcome, determines up or down movement
        is_up = self.rng.random() < self.probability
        S_T = self.up_price if is_up else self.down_price

        # stanadrd European call option payoff, profit if stock above price, 0 otherwise
        payoff = max(S_T - self.K, 0.0)

        # portfolio at maturity
        portfolio = delta * S_T + self.B

        # replication error
        err = portfolio - payoff

        # reward: negative absolute error (want to minimize error)
        reward = -abs(err)  # L1 (Lasso) loss used to reduce extreme penalties

        # optional arbitrage bonus:
        # If agent replicated payoff very closely, and market price < implied fair price,
        # then agent could buy at market price and hedge, realizing (fair_price - market_price).
        arbitrage_bonus = 0.0
        if (self.market_option_price is not None):
            fair_price_estimate = delta * self.S0 + self.B
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
            'fair_price_estimate': (delta * self.S0 + self.B)
        }
        return next_state, reward, done, info

# -------------------------
# Policy and Value networks
# -------------------------

# neural network class for the policy (the agent’s decision rule for choosing hedge ratio
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
        self.mean_head = nn.Linear(hidden, 1)       # output mean of Gaussian (mean hedge ratio))
        # we'll parameterize log_std as a learnable scalar
        # one learnable scalar standard deviation
        self.log_std = nn.Parameter(torch.tensor(-0.5))  # exp(-0.5) ≈ 0.6 std (starting standard deviation)

    # Constructs a Gaussian distribution for the hedge ratio
    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h).squeeze(-1)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action_and_value(self, x, action=None):
        """PPO-style method that returns action, log_prob, and entropy"""
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action) #how likely the chosen hedge ratio was under the current policy. Used in PPO’s update formula
        entropy = dist.entropy()
        
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
# define defaults for PPO hyperparameters
def train_one_step_binomial_ppo(
    episodes=2000,          # Number of interactions with the environment, each episode = one option hedging attempt
    batch_size=64,          # Number of episodes before update
    lr_policy=3e-3,         # Updates how hedge ratio is chosen
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
                             probability=0.5, B= -40.0,
                             market_option_price=(8.0 if use_market_price else None),
                             seed=seed)

    obs_dim = 2  # [S_t, time_to_maturity]

    # Policy network outputs hedge ratio
    policy = PolicyNet(obs_dim)

    # Value network estimates expected payoff error.
    value = ValueNet(obs_dim)
    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_value = optim.Adam(value.parameters(), lr=lr_value)

    # storage for batch
    episode_data = []

    # reset environment to inital state, sample hedge ratio, store log-prob
    for ep in range(1, episodes + 1):
        s = env.reset()
        s_tensor = torch.tensor(s, dtype=torch.float32)
        
        # Sample action using PPO-style method
        with torch.no_grad():
            action, old_log_prob, _ = policy.get_action_and_value(s_tensor)
            old_value = value(s_tensor)
        
        action_item = action.item()

        # step env
        next_s, reward, done, info = env.step(action_item)

        # store (including old_log_prob and old_value for PPO)
        episode_data.append({
            's': s,
            'a': action_item,
            'old_log_prob': old_log_prob,   # store old policy log prob
            'old_value': old_value,         # store old value estimate
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

            # Tracks how hedge ratio and fair price evolve over training, mean reward should go to 0
            mean_reward = rewards.mean().item()
            mean_delta = actions.mean().item()
            std_delta = actions.std().item()
            mean_abs_advantage = torch.abs(advantages).mean().item()
            # implied fair price estimate using average delta 
            fair_price_est = mean_delta * env.S0 + env.B
            if verbose:
                print(f"ep {ep:5d} | reward {mean_reward:.2f} | Δ {mean_delta:.3f}±{std_delta:.3f} | price {fair_price_est:.2f} | |adv| {mean_abs_advantage:.2f} | KL {kl_div:.4f} | epochs {ppo_epoch+1}")

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
        episodes= 400000,
        batch_size= 128,
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

#Mean abs replication error → calculated after training, by simulating many episodes with the final policy.

#Std of hedge ratio → stochastic trained policy, so the hedge ratio will have some variation.