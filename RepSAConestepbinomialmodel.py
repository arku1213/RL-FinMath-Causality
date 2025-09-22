# SAC for the one-step binomial hedging problem
# Textbook SAC: actor (stochastic Gaussian policy), two Q-critics, target critics, replay buffer,
# automatic temperature (alpha) tuning, and off-policy updates using mini-batches.
#
# The environment is the same one-step binomial you provided: agent chooses (Δ, B) each episode,
# env returns reward based on replication error. We will reuse that environment verbatim.

import math
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

# -------------------------
# Environment (copy of your OneStepBinomialEnv)
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
    
    # This function resets the environment to the initial state
    def reset(self):
        self.t = 0 # initializes time to 0
        self.state = np.array([self.S0, 1.0], dtype=np.float32) # state = [stock price, time to maturity]
        return self.state # returns the initial state

    # This function takes an action (Δ, B) and returns the next state, reward, done flag, and info dict
    def step(self, action):
        Δ, B = float(action[0]), float(action[1]) # action = [delta, B]

        is_up = self.rng.random() < self.probability
        S_T = self.up_price if is_up else self.down_price
        payoff = max(S_T - self.K, 0.0)
        portfolio = Δ * S_T + B
        err = portfolio - payoff

        # replication L2 reward (you can change to L1 or superreplication later)
        reward = -(err ** 2)

        # optional arbitrage bonus
        arbitrage_bonus = 0.0
        if (self.market_option_price is not None):
            fair_price_estimate = Δ * self.S0 + B
            if abs(err) <= 1e-6:
                profit_est = fair_price_estimate - self.market_option_price
                if profit_est > 0:
                    arbitrage_bonus = profit_est
        reward = reward + arbitrage_bonus

        next_state = np.array([S_T, 0.0], dtype=np.float32)
        done = True
        info = {
            'is_up': is_up,
            'S_T': S_T,
            'payoff': payoff,
            'portfolio': portfolio,
            'err': err,
            'arbitrage_bonus': arbitrage_bonus,
            'fair_price_estimate': (Δ * self.S0 + B)
        }
        return next_state, reward, done, info

# -------------------------
# Replay buffer for SAC
# -------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # convert to tensors
        states = torch.tensor(np.array([b.state for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([b.action for b in batch]), dtype=torch.float32)
        rewards = torch.tensor(np.array([b.reward for b in batch]), dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(np.array([b.next_state for b in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([b.done for b in batch]), dtype=torch.float32).unsqueeze(-1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Neural network building blocks
# -------------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# -------------------------
# Critic (Q-network): takes state and action and returns scalar Q(s,a)
# We'll use two independent critics (Q1, Q2)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        # input: [state, action] concatenated
        self.q = mlp([obs_dim + act_dim, hidden, hidden, 1], activation=nn.ReLU, output_activation=nn.Identity)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q(x).squeeze(-1)  # returns shape (batch,)

# -------------------------
# Gaussian policy (actor) with reparameterization trick
# Output mean and log_std for each action dimension. We will NOT tanh-squash actions
# (keeping raw Gaussian outputs) to keep things straightforward for the hedging problem.
# If you later want bounds on Δ or B, you can apply a tanh+rescale with Jacobian correction.
# -------------------------
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden], activation=nn.ReLU, output_activation=nn.ReLU)
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, s):
        h = self.net(s)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std, log_std

    def sample(self, s):
        mean, std, log_std = self.forward(s)
        # reparameterization trick: action = mean + std * eps, eps ~ N(0,1)
        eps = torch.randn_like(mean)
        action = mean + std * eps
        # log prob of action under Gaussian (independent dims)
        var = std.pow(2)
        log_prob = -0.5 * (((action - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # sum over action dims
        return action, log_prob, mean, log_std

    def deterministic(self, s):
        mean, _, _ = self.forward(s)
        return mean

# -------------------------
# SAC Agent and training loop
# -------------------------
def train_sac_one_step(
    episodes=200000,
    env_seed=0,
    batch_size=256,
    replay_start_size=1000,
    lr=3e-4,
    gamma=0.99,            # discount factor (single-step so not important but kept)
    tau=5e-3,              # target network soft update rate
    policy_update_delay=1, # update policy every N critic updates (usually 1)
    target_entropy=None,   # if None, set to -action_dim
    device='cpu',
    print_every=1000,
    eval_every=20000
):
    torch.manual_seed(env_seed)
    np.random.seed(env_seed)
    random.seed(env_seed)

    env = OneStepBinomialEnv(seed=env_seed)
    obs_dim = 2
    act_dim = 2  # [Δ, B]

    # networks
    q1 = QNetwork(obs_dim, act_dim).to(device)
    q2 = QNetwork(obs_dim, act_dim).to(device)
    q1_target = QNetwork(obs_dim, act_dim).to(device)
    q2_target = QNetwork(obs_dim, act_dim).to(device)
    policy = GaussianPolicy(obs_dim, act_dim).to(device)

    # copy weights to targets
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # optimizers
    q_params = list(q1.parameters()) + list(q2.parameters())
    opt_q = optim.Adam(q_params, lr=lr)
    opt_policy = optim.Adam(policy.parameters(), lr=lr)

    # automatic entropy tuning
    if target_entropy is None:
        target_entropy = -float(act_dim)
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=lr)

    replay = ReplayBuffer(capacity=int(1e6))

    total_steps = 0
    updates = 0
    start_time = time.time()

    # helper: soft update target networks
    def soft_update(source, target, tau):
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

    # training loop
    for ep in range(1, episodes + 1):
        s = env.reset()
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

        # sample a single action from current policy (on-policy sample to fill buffer)
        with torch.no_grad():
            a, logp_a, _, _ = policy.sample(s_tensor)
        a_np = a.squeeze(0).cpu().numpy()
        next_s, r, done, info = env.step(a_np)

        # push to replay
        replay.push(s, a_np, float(r), next_s, float(done))
        total_steps += 1

        # training starts only after replay_start_size
        if len(replay) >= replay_start_size:
            # perform multiple gradient updates per environment step (commonly 1)
            for _ in range(1):
                # sample minibatch
                states, actions, rewards, next_states, dones = replay.sample(batch_size)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                # --- compute target Q value ---
                with torch.no_grad():
                    # sample next action from policy (for next_states)
                    next_action, next_logp, _, _ = policy.sample(next_states)
                    # target Q = r + gamma * (min(Q1_target, Q2_target) - alpha * logp_next)
                    q1_next = q1_target(next_states, next_action).unsqueeze(-1)
                    q2_next = q2_target(next_states, next_action).unsqueeze(-1)
                    q_next_min = torch.min(q1_next, q2_next)
                    alpha = torch.exp(log_alpha)
                    q_target = rewards + (1.0 - dones) * gamma * (q_next_min - alpha * next_logp)

                # current Q estimates
                q1_pred = q1(states, actions).unsqueeze(-1)
                q2_pred = q2(states, actions).unsqueeze(-1)

                # Q losses (MSE)
                q1_loss = F.mse_loss(q1_pred, q_target)
                q2_loss = F.mse_loss(q2_pred, q_target)
                q_loss = q1_loss + q2_loss

                opt_q.zero_grad()
                q_loss.backward()
                opt_q.step()
                updates += 1

                # --- policy update (delayed optionally) ---
                if updates % policy_update_delay == 0:
                    # sample actions from current policy
                    new_action, logp_new, _, _ = policy.sample(states)
                    # policy objective: minimize alpha * logp_new - Q (maximizes Q + entropy)
                    q1_new = q1(states, new_action).unsqueeze(-1)
                    q2_new = q2(states, new_action).unsqueeze(-1)
                    q_new_min = torch.min(q1_new, q2_new)
                    alpha = torch.exp(log_alpha)
                    policy_loss = (alpha * logp_new - q_new_min).mean()

                    opt_policy.zero_grad()
                    policy_loss.backward()
                    opt_policy.step()

                    # --- alpha (entropy temperature) update ---
                    alpha_loss = -(log_alpha * (logp_new + target_entropy).detach()).mean()
                    opt_alpha.zero_grad()
                    alpha_loss.backward()
                    opt_alpha.step()

                    # soft-update targets
                    soft_update(q1, q1_target, tau)
                    soft_update(q2, q2_target, tau)

        # occasional logging / evaluation
        if ep % print_every == 0:
            elapsed = time.time() - start_time
            # evaluate deterministic policy at S0
            with torch.no_grad():
                s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32).unsqueeze(0).to(device)
                mean_action = policy.deterministic(s0).squeeze(0).cpu().numpy()
                delta_mean = mean_action[0]
                B_mean = mean_action[1]
                fair_price_est = delta_mean * env.S0 + B_mean

            # compute current replay length and estimated alpha
            alpha_val = float(torch.exp(log_alpha).item())
            print(f"ep {ep:6d} | steps {total_steps:7d} | replay {len(replay):6d} | "
                  f"Δ {delta_mean:.4f} B {B_mean:.4f} | price {fair_price_est:.4f} | alpha {alpha_val:.4f} | time {elapsed:.1f}s")

        # optional full evaluation (compute mean abs replication error over many sims)
        if ep % eval_every == 0:
            with torch.no_grad():
                s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32).unsqueeze(0).to(device)
                mean_action = policy.deterministic(s0).squeeze(0).cpu().numpy()
                delta_mean = mean_action[0]
                B_mean = mean_action[1]
                N = 1000
                errs = []
                for _ in range(N):
                    _, _, _, info = env.step([delta_mean, B_mean])
                    errs.append(info['err'])
                errs = np.array(errs)
                print(f"  -> eval: mean abs replication error over {N} sims: {np.mean(np.abs(errs)):.6f}")

    # return trained components and environment for inspection
    return policy, q1, q2, env

# -------------------------
# Quick run / example usage
# -------------------------
if __name__ == "__main__":
    # WARNING: training can be slow; reduce episodes for quick tests
    policy, q1, q2, env = train_sac_one_step(
        episodes=200000,       # you can increase to match your PPO experiments
        env_seed=42,
        batch_size=256,
        replay_start_size=1000,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        policy_update_delay=1,
        target_entropy=None,
        device='cpu',
        print_every=2000,
        eval_every=20000
    )

    # final deterministic policy evaluation
    s0 = torch.tensor([env.S0, 1.0], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean_action = policy.deterministic(s0).squeeze(0).cpu().numpy()
    delta_mean = mean_action[0]
    B_mean = mean_action[1]
    fair_price_est = delta_mean * env.S0 + B_mean
    print("\n--- Final policy at S0 ---")
    print(f"Δ (mean) = {delta_mean:.4f}")
    print(f"B (mean) = {B_mean:.4f}")
    print(f"Implied fair price X = Δ * S0 + B = {fair_price_est:.4f}")

    # test many sims
    N = 1000
    errs = []
    for _ in range(N):
        _, _, _, info = env.step([delta_mean, B_mean])
        errs.append(info['err'])
    errs = np.array(errs)
    print(f"Mean abs replication error over {N} sims: {np.mean(np.abs(errs)):.6f}")
