# FOCUSED ON REPLICATION WITH LEARNING Δ AND B

# import packages
from ast import In
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

# -------------------------
# Environment Setup - includes setting up initial parameters and step function
# Overall: Provides the state of the world, allows the agent to take an action, and then calculates the reward and the next state based on the outcome of that action.
# Task: learn the optimal hedging strategy (the correct Δ and B values) by trying different actions and observing the resulting rewards. The environment's reward function guides the agent toward the theoretical "perfect hedge," where the replication error is minimized.
# -------------------------

# First, a class is made to simulate the environment for the one-step binomial model
# The environment provides the current stock price and time to maturity as state
# The agent takes an action consisting of hedge ratio (Δ) and bank position (B)
# The environment then simulates the stock price movement (up or down) and computes the reward
# The reward is based on the replication error (portfolio - option payoff) and optionally includes an arbitrage bonus
# The episode ends after one step (one-step binomial model)

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

        Δ, B = float(action[0]), float(action[1]) # Takes an action as input (two-element array) and assigns to hedge ratio and bank position, representing a hedging strategy

        is_up = self.rng.random() < self.probability # To determine if stock goes up or down, a random number is generated and compared to the probability of an upward move

        S_T = self.up_price if is_up else self.down_price # The stock price at maturity (time t) is set based on the outcome of the random draw

        payoff = max(S_T - self.K, 0.0) # The option's payoff is the maximum of the final stock price minus the strike price (self.K) or zero. This is a standard formula for a call option.

        portfolio = Δ * S_T + B # The portfolio's value at maturity is calculated as the sum of the value of the stock position (Δ * S_T) and the bank position (B). This differs from the fair price, which is calculated using the intial stock position.

        err = portfolio - payoff # The error is the  difference between the final portfolio value and the option's payoff. The  agent's goal is to find the values for the hedge ratio Δ and bank position B that minimize this error.

        # reward: harsher penalty if under-hedged
        if portfolio >= payoff:
            # If portfolio covers payoff → reward = -|portfolio - payoff|
            reward = -(err**2) # The reward incentivizes the agent to choose an action (Δ and B) that makes the portfolio value at maturity as close as possible to the option's payoff. A smaller error results in a higher (less negative) reward.
        else:
            # If portfolio < payoff → big penalty
            reward = -10 * (err**2)

        # optional arbitrage bonus
        arbitrage_bonus = 0.0
        if (self.market_option_price is not None):
            fair_price_estimate = Δ * self.S0 + B
            if abs(err) <= 1e-6: #If the replication error is tiny, the model considers the replication successful
                profit_est = fair_price_estimate - self.market_option_price #profit by comparing the fair price of the portfolio (fair_price_estimate) to a known market option price
                #encourages the agent to find profitable trading strategies
                if profit_est > 0:
                    arbitrage_bonus = profit_est
        reward = reward + arbitrage_bonus

        next_state = np.array([S_T, 0.0], dtype=np.float32)  # The next state is the final stock price and a time-to-maturity of 0, indicating the end of the simulation.
        done = True # set to True because the simulation represents a single time step to maturity.

        # The info dictionary provides additional information about the step, including whether the stock went up, the final stock price, the option payoff, the portfolio value, the replication error, any arbitrage bonus received, and an estimate of the fair price of the option based on the chosen hedge ratio and bank position. This can be useful for debugging and analysis.
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
# Policy and Value networks

# the policy and value are two fundamental components that work together to solve a problem.
# The policy is the agent's decision-making rule. It's what tells the agent what action to take in a given state.
# The value function estimates how good a certain state or action is. It predicts the expected total future reward from that point.
# In the context of this hedging problem, the policy network (PolicyNetwork) is the core of the agent's strategy. It takes the current state (stock price and time to maturity) and decides on the optimal hedging action, which is a pair of values: delta and B. The goal of training this network is to find the parameters (weights) that consistently produce an action that minimizes the hedging error, leading to a higher reward.
# The value network (ValueNetwork) is used as a baseline to help the agent learn more efficiently by reducing the variance in its policy updates, and assess how good the current state is in terms of expected future rewards.
# -------------------------

# Neural network class for the policy (the agent’s decision rule for choosing hedge ratio and B)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=64): 
        # obs_dim : number of features in the input (S_t, time_to_maturity), so 2
        # hidden : number of neurons in the hidden layer
        super().__init__()
        # Overall, these following lines define functions and layers to transform raw input into a more meaningful representation that captures relevant information
        self.net = nn.Sequential( # nn.Sequential is a container that organizes layers in a sequential order. The data will pass through these layers one after another in the neural network.
            nn.Linear(obs_dim, hidden), # first layer,  fully connected layer that takes the input state of size obs_dim (2) and transforms it into a hidden representation of size hidden.
            nn.Tanh(), # activation function, squashes the output of the linear layer to a range between -1 and 1, allowing the network to model more complex, non-linear relationships.
            nn.Linear(hidden, hidden), # second hidden layer, takes the output from the previous layer (size hidden) and maps it to another hidden representation of the same size.
            nn.Tanh(),
        )
        # taking processed state information and using it to define the policy's output, outputting the parameters for a probability distribution from which an action is sampled

        self.mean_Δ = nn.Linear(hidden, 1) # takes the hidden representation of size 2 and maps it to a single output value. Represents the mean of a Gaussian distribution for the hedge ratio and adjusts for an optimal Δ value.
        self.mean_B = nn.Linear(hidden, 1) # takes the hidden representation of size 2 and maps it to a single output value. Represents the mean of a Gaussian distribution for the bank position and predict the mean value for B that minimizes the replication error

        # By learning the logarithm of the standard deviation, the network ensures that the actual standard deviation is always positive. The initial value of -0.5 is a common starting point that provides a reasonable initial level of exploration.

        # Algorithm will update these parameters to adjust the amount of randomness in the agent's actions.
        # Good strategy -> the standard deviations will likely decrease over time
        # Explore more -> the standard deviations may increase.

        self.log_std_Δ = nn.Parameter(torch.tensor(-0.5)) # learnable parameter that represents the logarithm of the standard deviation
        self.log_std_B = nn.Parameter(torch.tensor(-0.5)) # learnable parameter that represents the logarithm of the standard deviation

    # Tying everything together, this method defines how the input state is processed through the network to produce the parameters of the action distribution. Input state is the current stock price and time to maturity
    def forward(self, x):
        h = self.net(x) # input state (x) is passed through the shared layers to get the hidden representation (h).
        mean_Δ = self.mean_Δ(h).squeeze(-1) # mean for Δ is calculated from h
        mean_B = self.mean_B(h).squeeze(-1) # mean for B is calculated from h.
        std_Δ = torch.exp(self.log_std_Δ) # standard deviation for Δ is computed by exponentiating the learnable log standard deviation parameters.
        std_B = torch.exp(self.log_std_B) # standard deviation for B is computed by exponentiating the learnable log standard deviation parameters.
        return mean_Δ, std_Δ, mean_B, std_B

    # Method for acting in the environment and updating the policy). It samples an action based on the current policy and computes the log-probability and entropy of that action.
    def get_action_and_value(self, x, action=None):
        # two Gaussian distributions and defines a range of possible actions and their probabilities
        mean_Δ, std_Δ, mean_B, std_B = self.forward(x)
        dist_Δ = torch.distributions.Normal(mean_Δ, std_Δ)
        dist_B = torch.distributions.Normal(mean_B, std_B)

        # randomly samples a value for Δ from dist_Δ and a value for B from dist_B, trying out a new action
        if action is None:
            Δ = dist_Δ.sample()
            B = dist_B.sample()
        else:
            Δ, B = action[:, 0], action[:, 1]

        log_prob = dist_Δ.log_prob(Δ) + dist_B.log_prob(B) # logarithm of the probability of the chosen action under the current policy. 
        # higher log_prob - action is more likely to be chosen by the current policy.  
        # final log_prob = sum of individual log probabilities because Δ and B are treated as independent.
        entropy = dist_Δ.entropy() + dist_B.entropy() # measures the randomness or unpredictability of the policy's action distribution. 
        # high entropy - policy is highly exploratory,  
        # low entropy - confident and deterministic. 
        # In PPO, adding an entropy bonus to the reward can encourage the agent to explore more, which helps it avoid getting stuck in suboptimal solutions.

        action = torch.stack([Δ, B], dim=-1) # combines Δ and B into a single, two-element tensor representing the complete action for the agent.
        return action, log_prob, entropy

# The value network's job is to predict the expected return or total future reward from a given state. It doesn't tell the agent what to do, but rather how good a situation it's in. This helps the agent evaluate its actions and make better decisions over time.

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), # first layer,  fully connected layer that takes the input state of size obs_dim (2) and transforms it into a hidden representation of size hidden.
            nn.Tanh(),  # activation function, squashes the output of the linear layer to a range between -1 and 1, allowing the network to model more complex, non-linear relationships.
            nn.Linear(hidden, hidden), # second hidden layer, takes the output from the previous layer (size hidden) and maps it to another hidden representation of the same size.
            nn.Tanh(),  # activation function, squashes the output of the linear layer to a range between -1 and 1, allowing the network to model more complex, non-linear relationships.
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1) # tensor value of the right dimension

# -------------------------
# PPO Training routine

# Policy Training: The policy is trained to choose actions that lead to a high reward. The agent explores the environment, and if a particular action results in a positive reward, the policy network is updated to be more likely to choose that action in the future.

# Value Training: The value network is trained to predict the total reward the agent can expect to receive from a given state. The value network's predictions are compared to the actual rewards received, and its weights are adjusted to make its predictions more accurate.

# These two networks are trained together. The value network's predictions are used to calculate the advantage, which is a key signal that tells the policy network how much better or worse an action was than expected. This makes the policy training more efficient and stable.
# -------------------------

def train_one_step_binomial_ppo(
    episodes=800000,          
    # Number of interactions with the environment, each episode = one option hedging attempt
    # More episodes - more chances to explore, learn from trial and error, more robust and optimal policy,
    # Fewer episodes - not having enough time to converge on an optimal solution.

    batch_size=128,          
    # number of episodes the agent collects before it performs a training update on the neural networks.
    # larger batch size - gradient updates more diverse, stable and reliable requires memory and can slow down the training process.
    # smaller batch size - more frequent, but potentially noisy, updates, unstable

    lr_policy=1e-4,         # Updates how hedge ratio and B are chosen
    lr_value=1e-4,          # Updates how well the critic predicts payoff error
    # These control the step size for the weight updates of the policy and value networks, respectively.
    # higher learning rate - learn faster but instability, can overshoot the optimal solution, leading to a volatile policy that never converges.
    # lower learning rate - more stable but much slower, can get stuck in a suboptimal solution and not have enough "momentum" to improve.

    gamma=1.0,              
    # Discount factor (how much future rewards count)
    # Since episodes are single-step, gamma has no effect here. In multi-step settings, it would determine the importance of future rewards.
    # Currently, the agent cares just as much about future rewards as it does about immediate rewards.
    # In multi-step settings, a higher gamma (close to 1) would make the agent consider long-term rewards more, while a lower gamma (close to 0) would make it focus on immediate rewards.
    
    clip_ratio=0.1,         
    # Controls how much the new policy can deviate from the old one during a single update
    # Larger clip ratio - larger policy updates, faster learning but risk of instability 
    # Smaller clip ratio - restricts updates, making learning more stable but slower. A very small value could prevent meaningful progress.

    ppo_epochs=15,
    # number of times the agent iterates over the same batch of data to perform updates.
    # More epochs - more thorough learning from each batch. if too high, might overfit
    # Fewer epochs - won't fully learn from the data it collected, leading to a less effective update and slower overall training.

    target_kl=0.02,  
    # Kullback-Leibler (KL) divergence between the old and new policies. If the divergence > 0.02  the PPO update loop stops early
    #higher target_kl allows  new policy to deviate more from the old one before stopping. This can speed up learning but risks instability.
    #lower target_kl forces the updates to be very small, making the learning process much more cautious and stable.       

    entropy_coef=0.01,  
    # This coefficient controls the strength of the entropy bonus, which encourages exploration.
    #higher entropy coefficient - force the policy to be more random and exploratory, help escape local optima but might not settle on a confident, deterministic solution.
    #lower entropy coefficient - makes the policy more focused on maximizing immediate reward, leading to less exploration. can lead to suboptimal policy.

    value_coef=1.0,         
    # This weights the value loss in the total loss function.
    #higher value coefficient - focus more on training the value network, better accurately predict rewards, stabilize training but slower policy learning.
    #lower value coefficient -  more emphasis on  policy updates,  speed up learning but  more instability if the value network's predictions are poor.

    max_grad_norm=1.0,   
    #This is used for gradient clipping, which prevents gradients from exploding during backpropagation.
    #higher value - allows for larger gradients, could speed up learning but increases the risk of instability.
    #lower value - more conservative and prevents gradients from getting too large, which can stabilize training, especially for very deep or complex networks.

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
    policy = PolicyNetwork(obs_dim)
    # Value network estimates expected payoff error.
    value = ValueNetwork(obs_dim)


    # initialize two separate optimizers. An optimizer is an algorithm that adjusts the weights of a neural network to minimize a loss function. It's the engine that drives the learning process.
    #Weights are numerical values that represent the strength of the connections between neurons in a neural network.  They are the primary parameters that the network learns from data.
    #The loss function (also called the cost function or objective function) is a mathematical formula that quantifies the difference between the network's predicted output and the actual target output. It essentially tells the network how "wrong" it is.
    #Adam is a popular and efficient optimization algorithm. It adapts the learning rate for each network parameter, which often leads to faster and more stable training.
    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy) #gets all the learnable parameters (weights and biases) of the PolicyNetwork
    opt_value = optim.Adam(value.parameters(), lr=lr_value) #gets all the learnable parameters (weights and biases) of the ValueNetwork

    # storage for batch
    episode_data = []

    # reset environment to inital state, sample hedge ratio and B, store log-prob
    for ep in range(1, episodes + 1): #loop iterates through a specified number of episodes, each one being a single hedging attempt.
        s = env.reset() #reset to its initial state, which includes setting the stock price and time to maturity.
        s_tensor = torch.tensor(s, dtype=torch.float32) # convert from array to tensor (tensors are n-dimensional array of numbers for n >= 3)
        
        # gradients are derivatives and are used to understand how a neural network's loss (error) changes with respect to its weights.
        # gradients are not tracked because only need to use the networks to make predictions
        with torch.no_grad():
            action, old_log_prob, _ = policy.get_action_and_value(s_tensor.unsqueeze(0)) #decision-making step, uses PolicyNetwork to sample an  action (Δ and B).and gets the old_log_prob (probability of this action under the current policy) for later use.
            old_value = value(s_tensor) # takes the same state and provides its estimate of the expected future reward (critic's value)

        action_np = action.squeeze(0).numpy()  #convert from tensor back to array

        # chosen action is executed in the environment
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

        # PPO update, takes batch of previous experience and uses to update the policy and value networks. 
        # goal - improve the agent's strategy by making it more likely to repeat actions that led to good outcomes.
        if (ep % batch_size) == 0: # checks if a full batch of experiences has been collected
            # converted into PyTorch tensors for efficient computation
            states = torch.tensor(np.array([d['s'] for d in episode_data]), dtype=torch.float32)
            actions = torch.tensor(np.array([d['a'] for d in episode_data]), dtype=torch.float32)
            old_log_probs = torch.stack([d['old_log_prob'] for d in episode_data])
            old_values = torch.stack([d['old_value'] for d in episode_data])
            rewards = torch.tensor(np.array([d['r'] for d in episode_data]), dtype=torch.float32)

            # total return from a state is simply the immediate reward (due to single-step episode and gamma = 1))
            # positive advantage - action yielded a better reward than the ValueNetwork predicted, policy should be updated to be more likely to take this action.
            # negative advantage - action was worse than expected, so the policy should be updated to be less likely to take it.
            # The .detach() method is crucial here; it prevents the gradient from flowing back to the ValueNetwork, ensuring that the advantage calculation only uses the old, fixed value predictions and does not affect the value network's training.
            returns = rewards
            advantages = returns - old_values.detach()

            # normalize advantages (normalized, stabilize training by making the advantage values consistent)
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # loop performs multiple passes over the same batch of data. This allows the networks to learn more from each experience while still staying within the bounds set by the PPO algorithm.
            for ppo_epoch in range(ppo_epochs):
                # get new policy outputs
                _, new_log_probs, entropies = policy.get_action_and_value(states, actions)
                new_values = value(states)

                # importance sampling ratio - measures how much the probability of the actions in the batch has changed from the old policy to the new one. A ratio of 1 means no change.
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Encourages policy to move in direction of advantage but prevents giant updates
                surr1 = ratio * advantages # standard policy gradient objective
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages # clipped surrogate objective. It limits the ratio to a small range around 1. The torch.min function then chooses the smaller of the two. This ensures that the policy update is never too large, preventing the new policy from deviating too much from the old one and leading to unstable training.

                # final policy loss, want to maximize the objective, so minimize the negative
                policy_loss = -torch.min(surr1, surr2).mean()

                # entropy bonus. It's added to encourage the policy to be more exploratory (i.e., less deterministic
                entropy_loss = -entropy_coef * entropies.mean()

                # total policy loss
                total_policy_loss = policy_loss + entropy_loss

                # This calculates the loss for the value network. It uses a mean squared error to compare the new value predictions to the actual returns.
                # Clipping is used to prevent large updates to the value function, which can destabilize training.

                value_pred_clipped = old_values + torch.clamp(
                    new_values - old_values, -clip_ratio, clip_ratio
                )
                # The value loss is scaled by value_coef to balance its importance relative to the policy
                value_loss1 = functional.mse_loss(new_values, returns)
                value_loss2 = functional.mse_loss(value_pred_clipped, returns)
                value_loss = value_coef * torch.max(value_loss1, value_loss2)

                # policy update
                opt_policy.zero_grad() # gradients from the previous update are reset to zero.
                total_policy_loss.backward() # gradients for both networks are computed via backpropagation
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm) # gradient clipping step prevents the gradients from becoming too large, which can destabilize training.
                opt_policy.step() # optimizers use the computed gradients to update the weights of the policy network

                # value update
                opt_value.zero_grad() # gradients from the previous update are reset to zero.
                value_loss.backward() # gradients for both networks are computed via backpropagation
                torch.nn.utils.clip_grad_norm_(value.parameters(), max_grad_norm) # gradient clipping step prevents the gradients from becoming too large, which can destabilize training.
                opt_value.step() # optimizers use the computed gradients to update the weights of the value network

                # early stopping based on KL divergence
                with torch.no_grad():
                    kl_div = (old_log_probs - new_log_probs).mean().item() # calculates the KL divergence
                    if kl_div > target_kl: # If the new policy has deviated too much, the loop breaks early
                        break

            # Tracks how hedge ratio and bank position evolve over training
            mean_reward = rewards.mean().item()
            mean_Δ = actions[:, 0].mean().item()
            std_Δ = actions[:, 0].std().item()
            mean_B = actions[:, 1].mean().item()
            std_B = actions[:, 1].std().item()
            mean_abs_advantage = torch.abs(advantages).mean().item()
            fair_price_est = mean_Δ * env.S0 + mean_B
            if verbose:
                print(f"ep {ep:5d} | reward {mean_reward:.2f} | Δ {mean_Δ:.3f}±{std_Δ:.3f} | "
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
        episodes= 2000000,
        batch_size= 256,
        seed=42,
        verbose=True,
        use_market_price=False
    )

    # After training, the code inspects the policy network at the initial state. It passes the initial state to the trained policy network to get the mean and standard deviation of its recommended actions. This shows what the agent has learned to do in its starting position.

    # Fair Price Estimation: It then calculates the implied fair price of the option based on the trained agent's policy. This is the agent's learned best estimate for the option's value at the beginning of the period.

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

    # run many sims to check replication error
    N = 1000 # The code then performs a large number of deterministic simulations (N = 1000) to test the learned policy.
    errs = []
    for _ in range(N):
        # It uses the mean values for Δ and B from the trained policy, ignoring the standard deviation to perform a "greedy" or deterministic action.
        # It runs the environment's step function repeatedly and records the replication error for each simulation.
        Δ = mean_Δ
        B = mean_B
        _, _, _, info = env.step([Δ, B])
        errs.append(info['err'])

    errs = np.array(errs)
    print(f"Mean abs replication error over {N} sims: {np.mean(np.abs(errs)):.6f}")

# --------------------------
#The agent observes the initial financial state.

#It uses its policy network to choose a hedging strategy (Δ and B).

#The environment calculates the result of that strategy (the replication error and reward).

#The agent collects this experience.

#After collecting a batch of experiences, it uses a PPO algorithm to analyze them and update the policy and value networks, learning from its mistakes.

#This loop repeats hundreds of thousands of times until the agent's policy network consistently chooses the optimal Δ and B values that result in a replication error very close to zero.

#Finally, the code tests the trained agent by running a large number of simulations to confirm that the learned strategy is, in fact, an effective and robust hedge.