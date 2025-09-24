import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# -----------------------------
# N-nomial Option Hedging Environment
# -----------------------------
class NNomialHedgeEnv:
    def __init__(self, S0, possible_prices, payoffs, probabilities, K):
        """
        S0: initial stock price
        possible_prices: list of terminal prices for n-nomial model
        payoffs: option payoff corresponding to each terminal price
        probabilities: risk-neutral probabilities for each terminal price
        K: strike price
        """
        self.S0 = S0
        self.possible_prices = np.array(possible_prices)
        self.payoffs = np.array(payoffs)
        self.probabilities = np.array(probabilities)
        self.K = K
        self.rng = np.random.default_rng()
        self.action_space = None  # continuous 2D: delta, B
        self.observation_space = 1  # S0 only for 1-step
        self.reset()
    
    def reset(self):
        self.state = np.array([self.S0], dtype=np.float32)
        return self.state
    
    def step(self, action):
        delta, B = action
        
        # Sample one realized terminal price
        S_T = self.rng.choice(self.possible_prices, p=self.probabilities)
        payoff = max(S_T - self.K, 0.0)
        portfolio = delta * S_T + B
        
        # Vectorized super-replication check across all states
        shortfalls = np.maximum(0, self.payoffs - (delta * self.possible_prices + B))
        max_shortfall = np.max(shortfalls)
        cost = delta * self.S0 + B
        
        # Reward: strong penalty for any shortfall, else minimize cost
        if max_shortfall > 1e-8:
            reward = -1e6  # catastrophic penalty
        else:
            reward = -cost  # valid hedge → minimize cost
        
        next_state = np.array([S_T], dtype=np.float32)
        done = True
        info = {"shortfalls": shortfalls, "max_shortfall": max_shortfall, "cost": cost}
        return next_state, reward, done, info

# -----------------------------
# DQN Agent
# -----------------------------
class DQNAgent:
    def __init__(self, env, hidden_units=64, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32):
        self.env = env
        self.state_size = env.observation_space
        self.action_size = 2  # delta and B
        self.hidden_units = hidden_units
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=50000)
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(self.hidden_units, activation='relu'))
        model.add(Dense(self.hidden_units, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # outputs delta and B
        model.compile(optimizer=Adam(lr=self.lr), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random delta in [0,1], B in [-S0,S0] (adjust as needed)
            return np.array([np.random.rand(), np.random.uniform(-self.env.S0, self.env.S0)])
        act_values = self.model.predict(state, verbose=0)
        return act_values[0]
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                future = self.model.predict(next_state, verbose=0)[0]
                target += self.gamma * np.max(future)
            target_full = self.model.predict(state, verbose=0)[0]
            target_full[:] = [target, target]  # delta and B updated
            states.append(state[0])
            targets.append(target_full)
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, episodes=1000):
        for e in range(1, episodes+1):
            state = self.env.reset().reshape(1,-1)
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1,-1)
                self.remember(state, action, reward, next_state, done)
                state = next_state
            self.replay()
            if e % 1000 == 0:
                print(f"Episode {e}, epsilon={self.epsilon:.3f}")

# -----------------------------
# Example usage: 1-step trinomial
# -----------------------------
if __name__ == "__main__":
    S0 = 100
    K = 100
    possible_prices = [80, 100, 120]
    payoffs = [max(S-K,0) for S in possible_prices]
    probabilities = [0.25, 0.5, 0.25]  # risk-neutral
    
    env = NNomialHedgeEnv(S0, possible_prices, payoffs, probabilities, K)
    agent = DQNAgent(env)
    
    agent.train(episodes=5000)
    
    # Test learned policy
    state = env.reset().reshape(1,-1)
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    print(f"Learned policy: Δ={action[0]:.3f}, B={action[1]:.3f}")
    print(f"Reward={reward:.2f}, Max Shortfall={info['max_shortfall']:.3f}, Cost={info['cost']:.2f}")
