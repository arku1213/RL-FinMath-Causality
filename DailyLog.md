Day 1 - Monday, September 15th:

- Set up problem, filled out paperwork
- Understood and learned more about binomial model framework, hedge ratio, fair price & Black-Scholes model

Day 2 - Tuesday, September 16th:

- Coded RL agent for solving one-period binomical model
- Learned more about RL (agent, action, state, environment, etc.)

Day 3 - Wednesday, Septmber 17th:

- Added comments to understand the code and RL better
- Refined objectives

Day 4 - Thursday, Septmber 18th:

- Updated code to find B (the borrowed amount from the bank) as well
- Further understanding the code and RL better
- New goals to look for to expand search

Day 5 - Friday, September 19th:

- Further understanding the code
- More objectives

Day 6 - Monday, September 22nd:

- Further understanding the code
- Changed -(abs(err)) to -(err^2)
- Looking at other algos (SAC, Deep Learning, Simulated Annealing)

Day 7 - Tuesday, September 23rd:

- Discussed Objectives - decided to look deeply into DQN as PPO is too slow, don't worry about SAC and SA
- Looked into DQN for Trinomial Model (1-step) and n-nomial Model (1-step)

Day 8 - Wednesday, September 24th:

- Reading into DQN
- Making n-nomial model
- DQN is discrete, want continuous - look back into SAC, DDPG, TD3 & MCPG

Day 9 - Thursday, September 25th:

- Focusing on SAC for n-nomial model case
- Determined that one step n-nomial would be tricky
- Moved onto multi-step n-nomial for SAC, TD3, PPO & DDPG -> SAC is the best

Day 10 - Friday, September 26th:

- Adapting multi-step n-nomial code to DDPG, TD3 & PPO
- Made good progress, although models are getting stuck

Day 11 - Monday, September 29th:

- Looked into bandits and discovered that Thompson Sampling is the best bandit approach
- Adapted code using Thompson Sampling for Binomial and Trinomial Case - works well

Day 12 - Tuesday, September 30th:

- Finished Co-op class assignments
- Look into Thompson Sampling and code

Day 13 - Wednesday, October 1st:

- Make Thompson Sampling notes
- Try adapting Thompson Sampling for n-nomial and look into higher steps
- Try other examples for Thompson Sampling
- New objective: Expand into Binary options

Day 14 - Thrusday, October 2nd:

- Trying to optimize Binary option code to get best results
- Trying SAC for Multi-step binary option

Day 15 - Friday, October 3rd:

- SAC didn't work as expected, tried Thompson Sampling
- Trying PPO for Multi-step Binary Options
- Fixed Complete & Incomplete Binary Options One-step cases

Day 16 - Monday, October 6th:

- Refined Complete & Incomplete Binary Options One-step cases
- Looking into Deep RL for Multi-step Binary Options - STUCK

Day 17 - Tuesday, October 7th:

- Trying to implement Dynamic Programming (Hybrid Approach)
- Moving away from Deep RL - looking into ES

Day 18 - Wednesday, October 8th ->
Day 22 - Tuesday, October 14th:

- Implementing Dynamic Programming (Hybrid Approach) for Complete Super-Replication N-Nomial Market.
  Tried:
- Bayesian Optimization (did not work)
- CMA-ES (worked until T=3)
- DQN (worked until T=3)
- Linear Optimization (worked until T=3)
- PPO (worked until T=3)
- Thompson Sampling (worked until T=3)
- DDPG -> WORKED UP UNTIL N=5, T=4 in N-nomial case, just takes a while

Next goals:

- Least Squares Monte Carlo
- Binaries for m scenarios fix

- https://arxiv.org/pdf/2504.05521
