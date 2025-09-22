- When changing the reward from -100 to -20, the reward only stays at -20 (for the most part) - why is this?

- In attempting to find the hedge ratio, B and fair price within the Replication case, we get approximately:

--- Final policy at S0 ---
Δ (mean) = 0.5541, std = 0.1184
B (mean) = -39.7905, std = 13.2875
Implied fair price X = Δ \* S0 + B = 15.6167
Mean abs replication error over 1000 sims: 6.137924

We can see that there are large errors which we want to reduce

- In attempting to find the hedge ratio, B and fair price within the Super Replication case, we get approximately:
  --- Final policy at S0 ---
  Δ (mean) = -1543.1360, std = 592.3601
  B (mean) = -1400.3784, std = 48447.3906
  Implied fair price X = Δ \* S0 + B = -155713.9771

Which is way off

Replication case: After changing to 1.5 million episodes, updating to batch size of 256 and decreasing entropy and policy and values, we get
--- Final policy at S0 ---
Δ (mean) = 0.4960, std = 0.0259
B (mean) = -39.9756, std = 2.3143
Implied fair price X = Δ \* S0 + B = 9.6260
Mean abs replication error over 1000 sims: 0.412213

Replication case: After changing to 2 million episodes, and max number of epochs to 15, we get
--- Final policy at S0 ---
Δ (mean) = 0.4971, std = 0.0130
B (mean) = -40.2649, std = 1.2056
Implied fair price X = Δ \* S0 + B = 9.4444
Mean abs replication error over 1000 sims: 0.578795

Replication case: After applying the previous changes and changing the clip ratio to 0.1, we get
--- Final policy at S0 ---
Δ (mean) = 0.5037, std = 0.0131
B (mean) = -40.1619, std = 1.2907
Implied fair price X = Δ \* S0 + B = 10.2050
Mean abs replication error over 1000 sims: 0.234253

Super-Replication case: Same parameters:
--- Final policy at S0 ---
Δ (mean) = 0.5068, std = 0.0128
B (mean) = -38.9966, std = 1.2753
Implied fair price X = Δ \* S0 + B = 11.6875
Mean abs replication error over 1000 sims: 1.741995

Replication DQN:
--- Final policy at S0 ---
Δ = 0.5000, B = -40.0000
Implied fair price X = Δ \* S0 + B = 10.0000
Mean abs replication error over 1000 sims: 0.000000

DQN gets perfect results for one step because:

- State space is tiny: [S0, t=1].
- Action space is manageable: Δ ∈ [-1,1] in 21 steps, B ∈ [-50,50] in 21 steps → 441 discrete actions.
- finds exact minimizer of squared error.

Super Replication DQN:
--- Final policy at S0 ---
Δ = 0.0000, B = 30.0000
Implied fair price X = Δ \* S0 + B = 30.0000
Mean abs replication error over 1000 sims: 14.400000
