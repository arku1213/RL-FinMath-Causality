################################################################################################################################

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

################################################################################################################################
