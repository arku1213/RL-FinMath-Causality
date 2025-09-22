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
