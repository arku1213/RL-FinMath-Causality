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
