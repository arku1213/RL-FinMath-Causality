Initial challenges included trying to get a hedge ratio close to 0.5 and a fair price close to 10. From the example output:

ep 128 | reward -54.95 | Δ 0.265±0.583 | price 26.50 | |adv| 0.80 | KL 0.6625 | epochs 2
ep 256 | reward -74.81 | Δ -0.387±0.651 | price -38.65 | |adv| 0.74 | KL 0.1464 | epochs 2
ep 384 | reward -56.68 | Δ -0.024±0.634 | price -2.41 | |adv| 0.78 | KL 0.0278 | epochs 2
ep 512 | reward -54.27 | Δ 0.137±0.586 | price 13.65 | |adv| 0.73 | KL -0.0011 | epochs 10
ep 640 | reward -51.43 | Δ 0.111±0.611 | price 11.13 | |adv| 0.80 | KL 0.0039 | epochs 10
ep 768 | reward -48.16 | Δ 0.059±0.553 | price 5.85 | |adv| 0.72 | KL 0.0086 | epochs 10
ep 896 | reward -47.43 | Δ 0.063±0.551 | price 6.28 | |adv| 0.74 | KL 0.0353 | epochs 3
ep 1024 | reward -47.60 | Δ 0.197±0.516 | price 19.74 | |adv| 0.81 | KL -0.0024 | epochs 10
ep 1152 | reward -46.60 | Δ 0.174±0.522 | price 17.35 | |adv| 0.76 | KL 0.0000 | epochs 10
ep 1280 | reward -44.78 | Δ 0.155±0.476 | price 15.52 | |adv| 0.80 | KL -0.0030 | epochs 10
ep 1408 | reward -45.24 | Δ 0.206±0.500 | price 20.56 | |adv| 0.81 | KL 0.0162 | epochs 10
ep 1536 | reward -40.84 | Δ 0.119±0.472 | price 11.90 | |adv| 0.78 | KL 0.0258 | epochs 7
ep 1664 | reward -45.24 | Δ 0.057±0.526 | price 5.65 | |adv| 0.78 | KL 0.0006 | epochs 10
ep 1792 | reward -40.88 | Δ -0.004±0.476 | price -0.45 | |adv| 0.80 | KL 0.0300 | epochs 2
ep 1920 | reward -45.98 | Δ 0.164±0.515 | price 16.41 | |adv| 0.76 | KL 0.0082 | epochs 10
ep 2048 | reward -38.68 | Δ 0.212±0.430 | price 21.21 | |adv| 0.80 | KL 0.0210 | epochs 2
ep 2176 | reward -36.63 | Δ 0.026±0.409 | price 2.63 | |adv| 0.77 | KL 0.0295 | epochs 4
ep 2304 | reward -42.42 | Δ 0.140±0.454 | price 14.05 | |adv| 0.76 | KL 0.0021 | epochs 10
ep 2432 | reward -42.52 | Δ 0.134±0.449 | price 13.44 | |adv| 0.77 | KL 0.0041 | epochs 10
ep 2560 | reward -40.23 | Δ 0.184±0.430 | price 18.43 | |adv| 0.77 | KL 0.0272 | epochs 3
ep 2688 | reward -33.93 | Δ 0.053±0.366 | price 5.27 | |adv| 0.76 | KL 0.0180 | epochs 10
ep 2816 | reward -39.58 | Δ 0.067±0.433 | price 6.75 | |adv| 0.80 | KL 0.0268 | epochs 7
ep 2944 | reward -36.98 | Δ 0.161±0.412 | price 16.13 | |adv| 0.75 | KL 0.0102 | epochs 10
ep 3072 | reward -38.24 | Δ 0.183±0.411 | price 18.35 | |adv| 0.81 | KL 0.0204 | epochs 2
ep 3200 | reward -36.36 | Δ 0.045±0.402 | price 4.54 | |adv| 0.77 | KL 0.0211 | epochs 3
ep 3328 | reward -32.89 | Δ 0.127±0.354 | price 12.71 | |adv| 0.75 | KL 0.0216 | epochs 4
ep 3456 | reward -38.65 | Δ 0.112±0.398 | price 11.23 | |adv| 0.79 | KL 0.0124 | epochs 10
ep 3584 | reward -34.70 | Δ 0.134±0.404 | price 13.40 | |adv| 0.74 | KL 0.0033 | epochs 10
ep 3712 | reward -31.32 | Δ 0.163±0.350 | price 16.26 | |adv| 0.80 | KL 0.0019 | epochs 10
ep 3840 | reward -29.73 | Δ 0.182±0.308 | price 18.17 | |adv| 0.76 | KL 0.0576 | epochs 3
ep 3968 | reward -34.38 | Δ 0.043±0.345 | price 4.26 | |adv| 0.78 | KL 0.0236 | epochs 3
ep 4096 | reward -30.39 | Δ 0.061±0.350 | price 6.06 | |adv| 0.74 | KL 0.0338 | epochs 2
ep 4224 | reward -30.14 | Δ 0.227±0.341 | price 22.68 | |adv| 0.81 | KL 0.0520 | epochs 3
ep 4352 | reward -34.54 | Δ 0.049±0.375 | price 4.86 | |adv| 0.72 | KL 0.0432 | epochs 3
ep 4480 | reward -30.11 | Δ 0.142±0.321 | price 14.22 | |adv| 0.79 | KL -0.0040 | epochs 10
ep 4608 | reward -30.82 | Δ 0.181±0.318 | price 18.09 | |adv| 0.78 | KL -0.0012 | epochs 10
ep 4736 | reward -32.27 | Δ 0.208±0.349 | price 20.81 | |adv| 0.80 | KL 0.0568 | epochs 3
ep 4864 | reward -26.41 | Δ 0.110±0.300 | price 10.99 | |adv| 0.75 | KL 0.0017 | epochs 10
ep 4992 | reward -29.10 | Δ 0.156±0.303 | price 15.63 | |adv| 0.79 | KL -0.0022 | epochs 10
ep 5120 | reward -27.74 | Δ 0.142±0.290 | price 14.21 | |adv| 0.75 | KL -0.0019 | epochs 10
ep 5248 | reward -26.81 | Δ 0.133±0.298 | price 13.31 | |adv| 0.79 | KL -0.0004 | epochs 10
ep 5376 | reward -28.26 | Δ 0.091±0.289 | price 9.08 | |adv| 0.81 | KL 0.0596 | epochs 3
ep 5504 | reward -26.75 | Δ 0.202±0.283 | price 20.19 | |adv| 0.79 | KL 0.0008 | epochs 10
ep 5632 | reward -27.23 | Δ 0.159±0.292 | price 15.89 | |adv| 0.76 | KL -0.0010 | epochs 10
ep 5760 | reward -28.79 | Δ 0.127±0.297 | price 12.71 | |adv| 0.78 | KL 0.0293 | epochs 8
ep 5888 | reward -26.10 | Δ 0.236±0.264 | price 23.56 | |adv| 0.78 | KL 0.0613 | epochs 3
ep 6016 | reward -26.54 | Δ 0.107±0.279 | price 10.66 | |adv| 0.78 | KL 0.0034 | epochs 10
ep 6144 | reward -22.81 | Δ 0.136±0.246 | price 13.56 | |adv| 0.80 | KL -0.0041 | epochs 10
ep 6272 | reward -28.25 | Δ 0.149±0.276 | price 14.90 | |adv| 0.81 | KL 0.0045 | epochs 10
ep 6400 | reward -24.82 | Δ 0.136±0.261 | price 13.62 | |adv| 0.82 | KL 0.0032 | epochs 10
ep 6528 | reward -24.21 | Δ 0.131±0.255 | price 13.06 | |adv| 0.71 | KL -0.0051 | epochs 10
ep 6656 | reward -24.50 | Δ 0.125±0.247 | price 12.45 | |adv| 0.79 | KL 0.0252 | epochs 3
ep 6784 | reward -22.79 | Δ 0.222±0.235 | price 22.21 | |adv| 0.78 | KL 0.0632 | epochs 3
ep 6912 | reward -24.13 | Δ 0.088±0.234 | price 8.83 | |adv| 0.77 | KL 0.0438 | epochs 3
ep 7040 | reward -21.77 | Δ 0.198±0.218 | price 19.79 | |adv| 0.80 | KL 0.0457 | epochs 3
ep 7168 | reward -21.83 | Δ 0.112±0.212 | price 11.18 | |adv| 0.75 | KL 0.0256 | epochs 4
ep 7296 | reward -21.43 | Δ 0.159±0.219 | price 15.90 | |adv| 0.79 | KL 0.0236 | epochs 3
ep 7424 | reward -23.57 | Δ 0.142±0.234 | price 14.21 | |adv| 0.77 | KL -0.0009 | epochs 10
ep 7552 | reward -21.00 | Δ 0.114±0.214 | price 11.40 | |adv| 0.77 | KL 0.0270 | epochs 3
ep 7680 | reward -21.82 | Δ 0.148±0.229 | price 14.80 | |adv| 0.79 | KL -0.0011 | epochs 10
ep 7808 | reward -21.02 | Δ 0.164±0.207 | price 16.37 | |adv| 0.82 | KL 0.0346 | epochs 3
ep 7936 | reward -18.23 | Δ 0.108±0.202 | price 10.76 | |adv| 0.78 | KL 0.0209 | epochs 2
ep 8064 | reward -23.06 | Δ 0.041±0.198 | price 4.15 | |adv| 0.81 | KL 0.0266 | epochs 3
ep 8192 | reward -22.69 | Δ 0.132±0.223 | price 13.24 | |adv| 0.78 | KL -0.0021 | epochs 10
ep 8320 | reward -17.95 | Δ 0.114±0.229 | price 11.38 | |adv| 0.80 | KL 0.0045 | epochs 10
ep 8448 | reward -21.01 | Δ 0.090±0.221 | price 8.96 | |adv| 0.75 | KL 0.0465 | epochs 3
ep 8576 | reward -18.32 | Δ 0.175±0.192 | price 17.48 | |adv| 0.80 | KL 0.0223 | epochs 4
ep 8704 | reward -20.23 | Δ 0.161±0.195 | price 16.09 | |adv| 0.81 | KL 0.0209 | epochs 4
ep 8832 | reward -21.53 | Δ 0.102±0.215 | price 10.19 | |adv| 0.75 | KL 0.0369 | epochs 2
ep 8960 | reward -18.74 | Δ 0.193±0.193 | price 19.27 | |adv| 0.81 | KL 0.0275 | epochs 3
ep 9088 | reward -21.26 | Δ 0.166±0.212 | price 16.60 | |adv| 0.76 | KL 0.0148 | epochs 10
ep 9216 | reward -21.63 | Δ 0.105±0.210 | price 10.46 | |adv| 0.81 | KL 0.0000 | epochs 10
ep 9344 | reward -20.54 | Δ 0.162±0.206 | price 16.18 | |adv| 0.79 | KL 0.0215 | epochs 6
ep 9472 | reward -18.18 | Δ 0.093±0.162 | price 9.29 | |adv| 0.73 | KL 0.0356 | epochs 2
ep 9600 | reward -20.59 | Δ 0.212±0.200 | price 21.24 | |adv| 0.78 | KL 0.0365 | epochs 5
ep 9728 | reward -17.89 | Δ 0.112±0.183 | price 11.16 | |adv| 0.75 | KL 0.0529 | epochs 3
ep 9856 | reward -18.87 | Δ 0.177±0.185 | price 17.66 | |adv| 0.80 | KL 0.0272 | epochs 4
ep 9984 | reward -21.37 | Δ 0.143±0.216 | price 14.29 | |adv| 0.78 | KL 0.0052 | epochs 10

--- Final policy at S0 ---
Δ (mean) = 0.1485, std = 0.1851
Implied fair price X = Δ \* S0 + B = 14.8481
Mean abs replication error over 1000 sims: 10.534896

We can see that the delta hedge is very low at 0.1485, and that the fair price is a bit high at 14.84 (when the number of episodes is 10 000)
