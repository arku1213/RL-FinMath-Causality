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

Fixed:
| epochs 10
ep 2048 | reward -37.17 | Δ 0.583±0.438 | price 18.28 | |adv| 0.78 | KL 0.0637 | epochs 3
ep 2176 | reward -36.93 | Δ 0.379±0.415 | price -2.07 | |adv| 0.76 | KL 0.0135 | epochs 10
ep 2304 | reward -42.10 | Δ 0.434±0.454 | price 3.44 | |adv| 0.76 | KL 0.0018 | epochs 10
ep 2432 | reward -42.33 | Δ 0.430±0.453 | price 3.03 | |adv| 0.77 | KL 0.0030 | epochs 10
ep 2560 | reward -39.16 | Δ 0.459±0.439 | price 5.88 | |adv| 0.75 | KL 0.0065 | epochs 10
ep 2688 | reward -32.78 | Δ 0.446±0.369 | price 4.61 | |adv| 0.75 | KL -0.0004 | epochs 10
ep 2816 | reward -39.29 | Δ 0.419±0.437 | price 1.88 | |adv| 0.83 | KL 0.0265 | epochs 5
ep 2944 | reward -36.94 | Δ 0.512±0.417 | price 11.18 | |adv| 0.75 | KL 0.0112 | epochs 10
ep 3072 | reward -38.47 | Δ 0.576±0.414 | price 17.58 | |adv| 0.81 | KL 0.0293 | epochs 4
ep 3200 | reward -36.01 | Δ 0.429±0.403 | price 2.93 | |adv| 0.77 | KL 0.0032 | epochs 10
ep 3328 | reward -30.92 | Δ 0.464±0.348 | price 6.35 | |adv| 0.75 | KL 0.0323 | epochs 4
ep 3456 | reward -36.48 | Δ 0.491±0.390 | price 9.14 | |adv| 0.79 | KL 0.0031 | epochs 10
ep 3584 | reward -33.18 | Δ 0.519±0.394 | price 11.89 | |adv| 0.75 | KL 0.0065 | epochs 10
ep 3712 | reward -28.62 | Δ 0.525±0.337 | price 12.51 | |adv| 0.78 | KL -0.0013 | epochs 10
ep 3840 | reward -25.99 | Δ 0.571±0.295 | price 17.13 | |adv| 0.76 | KL 0.0357 | epochs 3
ep 3968 | reward -30.24 | Δ 0.460±0.330 | price 6.01 | |adv| 0.75 | KL 0.0007 | epochs 10
ep 4096 | reward -30.24 | Δ 0.363±0.328 | price -3.74 | |adv| 0.75 | KL 0.0345 | epochs 4
ep 4224 | reward -26.87 | Δ 0.569±0.318 | price 16.94 | |adv| 0.81 | KL 0.0230 | epochs 5
ep 4352 | reward -30.37 | Δ 0.430±0.348 | price 3.05 | |adv| 0.73 | KL 0.0255 | epochs 3
ep 4480 | reward -27.29 | Δ 0.493±0.298 | price 9.34 | |adv| 0.80 | KL -0.0031 | epochs 10
ep 4608 | reward -26.72 | Δ 0.534±0.294 | price 13.44 | |adv| 0.77 | KL -0.0021 | epochs 10
ep 4736 | reward -27.97 | Δ 0.547±0.321 | price 14.72 | |adv| 0.78 | KL 0.0226 | epochs 2
ep 4864 | reward -24.00 | Δ 0.494±0.277 | price 9.40 | |adv| 0.79 | KL -0.0049 | epochs 10
ep 4992 | reward -25.09 | Δ 0.451±0.277 | price 5.05 | |adv| 0.77 | KL 0.0253 | epochs 3
ep 5120 | reward -23.40 | Δ 0.517±0.269 | price 11.73 | |adv| 0.76 | KL -0.0025 | epochs 10
ep 5248 | reward -23.56 | Δ 0.500±0.276 | price 9.99 | |adv| 0.75 | KL 0.0000 | epochs 10
ep 5376 | reward -23.90 | Δ 0.496±0.267 | price 9.61 | |adv| 0.80 | KL 0.0226 | epochs 2
ep 5504 | reward -23.44 | Δ 0.559±0.261 | price 15.91 | |adv| 0.77 | KL 0.0275 | epochs 4
ep 5632 | reward -24.20 | Δ 0.486±0.271 | price 8.58 | |adv| 0.77 | KL -0.0015 | epochs 10
ep 5760 | reward -23.62 | Δ 0.478±0.277 | price 7.80 | |adv| 0.82 | KL 0.0253 | epochs 2
ep 5888 | reward -24.16 | Δ 0.587±0.250 | price 18.68 | |adv| 0.75 | KL 0.0449 | epochs 3
ep 6016 | reward -23.27 | Δ 0.481±0.264 | price 8.08 | |adv| 0.76 | KL 0.0085 | epochs 10
ep 6144 | reward -20.34 | Δ 0.515±0.235 | price 11.54 | |adv| 0.78 | KL 0.0225 | epochs 3
ep 6272 | reward -25.03 | Δ 0.493±0.270 | price 9.32 | |adv| 0.79 | KL 0.0038 | epochs 10
ep 6400 | reward -22.59 | Δ 0.491±0.257 | price 9.08 | |adv| 0.80 | KL 0.0078 | epochs 10
ep 6528 | reward -20.64 | Δ 0.503±0.250 | price 10.27 | |adv| 0.72 | KL -0.0040 | epochs 10
ep 6656 | reward -20.64 | Δ 0.454±0.241 | price 5.37 | |adv| 0.78 | KL 0.0259 | epochs 5
ep 6784 | reward -20.12 | Δ 0.531±0.230 | price 13.13 | |adv| 0.75 | KL 0.0306 | epochs 2
ep 6912 | reward -21.22 | Δ 0.427±0.229 | price 2.67 | |adv| 0.77 | KL 0.0583 | epochs 3
ep 7040 | reward -18.28 | Δ 0.546±0.213 | price 14.63 | |adv| 0.78 | KL 0.0447 | epochs 3
ep 7168 | reward -18.87 | Δ 0.465±0.208 | price 6.47 | |adv| 0.81 | KL 0.0282 | epochs 3
ep 7296 | reward -18.83 | Δ 0.537±0.215 | price 13.67 | |adv| 0.75 | KL 0.0283 | epochs 3
ep 7424 | reward -20.83 | Δ 0.496±0.229 | price 9.62 | |adv| 0.74 | KL 0.0053 | epochs 10
ep 7552 | reward -19.48 | Δ 0.468±0.210 | price 6.77 | |adv| 0.81 | KL -0.0013 | epochs 10
ep 7680 | reward -19.52 | Δ 0.449±0.221 | price 4.92 | |adv| 0.77 | KL 0.0704 | epochs 3
ep 7808 | reward -18.07 | Δ 0.545±0.203 | price 14.47 | |adv| 0.76 | KL 0.0217 | epochs 7
ep 7936 | reward -17.29 | Δ 0.513±0.197 | price 11.32 | |adv| 0.75 | KL 0.0543 | epochs 3
ep 8064 | reward -19.26 | Δ 0.419±0.194 | price 1.89 | |adv| 0.79 | KL 0.0789 | epochs 4
ep 8192 | reward -18.85 | Δ 0.542±0.218 | price 14.22 | |adv| 0.82 | KL 0.0222 | epochs 3
ep 8320 | reward -17.66 | Δ 0.495±0.226 | price 9.48 | |adv| 0.79 | KL 0.0063 | epochs 10
ep 8448 | reward -19.43 | Δ 0.494±0.218 | price 9.37 | |adv| 0.81 | KL 0.0051 | epochs 10
ep 8576 | reward -16.49 | Δ 0.497±0.185 | price 9.65 | |adv| 0.77 | KL 0.0263 | epochs 2
ep 8704 | reward -18.24 | Δ 0.452±0.188 | price 5.24 | |adv| 0.78 | KL 0.0067 | epochs 10
ep 8832 | reward -18.68 | Δ 0.443±0.204 | price 4.33 | |adv| 0.73 | KL 0.0441 | epochs 3
ep 8960 | reward -14.83 | Δ 0.528±0.182 | price 12.80 | |adv| 0.82 | KL 0.0432 | epochs 3
ep 9088 | reward -18.29 | Δ 0.488±0.200 | price 8.79 | |adv| 0.75 | KL 0.0080 | epochs 10
ep 9216 | reward -18.18 | Δ 0.450±0.199 | price 5.04 | |adv| 0.78 | KL 0.0374 | epochs 2
ep 9344 | reward -18.20 | Δ 0.535±0.199 | price 13.53 | |adv| 0.81 | KL 0.0129 | epochs 10
ep 9472 | reward -14.30 | Δ 0.482±0.156 | price 8.24 | |adv| 0.82 | KL 0.0415 | epochs 2
ep 9600 | reward -18.53 | Δ 0.567±0.192 | price 16.71 | |adv| 0.79 | KL 0.0310 | epochs 3
ep 9728 | reward -16.22 | Δ 0.478±0.176 | price 7.85 | |adv| 0.76 | KL 0.0526 | epochs 3
ep 9856 | reward -15.49 | Δ 0.540±0.178 | price 13.98 | |adv| 0.77 | KL 0.0216 | epochs 4
ep 9984 | reward -17.97 | Δ 0.511±0.207 | price 11.13 | |adv| 0.77 | KL 0.0079 | epochs 10
ep 10112 | reward -16.11 | Δ 0.518±0.185 | price 11.79 | |adv| 0.80 | KL 0.0127 | epochs 10
ep 10240 | reward -13.63 | Δ 0.520±0.149 | price 11.95 | |adv| 0.82 | KL -0.0027 | epochs 10
ep 10368 | reward -15.71 | Δ 0.466±0.171 | price 6.61 | |adv| 0.77 | KL 0.0215 | epochs 3
ep 10496 | reward -14.59 | Δ 0.541±0.157 | price 14.09 | |adv| 0.78 | KL 0.0251 | epochs 2
ep 10624 | reward -14.33 | Δ 0.443±0.152 | price 4.29 | |adv| 0.81 | KL 0.0766 | epochs 5
ep 10752 | reward -14.89 | Δ 0.536±0.168 | price 13.64 | |adv| 0.79 | KL -0.0012 | epochs 10
ep 10880 | reward -15.40 | Δ 0.538±0.171 | price 13.83 | |adv| 0.79 | KL 0.0483 | epochs 3
ep 11008 | reward -15.25 | Δ 0.455±0.170 | price 5.47 | |adv| 0.79 | KL 0.0761 | epochs 3
ep 11136 | reward -14.04 | Δ 0.545±0.154 | price 14.46 | |adv| 0.70 | KL 0.0322 | epochs 3
ep 11264 | reward -15.33 | Δ 0.470±0.171 | price 7.05 | |adv| 0.75 | KL 0.0662 | epochs 3
ep 11392 | reward -17.04 | Δ 0.559±0.176 | price 15.86 | |adv| 0.77 | KL 0.0433 | epochs 3
ep 11520 | reward -12.29 | Δ 0.480±0.138 | price 8.00 | |adv| 0.78 | KL 0.0274 | epochs 3
ep 11648 | reward -12.39 | Δ 0.505±0.146 | price 10.49 | |adv| 0.81 | KL -0.0065 | epochs 10
ep 11776 | reward -13.16 | Δ 0.516±0.139 | price 11.57 | |adv| 0.79 | KL 0.0011 | epochs 10
ep 11904 | reward -11.93 | Δ 0.521±0.131 | price 12.10 | |adv| 0.75 | KL -0.0025 | epochs 10
ep 12032 | reward -13.70 | Δ 0.497±0.160 | price 9.70 | |adv| 0.78 | KL 0.0124 | epochs 10
ep 12160 | reward -11.87 | Δ 0.519±0.139 | price 11.92 | |adv| 0.74 | KL -0.0013 | epochs 10
ep 12288 | reward -13.06 | Δ 0.525±0.147 | price 12.52 | |adv| 0.76 | KL 0.0051 | epochs 10
ep 12416 | reward -12.56 | Δ 0.536±0.141 | price 13.65 | |adv| 0.74 | KL 0.0305 | epochs 6
ep 12544 | reward -11.91 | Δ 0.503±0.134 | price 10.34 | |adv| 0.76 | KL 0.0042 | epochs 10
ep 12672 | reward -12.10 | Δ 0.493±0.140 | price 9.32 | |adv| 0.71 | KL 0.0009 | epochs 10
ep 12800 | reward -11.90 | Δ 0.487±0.134 | price 8.71 | |adv| 0.78 | KL 0.0093 | epochs 10
ep 12928 | reward -13.00 | Δ 0.489±0.145 | price 8.91 | |adv| 0.74 | KL 0.0730 | epochs 3
ep 13056 | reward -11.90 | Δ 0.555±0.122 | price 15.45 | |adv| 0.78 | KL 0.0410 | epochs 3
ep 13184 | reward -10.72 | Δ 0.501±0.128 | price 10.09 | |adv| 0.69 | KL 0.0353 | epochs 6
ep 13312 | reward -11.74 | Δ 0.515±0.132 | price 11.55 | |adv| 0.80 | KL 0.0203 | epochs 6
ep 13440 | reward -13.11 | Δ 0.553±0.135 | price 15.26 | |adv| 0.74 | KL 0.0403 | epochs 2
ep 13568 | reward -10.36 | Δ 0.511±0.118 | price 11.07 | |adv| 0.76 | KL 0.0232 | epochs 3
ep 13696 | reward -11.65 | Δ 0.493±0.125 | price 9.33 | |adv| 0.78 | KL 0.0208 | epochs 2
ep 13824 | reward -10.78 | Δ 0.540±0.112 | price 14.05 | |adv| 0.80 | KL 0.0489 | epochs 3
ep 13952 | reward -11.13 | Δ 0.506±0.130 | price 10.62 | |adv| 0.80 | KL 0.0263 | epochs 2
ep 14080 | reward -11.91 | Δ 0.477±0.139 | price 7.67 | |adv| 0.72 | KL 0.0330 | epochs 3
ep 14208 | reward -11.26 | Δ 0.518±0.130 | price 11.75 | |adv| 0.76 | KL 0.0224 | epochs 3
ep 14336 | reward -11.63 | Δ 0.488±0.135 | price 8.76 | |adv| 0.75 | KL 0.0354 | epochs 3
ep 14464 | reward -9.99 | Δ 0.527±0.121 | price 12.69 | |adv| 0.73 | KL 0.0241 | epochs 3
ep 14592 | reward -12.30 | Δ 0.484±0.143 | price 8.36 | |adv| 0.76 | KL 0.0210 | epochs 4
ep 14720 | reward -10.36 | Δ 0.493±0.116 | price 9.30 | |adv| 0.71 | KL 0.0622 | epochs 3
ep 14848 | reward -11.64 | Δ 0.539±0.126 | price 13.92 | |adv| 0.80 | KL 0.0272 | epochs 4
ep 14976 | reward -9.85 | Δ 0.519±0.118 | price 11.88 | |adv| 0.78 | KL 0.0212 | epochs 3
ep 15104 | reward -11.01 | Δ 0.499±0.131 | price 9.93 | |adv| 0.77 | KL 0.0003 | epochs 10
ep 15232 | reward -9.93 | Δ 0.479±0.117 | price 7.85 | |adv| 0.82 | KL 0.0478 | epochs 3
ep 15360 | reward -10.62 | Δ 0.537±0.118 | price 13.67 | |adv| 0.76 | KL 0.0641 | epochs 3
ep 15488 | reward -11.08 | Δ 0.495±0.128 | price 9.55 | |adv| 0.81 | KL 0.0202 | epochs 4
ep 15616 | reward -8.96 | Δ 0.525±0.100 | price 12.45 | |adv| 0.78 | KL 0.0361 | epochs 2
ep 15744 | reward -9.64 | Δ 0.509±0.117 | price 10.86 | |adv| 0.76 | KL 0.0472 | epochs 2
ep 15872 | reward -11.73 | Δ 0.458±0.133 | price 5.83 | |adv| 0.74 | KL 0.0498 | epochs 3
ep 16000 | reward -12.48 | Δ 0.492±0.133 | price 9.21 | |adv| 0.76 | KL 0.0305 | epochs 2
ep 16128 | reward -9.07 | Δ 0.528±0.102 | price 12.80 | |adv| 0.79 | KL 0.0421 | epochs 3
ep 16256 | reward -10.51 | Δ 0.493±0.122 | price 9.26 | |adv| 0.79 | KL -0.0003 | epochs 10
ep 16384 | reward -10.27 | Δ 0.496±0.117 | price 9.59 | |adv| 0.77 | KL 0.0200 | epochs 7
ep 16512 | reward -9.24 | Δ 0.481±0.104 | price 8.09 | |adv| 0.80 | KL 0.0246 | epochs 3
ep 16640 | reward -9.61 | Δ 0.493±0.115 | price 9.28 | |adv| 0.76 | KL 0.0001 | epochs 10
ep 16768 | reward -8.37 | Δ 0.504±0.104 | price 10.40 | |adv| 0.77 | KL 0.0255 | epochs 3
ep 16896 | reward -10.26 | Δ 0.464±0.112 | price 6.39 | |adv| 0.79 | KL 0.0902 | epochs 3
ep 17024 | reward -8.94 | Δ 0.520±0.096 | price 11.96 | |adv| 0.79 | KL 0.0257 | epochs 3
ep 17152 | reward -9.44 | Δ 0.499±0.107 | price 9.87 | |adv| 0.81 | KL 0.0226 | epochs 3
ep 17280 | reward -10.72 | Δ 0.521±0.115 | price 12.07 | |adv| 0.77 | KL 0.0021 | epochs 10
ep 17408 | reward -10.77 | Δ 0.529±0.115 | price 12.89 | |adv| 0.80 | KL 0.0397 | epochs 3
ep 17536 | reward -9.77 | Δ 0.486±0.113 | price 8.65 | |adv| 0.75 | KL 0.0209 | epochs 7
ep 17664 | reward -9.85 | Δ 0.508±0.120 | price 10.75 | |adv| 0.79 | KL 0.0098 | epochs 10
ep 17792 | reward -9.49 | Δ 0.517±0.108 | price 11.70 | |adv| 0.78 | KL 0.0302 | epochs 3
ep 17920 | reward -8.66 | Δ 0.487±0.100 | price 8.67 | |adv| 0.80 | KL 0.0083 | epochs 10
ep 18048 | reward -9.85 | Δ 0.522±0.110 | price 12.19 | |adv| 0.76 | KL 0.0349 | epochs 2
ep 18176 | reward -8.22 | Δ 0.490±0.108 | price 8.95 | |adv| 0.75 | KL 0.0043 | epochs 10
ep 18304 | reward -9.21 | Δ 0.500±0.104 | price 10.05 | |adv| 0.78 | KL 0.0024 | epochs 10
ep 18432 | reward -9.09 | Δ 0.502±0.105 | price 10.18 | |adv| 0.74 | KL 0.0055 | epochs 10
ep 18560 | reward -8.15 | Δ 0.516±0.095 | price 11.62 | |adv| 0.82 | KL 0.0548 | epochs 2
ep 18688 | reward -9.37 | Δ 0.476±0.104 | price 7.59 | |adv| 0.80 | KL 0.0301 | epochs 3
ep 18816 | reward -9.02 | Δ 0.493±0.099 | price 9.30 | |adv| 0.78 | KL 0.0358 | epochs 2
ep 18944 | reward -8.20 | Δ 0.527±0.097 | price 12.74 | |adv| 0.75 | KL 0.0661 | epochs 3
ep 19072 | reward -7.43 | Δ 0.484±0.088 | price 8.39 | |adv| 0.76 | KL 0.0402 | epochs 3
ep 19200 | reward -7.78 | Δ 0.522±0.087 | price 12.22 | |adv| 0.79 | KL 0.0634 | epochs 3
ep 19328 | reward -8.03 | Δ 0.480±0.094 | price 8.02 | |adv| 0.78 | KL 0.0421 | epochs 3
ep 19456 | reward -7.68 | Δ 0.508±0.089 | price 10.78 | |adv| 0.80 | KL 0.0026 | epochs 10
ep 19584 | reward -9.20 | Δ 0.505±0.103 | price 10.49 | |adv| 0.76 | KL 0.0305 | epochs 2
ep 19712 | reward -8.14 | Δ 0.473±0.092 | price 7.31 | |adv| 0.77 | KL 0.0240 | epochs 3
ep 19840 | reward -7.66 | Δ 0.518±0.092 | price 11.75 | |adv| 0.80 | KL 0.0289 | epochs 2
ep 19968 | reward -10.01 | Δ 0.475±0.108 | price 7.47 | |adv| 0.78 | KL 0.0032 | epochs 10

--- Final policy at S0 ---
Δ (mean) = 0.4929, std = 0.0960
Implied fair price X = Δ \* S0 + B = 9.2946
Mean abs replication error over 1000 sim8.12 | Δ 0.096±0.166

B was 0 instead of -40

Replication - at 400 000 episodes, we get --- Final policy at S0 ---
Δ (mean) = 0.5047, std = 0.0107
Implied fair price X = Δ \* S0 + B = 10.4721
Mean abs replication error over 1000 sims: 0.519605
