import numpy as np
import matplotlib.pyplot as plt

class AmericanPutBinomial:
    """
    American Put Option Pricing using the Binomial Model
    Solves the Optimal Stopping Problem
    """
    
    def __init__(self, S0, K, T, r, sigma, N):
        """
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annual)
        sigma : float
            Volatility (annual)
        N : int
            Number of time steps
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        
        # Calculate binomial parameters
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability
        self.discount = np.exp(-r * self.dt)  # Discount factor
        
        # Storage for results
        self.stock_tree = None
        self.option_tree = None
        self.exercise_tree = None
        
    def build_stock_tree(self):
        """Build the stock price tree"""
        tree = np.zeros((self.N + 1, self.N + 1))
        
        for i in range(self.N + 1):
            for j in range(i + 1):
                tree[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
        
        self.stock_tree = tree
        return tree
    
    def price_option(self):
        """
        Price the American Put using backward induction
        Solves the optimal stopping problem at each node
        """
        if self.stock_tree is None:
            self.build_stock_tree()
        
        # Initialize option value tree and exercise decision tree
        option_tree = np.zeros((self.N + 1, self.N + 1))
        exercise_tree = np.zeros((self.N + 1, self.N + 1), dtype=bool)
        
        # Terminal payoff at maturity (intrinsic value only)
        for j in range(self.N + 1):
            option_tree[j, self.N] = max(self.K - self.stock_tree[j, self.N], 0)
            exercise_tree[j, self.N] = option_tree[j, self.N] > 0
        
        # Backward induction - THE OPTIMAL STOPPING PROBLEM
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                # Stock price at this node
                S = self.stock_tree[j, i]
                
                # Intrinsic value (immediate exercise payoff)
                intrinsic = max(self.K - S, 0)
                
                # Continuation value (expected discounted value of holding)
                continuation = self.discount * (
                    self.p * option_tree[j, i + 1] + 
                    (1 - self.p) * option_tree[j + 1, i + 1]
                )
                
                # OPTIMAL DECISION: exercise if intrinsic > continuation
                if intrinsic > continuation:
                    option_tree[j, i] = intrinsic
                    exercise_tree[j, i] = True
                else:
                    option_tree[j, i] = continuation
                    exercise_tree[j, i] = False
        
        self.option_tree = option_tree
        self.exercise_tree = exercise_tree
        
        return option_tree[0, 0]
    
    def get_exercise_boundary(self):
        """
        Extract the early exercise boundary
        Returns arrays of times and corresponding boundary stock prices
        """
        if self.exercise_tree is None:
            self.price_option()
        
        times = []
        boundary_prices = []
        
        for i in range(self.N + 1):
            # Find the highest stock price at time i where exercise is optimal
            exercise_prices = []
            for j in range(i + 1):
                if self.exercise_tree[j, i]:
                    exercise_prices.append(self.stock_tree[j, i])
            
            if exercise_prices:
                times.append(i * self.dt)
                boundary_prices.append(max(exercise_prices))
        
        return np.array(times), np.array(boundary_prices)
    
    def print_tree(self, show_first_n_steps=5):
        """Print the stock tree and exercise decisions"""
        if self.option_tree is None:
            self.price_option()
        
        display_N = min(self.N, show_first_n_steps)
        
        print("\n" + "="*80)
        print("STOCK PRICE TREE WITH EXERCISE DECISIONS")
        print("="*80)
        print("Format: [Stock Price | Option Value | Decision]")
        print("EXERCISE = optimal to exercise immediately")
        print("HOLD = optimal to continue holding\n")
        
        for i in range(display_N + 1):
            print(f"\nTime Step {i} (t = {i*self.dt:.2f}):")
            for j in range(i + 1):
                stock_price = self.stock_tree[j, i]
                option_value = self.option_tree[j, i]
                decision = "EXERCISE" if self.exercise_tree[j, i] else "HOLD"
                
                print(f"  Node ({i},{j}): S=${stock_price:6.2f} | V=${option_value:6.3f} | {decision}")
    
    def print_exercise_boundary(self):
        """Print the exercise boundary"""
        times, boundary = self.get_exercise_boundary()
        
        print("\n" + "="*80)
        print("EARLY EXERCISE BOUNDARY")
        print("="*80)
        print("Exercise the option if stock price falls below these levels:\n")
        print(f"{'Time (years)':<15} {'Boundary Price':<20} {'Strike Price':<15}")
        print("-" * 50)
        
        for t, b in zip(times, boundary):
            print(f"{t:<15.3f} ${b:<19.2f} ${self.K:<14.2f}")
        
        print("\nInterpretation: At each time, if S < Boundary Price, EXERCISE immediately.")
        print(f"                 Otherwise, HOLD the option.")
    
    def print_results(self):
        """Print comprehensive results"""
        option_price = self.price_option()
        
        print("\n" + "="*80)
        print("AMERICAN PUT OPTION - BINOMIAL MODEL")
        print("="*80)
        print(f"\nParameters:")
        print(f"  Initial Stock Price (S0): ${self.S0:.2f}")
        print(f"  Strike Price (K):         ${self.K:.2f}")
        print(f"  Time to Maturity (T):     {self.T:.2f} years")
        print(f"  Risk-free Rate (r):       {self.r*100:.2f}%")
        print(f"  Volatility (σ):           {self.sigma*100:.2f}%")
        print(f"  Number of Steps (N):      {self.N}")
        print(f"\nBinomial Parameters:")
        print(f"  Time step (dt):           {self.dt:.4f}")
        print(f"  Up factor (u):            {self.u:.4f}")
        print(f"  Down factor (d):          {self.d:.4f}")
        print(f"  Risk-neutral prob (p):    {self.p:.4f}")
        print(f"  Discount factor:          {self.discount:.4f}")
        print(f"\nResult:")
        print(f"  American Put Value:       ${option_price:.4f}")
        print("="*80)
    
    def simulate_path(self, seed=None):
        """
        Simulate a single stock price path through the binomial tree
        Returns: path_moves (list of 'U' or 'D'), path_prices, path_times
        """
        if seed is not None:
            np.random.seed(seed)
        
        path_moves = []
        path_prices = [self.S0]
        path_times = [0.0]
        
        current_price = self.S0
        
        for i in range(self.N):
            # Generate random move (Up or Down)
            if np.random.random() < self.p:
                move = 'U'
                current_price *= self.u
            else:
                move = 'D'
                current_price *= self.d
            
            path_moves.append(move)
            path_prices.append(current_price)
            path_times.append((i + 1) * self.dt)
        
        return path_moves, path_prices, path_times
    
    def find_optimal_exercise_time_for_path(self, path_moves):
        """
        Given a path (sequence of 'U' and 'D' moves), find the optimal exercise time τ*
        
        Parameters:
        -----------
        path_moves : list of str
            Sequence of 'U' (up) or 'D' (down) moves
        
        Returns:
        --------
        optimal_time : float or None
            The first time at which exercise is optimal along this path
            Returns None if never optimal to exercise early
        node_info : dict
            Information about the exercise node
        """
        if self.exercise_tree is None:
            self.price_option()
        
        # Start at root
        j = 0  # Current position in the tree (number of down moves)
        
        # Check each time step along the path
        for i, move in enumerate(path_moves):
            # Current node is (i, j)
            current_time = i * self.dt
            current_stock_price = self.stock_tree[j, i]
            current_option_value = self.option_tree[j, i]
            should_exercise = self.exercise_tree[j, i]
            
            if should_exercise:
                # Found the optimal exercise time!
                intrinsic_value = max(self.K - current_stock_price, 0)
                return current_time, {
                    'time': current_time,
                    'time_step': i,
                    'stock_price': current_stock_price,
                    'option_value': current_option_value,
                    'intrinsic_value': intrinsic_value,
                    'node': (i, j)
                }
            
            # Move to next node based on the move
            if move == 'D':
                j += 1  # Down move increases j
        
        # Check the final node at maturity
        i = self.N
        current_time = self.T
        current_stock_price = self.stock_tree[j, i]
        intrinsic_value = max(self.K - current_stock_price, 0)
        
        # At maturity, exercise if in-the-money
        if intrinsic_value > 0:
            return current_time, {
                'time': current_time,
                'time_step': i,
                'stock_price': current_stock_price,
                'option_value': intrinsic_value,
                'intrinsic_value': intrinsic_value,
                'node': (i, j)
            }
        else:
            # Out of the money at maturity - option expires worthless
            return None, {
                'time': current_time,
                'time_step': i,
                'stock_price': current_stock_price,
                'option_value': 0.0,
                'intrinsic_value': 0.0,
                'node': (i, j),
                'expired_worthless': True
            }
    
    def simulate_multiple_paths(self, num_paths=10, show_details=True):
        """
        Simulate multiple paths and find optimal exercise time for each
        
        Parameters:
        -----------
        num_paths : int
            Number of paths to simulate
        show_details : bool
            Whether to print detailed path information
        
        Returns:
        --------
        results : list of dict
            Information about each path and its optimal exercise time
        """
        if self.exercise_tree is None:
            self.price_option()
        
        results = []
        
        print(f"\n{'='*80}")
        print(f"SIMULATING {num_paths} STOCK PRICE PATHS")
        print(f"{'='*80}\n")
        
        for path_num in range(num_paths):
            path_moves, path_prices, path_times = self.simulate_path(seed=path_num)
            optimal_time, node_info = self.find_optimal_exercise_time_for_path(path_moves)
            
            result = {
                'path_number': path_num + 1,
                'path_moves': path_moves,
                'path_prices': path_prices,
                'path_times': path_times,
                'optimal_exercise_time': optimal_time,
                'exercise_info': node_info
            }
            results.append(result)
            
            if show_details:
                print(f"Path #{path_num + 1}:")
                print(f"  Moves: {' → '.join(path_moves[:5])}{'...' if len(path_moves) > 5 else ''}")
                print(f"  Final Stock Price: ${path_prices[-1]:.2f}")
                
                if optimal_time is not None:
                    print(f"  ✓ Optimal Exercise Time τ*: {optimal_time:.3f} years (step {node_info['time_step']})")
                    print(f"    Stock Price at Exercise: ${node_info['stock_price']:.2f}")
                    print(f"    Payoff at Exercise: ${node_info['intrinsic_value']:.2f}")
                else:
                    if node_info.get('expired_worthless'):
                        print(f"  ✗ Never exercised - expired worthless (final S = ${node_info['stock_price']:.2f} > K = ${self.K:.2f})")
                    else:
                        print(f"  → Hold until maturity")
                print()
        
        # Summary statistics
        exercise_times = [r['optimal_exercise_time'] for r in results if r['optimal_exercise_time'] is not None]
        early_exercises = [t for t in exercise_times if t < self.T]
        
        print(f"{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"  Total Paths Simulated: {num_paths}")
        print(f"  Paths with Early Exercise: {len(early_exercises)} ({len(early_exercises)/num_paths*100:.1f}%)")
        print(f"  Paths Exercised at Maturity: {len(exercise_times) - len(early_exercises)}")
        print(f"  Paths Expired Worthless: {num_paths - len(exercise_times)}")
        
        if early_exercises:
            print(f"\n  Early Exercise Times:")
            print(f"    Average: {np.mean(early_exercises):.3f} years")
            print(f"    Earliest: {np.min(early_exercises):.3f} years")
            print(f"    Latest: {np.max(early_exercises):.3f} years")
        
        print(f"{'='*80}")
        
        return results
    
    def compare_european_american(self):
        """Compare American put value with European put value"""
        american_value = self.option_tree[0, 0]
        
        # Price European put using the same tree (no early exercise)
        european_tree = np.zeros((self.N + 1, self.N + 1))
        
        # Terminal payoff
        for j in range(self.N + 1):
            european_tree[j, self.N] = max(self.K - self.stock_tree[j, self.N], 0)
        
        # Backward induction (no early exercise check)
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                european_tree[j, i] = self.discount * (
                    self.p * european_tree[j, i + 1] + 
                    (1 - self.p) * european_tree[j + 1, i + 1]
                )
        
        european_value = european_tree[0, 0]
        early_exercise_premium = american_value - european_value
        
        print(f"\n{'='*80}")
        print("COMPARISON: American vs European Put")
        print(f"{'='*80}")
        print(f"  European Put Value:       ${european_value:.4f}")
        print(f"  American Put Value:       ${american_value:.4f}")
        print(f"  Early Exercise Premium:   ${early_exercise_premium:.4f} ({early_exercise_premium/european_value*100:.2f}%)")
        print(f"\nWhy is American Put more valuable?")
        print(f"  The ability to exercise early is valuable, especially when the put")
        print(f"  is deep in-the-money. By exercising early, you receive the strike")
        print(f"  price K immediately and can earn interest on it.")
        print(f"{'='*80}")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("AMERICAN PUT OPTION PRICING - OPTIMAL STOPPING PROBLEM")
    print("="*80)
    
    # Parameters
    S0 = 100      # Initial stock price
    K = 100       # Strike price (at-the-money)
    T = 1.0       # 1 year to maturity
    r = 0.05      # 5% risk-free rate
    sigma = 0.20  # 20% volatility
    N = 10        # 10 time steps
    
    # Create and solve the model
    model = AmericanPutBinomial(S0, K, T, r, sigma, N)
    
    # Print results
    model.print_results()
    
    # Compare with European put
    model.compare_european_american()
    
    # Show the tree structure
    model.print_tree(show_first_n_steps=5)
    
    # Show exercise boundary
    model.print_exercise_boundary()
    
    # NEW: Simulate multiple paths and find optimal exercise time for each
    print("\n" + "="*80)
    print("PATH SIMULATION: Finding τ* for Each Path")
    print("="*80)
    print("Now we'll simulate individual stock price paths and determine the")
    print("optimal exercise time τ* for each path.\n")
    
    # Simulate 15 paths
    path_results = model.simulate_multiple_paths(num_paths=15, show_details=True)
    
    print("\n" + "="*80)
 