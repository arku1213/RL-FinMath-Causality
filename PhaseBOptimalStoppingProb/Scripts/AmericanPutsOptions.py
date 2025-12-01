import numpy as np

class AmericanPutBinomial:
    """
    American Put Option Pricing using the Binomial Model
    Solves the Optimal Stopping Problem via Backward Induction
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
        
        At each node, we choose the stopping time τ to maximize:
        E[e^(-rτ) * (K - S_τ)^+]
        
        This is done via dynamic programming (backward induction):
        V(t,S) = max{K - S, e^(-r*dt) * E[V(t+dt, S')]}
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
    
    def print_results(self):
        """Print option pricing results"""
        option_price = self.price_option()
        
        print("\n" + "="*80)
        print("AMERICAN PUT OPTION - OPTIMAL STOPPING PROBLEM")
        print("="*80)
        print(f"\nParameters:")
        print(f"  S₀ = ${self.S0:.2f}  |  K = ${self.K:.2f}  |  T = {self.T:.2f} years")
        print(f"  r = {self.r*100:.2f}%  |  σ = {self.sigma*100:.2f}%  |  N = {self.N} steps")
        print(f"\nBinomial Parameters:")
        print(f"  dt = {self.dt:.4f}  |  u = {self.u:.4f}  |  d = {self.d:.4f}")
        print(f"  p = {self.p:.4f}  |  discount = {self.discount:.4f}")
        print(f"\nOptimal Stopping Problem Solution:")
        print(f"  V(0, S₀) = ${option_price:.4f}")
        print(f"\n  Interpretation: The optimal exercise strategy yields")
        print(f"  an expected discounted payoff of ${option_price:.4f}")
        print("="*80)
    
    def print_tree(self, show_first_n_steps=5):
        """Print the stock tree and exercise decisions"""
        if self.option_tree is None:
            self.price_option()
        
        display_N = min(self.N, show_first_n_steps)
        
        print("\n" + "="*60)
        print("OPTIMAL EXERCISE POLICY")
        print("="*60)
        
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
        
        print("\n" + "="*60)
        print("EARLY EXERCISE BOUNDARY")
        print("="*60)
        print("Exercise immediately (τ* = t) if stock price falls below:\n")
        print(f"{'Time t':<15} {'Exercise if S ≤':<20}")
        print("-" * 35)
        
        for t, b in zip(times, boundary):
            print(f"{t:<15.3f} ${b:<19.2f}")
        
        print("="*60)

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
        print("AMERICAN vs EUROPEAN PUT")
        print(f"{'='*80}")
        print(f"  European (τ = T fixed):     ${european_value:.4f}")
        print(f"  American (optimal τ):       ${american_value:.4f}")
        print(f"  Early Exercise Premium:     ${early_exercise_premium:.4f} ({early_exercise_premium/european_value*100:.2f}%)")
        print(f"\nWhy is American Put more valuable?")
        print(f"  Flexibility to choose τ optimally allows capturing")
        print(f"  intrinsic value early when deep in-the-money.")
        print(f"{'='*80}")


# Example usage
if __name__ == "__main__":
    # Parameters
    S0 = 100      # Initial stock price
    K = 100       # Strike price (at-the-money)
    T = 1.0       # 1 year to maturity
    r = 0.05      # 5% risk-free rate
    sigma = 0.20  # 20% volatility
    N = 10        # 10 time steps
    
    # Create and solve the model
    model = AmericanPutBinomial(S0, K, T, r, sigma, N)
    
    # Solve the optimal stopping problem
    model.price_option()
    
    # Show the optimal policy
    model.print_tree(show_first_n_steps=5)
    
    # Show exercise boundary
    model.print_exercise_boundary()