import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import pandas as pd

class NewtonInterpolation:
    def __init__(self, f: Callable[[float], float], x0: float, xn: float, n: int):
        """
        Initialize Newton interpolation with given function and parameters.
        
        Args:
            f: Function to interpolate
            x0: Starting point of interval
            xn: End point of interval
            n: Number of nodes
        """
        self.f = f
        self.x0 = x0
        self.xn = xn
        self.n = n
        self.h = (xn - x0) / n
        
        # Generate interpolation nodes
        self.x_nodes = np.array([x0 + i * self.h for i in range(n + 1)])
        self.f_nodes = np.array([f(x) for x in self.x_nodes])
        
        # Precompute divided differences table
        self.divided_diff_table = self._compute_divided_differences()
    
    def _compute_divided_differences(self) -> np.ndarray:
        """Compute divided differences table."""
        n = len(self.x_nodes)
        table = np.zeros((n, n))
        
        # Fill first column with function values
        table[:, 0] = self.f_nodes
        
        # Fill the rest of the table
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / \
                             (self.x_nodes[i + j] - self.x_nodes[i])
        
        return table
    
    def w_k(self, x: float, k: int) -> float:
        """
        Compute w_k(x) = product from i=0 to k of (x - x_i).
        
        Args:
            x: Point to evaluate at
            k: Order of the product
            
        Returns:
            Value of w_k(x)
        """
        if k < 0:
            return 1.0
        
        product = 1.0
        for i in range(k + 1):
            product *= (x - self.x_nodes[i])
        
        return product
    
    def divided_difference(self, k: int) -> float:
        """
        Get divided difference f[x_0, ..., x_k] from precomputed table.
        
        Args:
            k: Order of divided difference
            
        Returns:
            Value of divided difference
        """
        if k < 0 or k >= len(self.x_nodes):
            return 0.0
        
        return self.divided_diff_table[0, k]
    
    def newton_polynomial(self, x: float) -> float:
        """
        Evaluate Newton interpolating polynomial N_n(x).
        
        Args:
            x: Point to evaluate at
            
        Returns:
            Value of N_n(x)
        """
        result = self.f_nodes[0]  # f_0
        
        for k in range(1, self.n + 1):
            result += self.w_k(x, k - 1) * self.divided_difference(k)
        
        return result
    
    def interpolation_error(self, x: float) -> float:
        """
        Compute interpolation error eps(x) = |f(x) - N_n(x)|.
        
        Args:
            x: Point to evaluate at
            
        Returns:
            Absolute error
        """
        return abs(self.f(x) - self.newton_polynomial(x))
    
    def w_n(self, x: float) -> float:
        """
        Compute w_n(x) = product from i=0 to n of (x - x_i).
        
        Args:
            x: Point to evaluate at
            
        Returns:
            Value of w_n(x)
        """
        return self.w_k(x, self.n)

def tabulate_and_plot(interpolator: NewtonInterpolation, a: float, b: float, 
                      num_points: int = None, title_suffix: str = "", show_plot: bool = True):
    """
    Tabulate and plot f(x), N_n(x), error, and w_n(x) on interval [a,b].
    
    Args:
        interpolator: Newton interpolation object
        a: Start of interval
        b: End of interval
        num_points: Number of points for tabulation (default: 20*n)
        title_suffix: Suffix for plot title
        show_plot: Whether to show the plot
    """
    if num_points is None:
        num_points = 20 * interpolator.n
    
    h_tab = (b - a) / num_points
    x_values = np.array([a + i * h_tab for i in range(num_points + 1)])
    
    # Compute function values
    f_values = np.array([interpolator.f(x) for x in x_values])
    n_values = np.array([interpolator.newton_polynomial(x) for x in x_values])
    error_values = np.array([interpolator.interpolation_error(x) for x in x_values])
    w_n_values = np.array([interpolator.w_n(x) for x in x_values])
    
    # Create tabulation DataFrame
    df = pd.DataFrame({
        'x': x_values,
        'f(x)': f_values,
        'N_n(x)': n_values,
        'error |f(x)-N_n(x)|': error_values,
        'w_n(x)': w_n_values
    })
    
    print(f"\nTabulation for interval [{a:.2f}, {b:.2f}] with {num_points} points:")
    print("First 10 rows:")
    print(df.head(10).round(6))
    print("\nLast 5 rows:")
    print(df.tail(5).round(6))
    
    # Statistics
    max_error = np.max(error_values)
    mean_error = np.mean(error_values)
    print(f"\nError statistics:")
    print(f"Maximum error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")
    
    # Create plots
    if show_plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Function and interpolation
        ax1.plot(x_values, f_values, 'b-', label='f(x)', linewidth=2)
        ax1.plot(x_values, n_values, 'r--', label='N_n(x)', linewidth=2)
        ax1.plot(interpolator.x_nodes, interpolator.f_nodes, 'go', 
                markersize=8, label='Interpolation nodes')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Function and Interpolation {title_suffix}')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Error
        ax2.plot(x_values, error_values, 'r-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('|f(x) - N_n(x)|')
        ax2.set_title(f'Interpolation Error {title_suffix}')
        ax2.grid(True)
        ax2.set_yscale('log')
        
        # Plot 3: w_n(x) function
        ax3.plot(x_values, w_n_values, 'g-', linewidth=2)
        ax3.plot(interpolator.x_nodes, np.zeros_like(interpolator.x_nodes), 
                'ro', markersize=8, label='Nodes (roots)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('w_n(x)')
        ax3.set_title(f'Function w_n(x) {title_suffix}')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: All functions together (normalized)
        f_norm = f_values / np.max(np.abs(f_values))
        n_norm = n_values / np.max(np.abs(n_values))
        error_norm = error_values / np.max(error_values) if np.max(error_values) > 0 else error_values
        w_norm = w_n_values / np.max(np.abs(w_n_values))
        
        ax4.plot(x_values, f_norm, 'b-', label='f(x) normalized', alpha=0.7)
        ax4.plot(x_values, n_norm, 'r--', label='N_n(x) normalized', alpha=0.7)
        ax4.plot(x_values, error_norm, 'orange', label='Error normalized', alpha=0.7)
        ax4.plot(x_values, w_norm, 'g-', label='w_n(x) normalized', alpha=0.7)
        ax4.set_xlabel('x')
        ax4.set_ylabel('Normalized values')
        ax4.set_title(f'All Functions (Normalized) {title_suffix}')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return df, max_error, mean_error

def investigate_accuracy_vs_nodes(f: Callable[[float], float], a: float, b: float):
    """Investigate accuracy vs number of nodes with fixed interval."""
    print("="*60)
    print("INVESTIGATION 1: Accuracy vs Number of Nodes (Fixed Interval)")
    print("="*60)
    
    node_counts = [5, 10, 20]
    results = []
    
    for n in node_counts:
        print(f"\n--- Analysis for n = {n} nodes ---")
        interpolator = NewtonInterpolation(f, a, b, n)
        df, max_error, mean_error = tabulate_and_plot(interpolator, a, b, 
                                                     title_suffix=f"(n={n})", 
                                                     show_plot=False)
        
        results.append({
            'n': n,
            'h': (b - a) / n,
            'max_error': max_error,
            'mean_error': mean_error,
            'interpolator': interpolator
        })
    
    # Summary table
    print(f"\n{'='*50}")
    print("SUMMARY: Accuracy vs Number of Nodes")
    print(f"{'='*50}")
    summary_df = pd.DataFrame([{
        'Nodes (n)': r['n'],
        'Step (h)': f"{r['h']:.4f}",
        'Max Error': f"{r['max_error']:.6f}",
        'Mean Error': f"{r['mean_error']:.6f}"
    } for r in results])
    print(summary_df.to_string(index=False))
    
    return results

def investigate_accuracy_vs_interval(f: Callable[[float], float], a: float, h: float):
    """Investigate accuracy vs interval size with fixed step."""
    print("="*60)
    print("INVESTIGATION 2: Accuracy vs Interval Size (Fixed Step)")
    print("="*60)
    
    node_counts = [5, 10, 20]
    results = []
    
    for n in node_counts:
        b = a + h * n
        print(f"\n--- Analysis for n = {n} nodes, interval [{a}, {b:.4f}] ---")
        interpolator = NewtonInterpolation(f, a, b, n)
        df, max_error, mean_error = tabulate_and_plot(interpolator, a, b,
                                                     title_suffix=f"(interval=[{a},{b:.2f}])",
                                                     show_plot=False)
        
        results.append({
            'n': n,
            'interval': f"[{a}, {b:.4f}]",
            'interval_size': b - a,
            'max_error': max_error,
            'mean_error': mean_error,
            'interpolator': interpolator
        })
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Accuracy vs Interval Size")
    print(f"{'='*60}")
    summary_df = pd.DataFrame([{
        'Nodes (n)': r['n'],
        'Interval': r['interval'],
        'Size': f"{r['interval_size']:.4f}",
        'Max Error': f"{r['max_error']:.6f}",
        'Mean Error': f"{r['mean_error']:.6f}"
    } for r in results])
    print(summary_df.to_string(index=False))
    
    return results

def demonstrate_divided_differences(interpolator: NewtonInterpolation):
    """Demonstrate divided differences calculation."""
    print("="*60)
    print("DEMONSTRATION: Divided Differences")
    print("="*60)
    
    print("Divided differences table (first few orders):")
    for k in range(min(6, len(interpolator.x_nodes))):
        print(f"f[x_0,...,x_{k}] = {interpolator.divided_difference(k):.6f}")
    
    # Show interpolation at a specific point
    x_test = (interpolator.x0 + interpolator.xn) / 3
    print(f"\nInterpolation at x = {x_test:.4f}:")
    print(f"True value: f({x_test:.4f}) = {interpolator.f(x_test):.6f}")
    print(f"Interpolated: N_{interpolator.n}({x_test:.4f}) = {interpolator.newton_polynomial(x_test):.6f}")
    print(f"Error: {interpolator.interpolation_error(x_test):.6f}")
    
    # Show w_k values
    print(f"\nw_k values at x = {x_test:.4f}:")
    for k in range(min(6, len(interpolator.x_nodes))):
        print(f"w_{k}({x_test:.4f}) = {interpolator.w_k(x_test, k):.6f}")

def main():
    """Main function to run the Newton interpolation program."""
    
    # Define test function
    def test_function(x):
        """Test function: f(x) = sin(x) + 0.5*cos(2*x)"""
        return np.sin(x) + 0.5 * np.cos(2 * x)
    
    print("NEWTON INTERPOLATION PROGRAM")
    print("Function: f(x) = sin(x) + 0.5*cos(2x)")
    print("="*60)
    
    # Initialize parameters
    x0, xn, n = 0, np.pi, 5
    
    # Create interpolation object
    interpolator = NewtonInterpolation(test_function, x0, xn, n)
    
    # Demonstrate basic functionality
    demonstrate_divided_differences(interpolator)
    
    # Example tabulation with plots
    print("\n" + "="*60)
    print("EXAMPLE: Tabulation and Plotting")
    print("="*60)
    df, max_error, mean_error = tabulate_and_plot(interpolator, 0, np.pi, 
                                                 title_suffix="(Example)")
    
    # Run investigations
    results1 = investigate_accuracy_vs_nodes(test_function, 0, np.pi)
    results2 = investigate_accuracy_vs_interval(test_function, 0, np.pi / 5)
    
    # Show comparison plots for different node counts
    print("\n" + "="*60)
    print("COMPARISON PLOTS")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Error vs nodes (fixed interval)
    nodes = [r['n'] for r in results1]
    max_errors1 = [r['max_error'] for r in results1]
    mean_errors1 = [r['mean_error'] for r in results1]
    
    ax1.semilogy(nodes, max_errors1, 'ro-', label='Max Error', markersize=8)
    ax1.semilogy(nodes, mean_errors1, 'bs-', label='Mean Error', markersize=8)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Error')
    ax1.set_title('Error vs Number of Nodes (Fixed Interval)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Error vs interval size (fixed step)
    interval_sizes = [r['interval_size'] for r in results2]
    max_errors2 = [r['max_error'] for r in results2]
    mean_errors2 = [r['mean_error'] for r in results2]
    
    ax2.semilogy(interval_sizes, max_errors2, 'ro-', label='Max Error', markersize=8)
    ax2.semilogy(interval_sizes, mean_errors2, 'bs-', label='Mean Error', markersize=8)
    ax2.set_xlabel('Interval Size')
    ax2.set_ylabel('Error')
    ax2.set_title('Error vs Interval Size (Fixed Step)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

