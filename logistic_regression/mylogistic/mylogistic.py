import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from typing import Any

def logistic(Y: np.ndarray) -> np.ndarray:
    """
    Applies the logistic (sigmoid) function to each element of the input array Y.

    The logistic (sigmoid) function is defined as:
    
        σ(z) = 1 / (1 + exp(-z))
    
    and is computed elementwise on the array Y.

    Parameters
    ----------
    Y : np.ndarray
        A NumPy array (of any shape) containing the input values (z).

    Returns
    -------
    np.ndarray
        A NumPy array of the same shape as Y with the sigmoid function applied to each element.
    """
    # Define a helper function to compute the sigmoid for a single value.
    def sigmoid(zi: float) -> float:
        from math import exp
        return 1 / (1 + exp(-zi))
    
    # Vectorize the sigmoid function so that it can be applied elementwise.
    logistic_values = np.vectorize(sigmoid)
    
    # Return the result with the sigmoid function applied to every element of Y.
    return logistic_values(Y)

# Example usage:
if __name__ == "__main__":
    # Create an example input array of values
    y_values = np.linspace(-10, 10, 100)  # 100 evenly-spaced numbers between -10 and 10
    
    # Compute the logistic values for the input array
    logistic_output = logistic(y_values)
    
    # For visualization purposes, we also want to show the "heaviside" function result.
    # Assume a heaviside function is already defined somewhere
    # Here we define a simple heaviside function for demonstration.
    def heaviside(Y: np.ndarray) -> np.ndarray:
        """
        Applies the Heaviside function to each element of Y.
        Returns 1.0 if the element is > 0, otherwise returns 0.0.
        """
        return np.where(Y > 0, 1.0, 0.0)
    
    heaviside_output = heaviside(y_values)
    
    # Configure plot settings for higher resolution and better style.
    mpl.rc("savefig", dpi=120)  # Adjust for higher-resolution figures
    sns.set_style("darkgrid")
    
    # Plot the heaviside function output (in blue) and the logistic function output (in red dashed).
    plt.figure(figsize=(8, 6))
    plt.plot(y_values, heaviside_output, 'b', label="Heaviside")
    plt.plot(y_values, logistic_output, 'r--', label="Logistic (sigmoid)")
    
    # Set labels and title for clarity.
    plt.xlabel("Input y")
    plt.ylabel("Output")
    plt.title("Heaviside Function vs. Logistic (Sigmoid) Function")
    plt.legend()
    plt.show()