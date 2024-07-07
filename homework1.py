import numpy as np
import matplotlib.pyplot as plt

def approx_ln(x, n):
    """
    Approximates the natural logarithm of x using an iterative method.

    Parameters:
        x (float): The value for which to approximate the natural logarithm. Must be greater than 0.
        n (int): The number of iterations for the approximation.

    Returns:
        float: The approximate value of the natural logarithm of x.

    Raises:
        ValueError: If x is less than or equal to 0.
    """
    if x <= 0:
        raise ValueError("x must be greater than 0")
    
    # Initial values
    a_curr = (1 + x) / 2
    g_curr = np.sqrt(x)
    
    # Iterative calculation
    for _ in range(n):
        a_next = (a_curr + g_curr) / 2
        g_next = np.sqrt(a_next * g_curr)
        a_curr, g_curr = a_next, g_next
    
    # Final approximation
    ln_approx = (x - 1) / a_curr
    return ln_approx

def plot_absolute_error(x, max_n):
    """
    Plots the absolute error of the approximation for a given x as a function of n.

    Parameters:
        Same as for approx_ln.
    """
    true_ln_x = np.log(x)
    errors = []
    n_values = range(1, max_n + 1)
    
    for n in n_values:
        approx_ln_x = approx_ln(x, n)
        error = abs(true_ln_x - approx_ln_x)
        errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, errors, marker='o')
    plt.title(f'Absolute Error of approx_ln(x, n) for x = {x}')
    plt.xlabel('Number of iterations (n)')
    plt.ylabel('Absolute Error')
    plt.yscale('log')  # Log scale for better visualization of error convergence
    plt.grid(True)
    plt.show()

# Generate a range of x values
x_values = np.linspace(0.1, 5, 400)

# Compute the true ln(x) values
true_ln_values = np.log(x_values)

# Different values of n for approximation
n_values = [1, 2, 5, 10]

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Task 2
# Plot ln(x) and approx_ln(x) for different n values
for n in n_values:
    approx_ln_values = [approx_ln(x, n) for x in x_values]
    axes[0].plot(x_values, approx_ln_values, label=f'approx_ln, n={n}')
axes[0].plot(x_values, true_ln_values, label='ln(x)', color='black', linestyle='--')
axes[0].set_title('True ln(x) and Approximations')
axes[0].set_xlabel('x')
axes[0].set_ylabel('ln(x)')
axes[0].legend()
axes[0].grid(True)

# Plot the difference between ln(x) and approx_ln(x)
for n in n_values:
    approx_ln_values = [approx_ln(x, n) for x in x_values]
    difference = true_ln_values - approx_ln_values
    axes[1].plot(x_values, difference, label=f'n={n}')
axes[1].set_title('Difference between ln(x) and approx_ln(x)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Difference')
axes[1].legend()
axes[1].grid(True)

# Show plots
plt.tight_layout()
plt.show()

# Task 3: Plot the absolute error for x = 1.41
plot_absolute_error(1.41, 50)

# Task 4: Implement the algorithm for a converging function
def fast_approx_ln(x, n):
    """
    A faster converging function to approximate the natural logarithm using B.C. Carlson's algorithm.

    Parameters:
    x : float
        The value for which to approximate the natural logarithm. Must be greater than 0.
    n : int
        The number of iterations for the approximation. Must be greater than or equal to 1.

    Returns:
    float
        The approximate value of the natural logarithm of x.

    Raises:
    ValueError
        If n is less than 1 or if x is less than 0.
    """
    if n < 1 or x < 0:
        raise ValueError("n must be greater than or equal to 1, x must be greater than or equal to 0")
        
    # Step 1: Initialize a_0 and g_0
    a_0 = (1 + x) / 2
    g_0 = np.sqrt(x)
    
    # Arrays to store a_i and g_i values
    a = [a_0]
    g = [g_0]
    
    # Step 2: Iterate to compute a_i and g_i
    for _ in range(n):
        a_next = (a[-1] + g[-1]) / 2
        g_next = np.sqrt(a_next * g[-1])
        a.append(a_next)
        g.append(g_next)
        
    # Step 3: Initialize the d values
    d = np.zeros((n + 1, n + 1))  # d is an (n+1)x(n+1) matrix of zeros
    for i in range(n + 1):
        d[0, i] = a[i]

    # Step 4: Refine the d values
    for i in range(0, n + 1):
        for k in range(1, i + 1):
            d[k, i] = (d[k - 1, i] - 4**-k * d[k - 1, i - 1]) / (1 - 4**-k)
    
    # Step 5: Return the final approximation
    return (x - 1) / d[n, n]

# Example usage
x = 1.41
n = 10
print(f"The approximated ln({x}) value is: \n{fast_approx_ln(x, n)}")  # prints the approximated ln(x) value


# Task 5: Make a plot.

# Generate a range of x values
x_values = np.linspace(0.1, 20, 400)

# True ln(x) values
true_ln_values = np.log(x_values)

# Iterations to be plotted
iterations = [2, 3, 4, 5, 6]

# Initialize a figure
plt.figure(figsize=(10, 6))

# Plot error behavior for each iteration
for n in iterations:
    approx_ln_values = [fast_approx_ln(x, n) for x in x_values]
    errors = np.abs(true_ln_values - approx_ln_values)
    plt.plot(x_values, errors, label=f'iteration {n}', marker='o', linestyle='None')

# Set plot properties
plt.yscale('log')  # Log scale for y-axis
plt.xlabel('x')
plt.ylabel('error')
plt.title('Error behavior of the accelerated Carlsson method for the log')
plt.legend()
plt.grid(True)
plt.show()

