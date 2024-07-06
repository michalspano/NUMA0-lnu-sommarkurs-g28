#!/usr/bin/env python3

"""
File:        main.py
Authors:     Michal Spano
Description: Source code for HW1 of Summer Course 'NUMA0'.
             The provided source code is documented with the help of String
             Literals. It has been 'manually' tested as per the requirements.
             See the attached JupyterNotebook to interact with the code.
"""

from numpy import sqrt, log
import matplotlib.pyplot as plt


def approx_ln(x: float, n: int) -> float:
    """
    Task 1:
    A function that approximates a logarithm by n steps based on the described
    algorithm. The returned type is a decimal value.
    """
    if x <= 0:
        raise ValueError("The input value must be greater than 0.")
    
    # Initialize the mean values
    a, g = (1 + x) / 2, sqrt(x)
    
    # Iterate in n+1 steps
    for _ in range(n + 1):
        a = (a + g) / 2
        g = sqrt(a * g)
    
    # Return the computed approximation
    return (x - 1) / a


def plot_func(diff_mode: bool = False) -> None:
    """
    Task 2:
    This helper function is used to plot `approx_ln` against and its
    established implementation (from the `numpy` library) `log`.
    """
    ns = [1, 4, 7, 9, 12, 15, 20, 100]
    x_range = 500
    
    # Show expected ln(x).
    ln_ys = [log(x) for x in range(1, x_range)]
    if not diff_mode:
        plt.plot(ln_ys, linewidth=2, label=f"$\\ln({x_range})$",
                 linestyle="dashed", color="blue")
    
    # Compute for different n values. Compare against ln(x).
    for n in ns:
        # approx_ln_ys = [approx_ln(x, n) for x in range(1, x_range)]
        approx_ln_ys = [approx_ln(x, n) for x in range(1, x_range)]
        delta_y      = [abs(a - b) for a, b in zip(approx_ln_ys, ln_ys)]

        if not diff_mode:
            plt.plot(approx_ln_ys, linewidth=1.5,
                     label=f"approx_ln({x_range},{n})")
        else:
            plt.plot(delta_y, linewidth=1.5, linestyle="dashed",
                     label=f"$\\delta$, step={n}")

    plt.xlabel("x")

    if not diff_mode:
        plt.ylabel("$\\ln(x)$")
        plt.title("Task 2 - Plot")
    else:
        plt.ylabel("$\\delta$")
        plt.title("Task 2 - Difference")

    plt.legend(loc='best')
    plt.show()


def task3() -> None:
    """
    In this example, we let x = 1.41. We then proceed to plot the absolute
    value of the error (delta) against n. We let n be 1, 10, ..., 100. This
    graph is represented with a dotted line where the x values represent the
    individual steps (n) and y represent the delta.

    Note: a function is used to perform this task for the sake of clarity.
    """
    x = 1.41
    ns = [1] + [i for i in range(10, 110, 10)] # 1, 10, 20, ..., 100

    approx_ln_ys = [approx_ln(x, n) for n in ns]
    delta_y      = [abs(y - log(x)) for y in approx_ln_ys]

    plt.plot(ns, delta_y, "go-")
    plt.xlabel("steps")
    plt.ylabel("abs error ($\\delta$)")
    plt.show()


def fast_approx_ln(x: float, n: int) -> float:
    """
    Task 4: this function implements the raid method from the article to
    accelerate the convergence. It improves the initially developed algorithm
    to approximate ln(x).
    """
    if x <= 0:
        raise ValueError("The input value must be greater than 0.")
    
    # Initial mean values
    a, g = (1 + x) / 2, sqrt(x)
    
    # Initialize '(n+1) x (n+1)' array called `d`
    d = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    
    # Iterate in n+1 steps, instantiate d_{0,i}
    for i in range(n + 1):
        d[0][i] = a
        a = (a + g) / 2
        g = sqrt(a * g)
    
    # Compute remaining d_{k,i} s.t. k = 1..i, whenever i > 0.
    for i in range(1, n + 1):
        for k in range(1, i + 1):
            d[k][i] = (d[k - 1][i] - 4**(-k) * d[k - 1][i-1]) / (1 - 4**(-k))
    
    # An approximation to ln(x) is taken as the following
    return (x - 1) / d[n][n]


def accelerated_log_plot() -> None:
    """
    Task 5: replicate the plot provided in the homework using the
    `fast_approx_ln` method.
    """
    x_range = 20               # from the plot
    xs = range(1, x_range + 1) # from the plot
    colors = ["blue", "orange", "green", "red", "purple"] # exact colors from the plot
    
    for i in range(2, 7): # iterations 2..6
        fast_approx_ln_ys = [fast_approx_ln(x, i) for x in xs]
        delta_y = [abs(y - log(x)) for x, y in zip(xs, fast_approx_ln_ys)]
        plt.plot(delta_y, linewidth=2.5, color=colors[i-2],
                 label=f"iteration {i}")
    
    plt.title("Error behavior of the accelerated Carlsson method for the $\\log$")
    plt.xlabel("x")
    plt.ylabel("error")
    plt.legend(loc="upper left")
    plt.show()


# Entry point of the program
if __name__ == "__main__":
    # plot_func()
    # plot_func(diff_mode=True)
    # task3()
    pass

