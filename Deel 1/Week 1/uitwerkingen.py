import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def draw_graph(data: np.ndarray) -> None:
    """
    Plots a scatter plot of the data in the first column of the given matrix on
    the x-axis and the data in the second column on the y-axis. The first
    column is rotated to the horizontal axis.

    :param data: the data to plot
    :return: None
    """
    x: np.ndarray = data[:, [0]]
    y: np.ndarray = data[:, [1]]
    plt.scatter(x, y)
    plt.show()
    return None


def compute_cost(X: np.ndarray, y: np.ndarray, θ: np.ndarray) -> float:
    """
    Computes the current cost of theta given X and y.
    Theta represents the current parameters, X represents the input vector and
    y represents the actual output values.

    :param X: the input vector
    :param y: the actual output values
    :param θ: the current parameters

    :return: the current cost of theta given X and y
    """
    J: float = 0.0
    length: int = X.shape[0]  # Because X is a matrix
    for i in range(length):
        J += (np.dot(X[i], θ) - y[i]) ** 2
    J /= 2 * length
    return J


def gradient_descent(X: np.ndarray,
                     y: np.ndarray,
                     θ: np.ndarray,
                     α: float,
                     num_iters: int) -> tuple[np.ndarray, list]:
    """
    Performs gradient descent to learn theta by taking num_iters gradient steps
    with learning rate alpha.

    :param X: the input vector
    :param y: the actual output values
    :param θ: the current parameters
    :param alpha: the learning rate
    :param num_iters: the number of iterations

    :return: the learned parameters [θ, costs]
    """
    m: int = X.shape[0]
    costs: list = []
    for i in range(1, num_iters):
        pred = (X @ θ.T) - y  # Predict by multiplying X with θ
        θ = θ - (α / m) * (pred.T @ X)  # Doesn't matter what this means
        costs.append(compute_cost(X, y, θ.T))  # Add the cost to the list
    return θ, costs


def draw_costs(data) -> None:
    """
    Draws a graph of the costs over the iterations.
    """
    plt.plot(data)
    plt.show()
    return None


def contour_plot(X: np.ndarray, y: np.ndarray) -> None:
    """
    Plots a contour plot of the cost function J(θ) over a grid of values for
    θ0 and θ1.

    :param X: the input vector
    :param y: the actual output values

    :return: None
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    t1 = np.linspace(-10, 10, 100)
    t2 = np.linspace(-1, 4, 100)
    T1, T2 = np.meshgrid(t1, t2)

    J_vals = np.zeros((len(t1), len(t2)))

    for i in range(len(t1)):
        for j in range(len(t2)):
            θ = np.array([t1[i], t2[j]])
            J_vals[i, j] = compute_cost(X, y, θ)

    ax.plot_surface(T1, T2, J_vals, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel(r'$\theta_0$', linespacing=3.2)
    ax.set_ylabel(r'$\theta_1$', linespacing=3.1)
    ax.set_zlabel(r'$J(\theta_0, \theta_1)$', linespacing=3.4)

    ax.dist = 10

    plt.show()
