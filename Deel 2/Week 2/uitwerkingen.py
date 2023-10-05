import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix


def plot_number(nr_vector: np.ndarray) -> None:
    """
    Plots the given vector as a 20x20 matrix. Makes it pink just because.

    :param nr_vector: the vector to plot
    :return: None
    """
    data = np.reshape(nr_vector, (20, 20), order='F')
    plt.matshow(data, cmap='PuRd')
    plt.show()
    return None


def sigmoid(z: np.matrix) -> np.matrix:
    """
    Computes the sigmoid of z. Uses the usual 1 / (1 + e^-z) formula.

    :param z: the input

    :return: the sigmoid of z
    """
    return 1 / (1 + np.exp(-z))


def get_y_matrix(y: np.ndarray, m: int) -> np.matrix:
    """
    Converts the given vector y to a matrix.

    :param y: the vector to convert
    :param m: the number of rows in the matrix

    :return: the matrix
    """
    y = y[:, 0] - 1  # Get only the first column and remove 1 from each value.
    rows = np.arange(m)  # Create an array with values from 0 to m - 1.
    data = np.ones(m)  # Create an array with m 1s.
    return csr_matrix((data, (rows, y))).toarray()  # Create a sparse matrix
    # with 1s at the correct
    # positions.


def predict_number(θ1: np.ndarray, θ2: np.ndarray, X: np.ndarray):
    """
    Returns a matrix with the output of the network given the values of
    θ1 and θ2. Each row in this matrix is the probability that the sample
    at that position (i) is the number that corresponds to the column.

    The matrices θ1 and θ2 correspond to the weights between the input
    layer and the hidden layer, and between the hidden layer and the
    output layer, respectively.

    :param θ1: the weights between the input layer and the hidden layer
    :param θ2: the weights between the hidden layer and the output layer
    :param X: the input matrix

    :return: the output of the network at the outer layer
    """
    a1 = np.c_[np.ones(X.shape[0]), X]  # Add ones to the input matrix.
    a2 = sigmoid(a1 @ θ1.T)  # Calculate the output of the hidden layer.
    a2 = np.c_[np.ones(a2.shape[0]), a2]  # Add ones to the hidden layer.
    a3 = sigmoid(a2 @ θ2.T)  # Calculate the output of the outer layer.
    return a3


def compute_cost(θ1: np.ndarray, θ2: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the cost of the network given the values of θ1 and θ2 using
    the given input matrix X and the given output vector y.

    :param θ1: the weights between the input layer and the hidden layer
    :param θ2: the weights between the hidden layer and the output layer
    :param X: the input matrix

    :return: the cost of the network
    """
    m = X.shape[0]
    y_matrix = get_y_matrix(y, m)  # Convert the output vector to a matrix.
    # Calculate the output of the network.
    J = predict_number(θ1, θ2, X)
    cost = np.sum(-y_matrix * np.log(J) - (1 - y_matrix) * np.log(1 - J))  # ??
    return cost / m  # Return the average cost.


def sigmoid_gradient(z: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the sigmoid function at z.

    :param z: the input

    :return: the gradient of the sigmoid function at z
    """
    return sigmoid(z) * (1 - sigmoid(z))


def nn_check_gradients(θ1: np.ndarray, θ2: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Backpropogates the network to calculate the gradients of the cost
    function with respect to θ1 and θ2. Then checks whether these
    gradients are correct by comparing them to the gradients calculated
    using the finite difference method.

    NOTE: really damn slow, don't panic if it takes a while.

    :param θ1: the weights between the input layer and the hidden layer
    :param θ2: the weights between the hidden layer and the output layer
    :param X: the input matrix
    :param y: the output vector

    :return: a tuple containing the gradients of the cost function with
    respect to θ1 and θ2
    """
    Δ2 = np.zeros(θ1.shape)
    Δ3 = np.zeros(θ2.shape)
    m: int = X.shape[0]

    for i in range(m):
        # === Step 1 ===
        # predict_number(θ1, θ2, X[i, :])  # Calculate the output of the network.
        a1 = np.c_[np.ones(X.shape[0]), X]  # Add ones to the input matrix.
        z2 = a1 @ θ1.T
        a2 = sigmoid(z2)  # Calculate the output of the hidden layer.
        a2 = np.c_[np.ones(a2.shape[0]), a2]  # Add ones to the hidden layer.
        z3 = a2 @ θ2.T
        a3 = sigmoid(z3)  # Calculate the output of the outer layer.

        # === Step 2 ===
        # For each output unit k in layer 3 (the output layer), set
        # δ3 = (a3 - y) where y is the expected output vector.
        δ3 = a3 - y[i, :]

        # === Step 3 ===
        # For the hidden layer, set δ2 = Θ2T * δ3 .* g'(z2) where g'(z2)
        # is the gradient of the sigmoid function at z2.
        δ2 = (δ3 @ θ2)[:, 1:] * sigmoid_gradient(z2)

        # === Step 4 ===
        # Accumulate the gradient from this example using the following
        # formula: Δ(l) = Δ(l) + δ(l+1) * a(l)T
        Δ2 += δ2.T @ a1
        Δ3 += δ3.T @ a2
    # === Step 5 ===
    # Obtain the unregularized gradient by dividing the accumulated
    # gradients by m.

    Δ2_grad = Δ2 / m
    Δ3_grad = Δ3 / m

    return Δ2_grad, Δ3_grad