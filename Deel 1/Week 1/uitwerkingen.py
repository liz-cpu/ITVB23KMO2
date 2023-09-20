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


def gradient_descent(X, y, theta, alpha, num_iters):
    #OPGAVE 3a
    # In deze opgave wordt elke parameter van theta num_iter keer geüpdate om de optimale waarden
    # voor deze parameters te vinden. Per iteratie moet je alle parameters van theta update.

    # Elke parameter van theta wordt verminderd met de som van de fout van alle datapunten
    # vermenigvuldigd met het datapunt zelf (zie Blackboard voor de formule die hierbij hoort).
    # Deze som zelf wordt nog vermenigvuldigd met de 'learning rate' alpha.

    # Een mogelijk stappenplan zou zijn:
    #
    # Voor elke iteratie van 1 tot num_iters:
    #   1. bepaal de voorspelling voor het datapunt, gegeven de huidige waarde van theta
    #   2. bepaal het verschil tussen deze voorspelling en de werkelijke waarde
    #   3. vermenigvuldig dit verschil met de i-de waarde van X
    #   4. update de i-de parameter van theta, namelijk door deze te verminderen met
    #      alpha keer het gemiddelde van de som van de vermenigvuldiging uit 3

    m,n = X.shape
    costs = []

    # YOUR CODE HERE

    # aan het eind van deze loop retourneren we de nieuwe waarde van theta
    # (wat is de dimensionaliteit van theta op dit moment?).

    return theta, costs


def draw_costs(data): 
    # OPGAVE 3b
    # YOUR CODE HERE
    pass

def contour_plot(X, y):
    #OPGAVE 4
    # Deze methode tekent een contour plot voor verschillende waarden van theta_0 en theta_1.
    # De infrastructuur en algemene opzet is al gegeven; het enige wat je hoeft te doen is 
    # de matrix J_vals vullen met waarden die je berekent aan de hand van de methode computeCost,
    # die je hierboven hebt gemaakt.
    # Je moet hiervoor door de waarden van t1 en t2 itereren, en deze waarden in een ndarray
    # zetten. Deze ndarray kun je vervolgens meesturen aan de functie computeCost. Bedenk of je nog een
    # transformatie moet toepassen of niet. Let op: je moet computeCost zelf *niet* aanpassen.

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    jet = plt.get_cmap('jet')

    t1 = np.linspace(-10, 10, 100)
    t2 = np.linspace(-1, 4, 100)
    T1, T2 = np.meshgrid(t1, t2)

    J_vals = np.zeros( (len(t2), len(t2)) )

    #YOUR CODE HERE 

    surf = ax.plot_surface(T1, T2, J_vals, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    xLabel = ax.set_xlabel(r'$\theta_0$', linespacing=3.2)
    yLabel = ax.set_ylabel(r'$\theta_1$', linespacing=3.1)
    zLabel = ax.set_zlabel(r'$J(\theta_0, \theta_1)$', linespacing=3.4)

    ax.dist = 10

    plt.show()
