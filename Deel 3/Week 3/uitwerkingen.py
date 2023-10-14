import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


# OPGAVE 1a
def plot_image(img: np.ndarray, label: str) -> None:
    """
    Plots an image with the given label.

    :param img: A 1D array of 784 pixels.
    :param label: The label of the image.

    :return: None
    """
    data = np.reshape(img, (28, 28), order='F')
    plt.matshow(data, cmap='PuRd')
    plt.title(label)
    plt.show()


# OPGAVE 1b
def scale_data(X: np.ndarray) -> np.ndarray:
    """
    Transform the given data to the range [0, 1].

    :param X: A matrix of data of which the values are in the range [0, 1024].
    """
    return X / np.max(X)


# OPGAVE 1c
def build_model():
    """
    Creates a keras.Sequential model with the following layers:
    - Flatten: Flattens the input to a 1D array.
    - Dense: A fully connected layer with 128 nodes and relu activation.
    - Dense: A fully connected layer with 10 nodes and softmax activation.

    :return: A compiled keras.Sequential model.
    """

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# OPGAVE 2a
def conf_matrix(labels, pred):
    return tf.math.confusion_matrix(labels, pred)


# OPGAVE 2b
def conf_els(conf, labels):
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)

    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html

    # YOUR CODE HERE
    pass

# OPGAVE 2c


def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    # VERVANG ONDERSTAANDE REGELS MET JE EIGEN CODE

    tp = 1
    fp = 1
    fn = 1
    tn = 1

    # BEREKEN HIERONDER DE JUISTE METRIEKEN EN RETOURNEER DIE
    # ALS EEN DICTIONARY

    rv = {'tpr': 0, 'ppv': 0, 'tnr': 0, 'fpr': 0}
    return rv
