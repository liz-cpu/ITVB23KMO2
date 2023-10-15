import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


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


def scale_data(X: np.ndarray) -> np.ndarray:
    """
    Transform the given data to the range [0, 1].

    :param X: A matrix of data of which the values are in the range [0, 1024].
    """
    return X / np.max(X)


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


def conf_matrix(labels, pred):
    """
    Calculates the confusion matrix for the given labels and predictions.

    :param labels: The labels of the data.
    :param pred: The predictions of the data.
    """
    return tf.math.confusion_matrix(labels, pred)


def conf_els(conf: np.ndarray, labels: list[str]) -> list[tuple[str, int, int, int, int]]:
    """
    Calculates the true positives, false positives, false negatives and true negatives for each label.

    :param conf: The confusion matrix.
    :param labels: The labels of the confusion matrix.

    :return: A list of tuples containing the label, true positives, false positives, false negatives and true negatives.
    """
    tp = np.diagonal(conf)
    fp = np.sum(conf, axis=1) - tp
    fn = np.sum(conf, axis=0) - tp
    tn = np.sum(conf) - tp - fp - fn
    return [(labels[i], tp[i], fp[i], fn[i], tn[i]) for i in range(len(labels))]


def conf_data(metrics: list[tuple[str, int, int, int, int]]) -> dict[str, int]:

    tp = sum([m[1] for m in metrics])
    fp = sum([m[2] for m in metrics])
    fn = sum([m[3] for m in metrics])
    tn = sum([m[4] for m in metrics])

    tpr = tp / (tp + fn)
    ppv = tp / (tp + fp)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)

    rv = {'tpr': tpr, 'ppv': ppv, 'tnr': tnr, 'fpr': fpr}
    return rv
