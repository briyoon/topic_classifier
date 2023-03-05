import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(matrix):
    # hard coded for our classes from 1 to 20
    labels = list(range(1, 21))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # show matrix, use colorbar for legend
    cax = ax.matshow(matrix, cmap=matplotlib.colormaps['Oranges'], alpha=0.95, interpolation='nearest')
    fig.colorbar(cax)

    axis_range = np.arange(len(labels))
    ax.xaxis.set_label_position('top')
    ax.set_xticks(axis_range)
    ax.set_yticks(axis_range)

    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.xaxis.tick_top()

    plt.xlabel('Actual', fontsize=20)
    plt.ylabel('Predicted', fontsize=20)
    plt.show()


def plot_average_class_weights(weights):
    x = range(0, len(weights[0]))
    plt.scatter(x, weights[0], alpha=0.3)


conf_mat_path = '/Users/estefangonzales/Desktop/CourseWork/2023/CS429/P2-NBLG/confusion_matrices'
path = f'{conf_mat_path}/CMAT_20.npy'
data = np.load(path)
plot_confusion_matrix(data)
