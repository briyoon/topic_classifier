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


def feature_scatter_plot(data, x_label, y_label, num_rows=1):
    x = range(0, len(data[0]))
    for row in range(0, num_rows):
        plt.scatter(x, data[row], alpha=0.3, s=4)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.show()


# k = num classes, n = num features + 1, start = first test index, end = last test index
def get_weight_sum(weight_path, k, n, start, end):
    summation = np.zeros(shape=(k, n))
    for test in range(start, end + 1):
        weights = np.load(f'{weight_path}/WEIGHTS_{test}.npy')
        summation = np.add(weights, summation)
    return summation


# returns mean and variance numpy arrays for the weights
# will do so for tests from start to end (based on indices)
# should change to list of test indices
def get_weight_statistics(weight_path, k, n, start, end):
    size = end - start
    norm = 1 / (size + 1)
    mean = get_weight_sum(weight_path, k, n, start, end)
    mean *= norm
    variance = np.zeros(shape=(k, n))
    for test in range(start, end + 1):
        weights = np.load(f'{weight_path}/WEIGHTS_{test}.npy')
        diff = np.add(weights, -1 * mean)
        variance = np.add(variance, np.square(diff))
    variance *= (1 / size)
    return [mean, variance]


weight_mat_path = '/Users/estefangonzales/Desktop/CourseWork/2023/CS429/P2-NBLG/weights'
rows = 20
cols = 61189
first_test = 1
last_test = 10

# results from the varied learning rate tests
[lr_mean, lr_var] = get_weight_statistics(weight_mat_path, rows, cols, first_test, last_test)
lr_std_dev = np.sqrt(lr_var)
feature_scatter_plot(lr_std_dev, 'Feature', 'StdDev', num_rows=1)

