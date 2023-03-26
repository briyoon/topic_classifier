import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_weight_vector_dist(weight_path: str):
    train_w = np.load(weight_path)
    diff_mat = np.zeros(shape=(20, 20))
    for row in range(0, 20):
        vec_norm = np.linalg.norm(train_w[row])
        train_w[row] /= vec_norm

    for i in range(0, 20):
        for j in range(0, 20):
            if i == j:
                continue
            vec_i = train_w[i]
            vec_j = train_w[j]
            diff = np.sum(np.square(vec_i - vec_j))
            diff_mat[i][j] = np.sqrt(diff)

    for row in range(0, len(diff_mat)):
        dist_mean = np.sum(diff_mat[row])
        dist_mean /= 19
        diff_mat[row] *= -1
        diff_mat[row] += dist_mean
        diff_mat[row][row] = 0

    plot_confusion_matrix(diff_mat, x_label='Compared Weights', y_label='Reference Weights')


def plot_confusion_matrix(matrix, title=None, x_label=None, y_label=None):
    # hard coded for our classes from 1 to 20
    y_labels = list(range(1, len(matrix) + 1))
    x_labels = list(range(1, len(matrix[0]) + 1))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # show matrix, use colorbar for legend
    cax = ax.matshow(matrix, cmap=matplotlib.colormaps['Oranges'], alpha=0.8, interpolation='nearest')
    fig.colorbar(cax)

    y_axis_range = np.arange(len(y_labels))
    x_axis_range = np.arange(len(x_labels))

    ax.xaxis.set_label_position('top')
    ax.set_xticks(x_axis_range)
    ax.set_yticks(y_axis_range)

    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.xaxis.tick_top()

    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=20)
    else:
        plt.xlabel('Actual', fontsize=20)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=20)
    else:
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


def plot_hyper_param_3D():
    terms = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0001]
    rev_terms = terms
    rev_terms.reverse()
    size_map = dict()
    size_range = list(range(0, len(terms)))
    for i in range(0, len(rev_terms)):
        size_map[rev_terms[i]] = size_range[i]

    data = pd.read_csv('results/param_performance.txt', delimiter='\t')

    x = data['lr'].to_numpy()
    y = data['pen'].to_numpy()
    z = data['acc'].to_numpy()
    ticks = np.arange(0, len(terms), 1)
    ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
    ax.set(xlabel='$\\eta$', ylabel='$\\lambda$', zlabel='Accuracy')

    for i in range(0, len(x)):
        label = int(round(z[i] * 100, 2))
        ax.scatter(x[i], y[i], z[i], color='red', lw=0.5, alpha=0.5, s=5)
        ax.text(x[i], y[i], z[i], '%s' % label, size=6, zorder=5, color='k')
    ax.plot_trisurf(x, y, z, edgecolor='darkred', zorder=0, lw=0.5, alpha=0.8,
                    cmap=matplotlib.colormaps['Oranges'])

    plt.tight_layout()
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.view_init(30, 50)

    ax.set_xticklabels(terms)
    ax.set_yticklabels(terms)
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
