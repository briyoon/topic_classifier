import time

import numpy as np
import numpy.typing as npt


# add column of ones to left of examples
def get_padded_examples(training_examples):
    num_ex = len(training_examples)
    padding = np.ones(shape=(num_ex, 1))
    padded_mat = np.append(padding, training_examples, axis=1)
    return padded_mat


def get_prob_matrix(samples, weights):
    k = len(weights)
    m = len(samples)
    x_transpose = samples.transpose()

    # WX is k x m matrix
    p_mat = np.matmul(weights, x_transpose)

    # set last row to 1's
    p_mat[k - 1] = np.ones(shape=(1, m))
    for i in range(0, len(p_mat[0])):
        max_term = max(p_mat[:, i])
        p_mat[:, i] -= max_term

    # P[i][j] = exp(P[i][j])
    p_mat = np.vectorize(np.exp)(p_mat)

    # axis 0 -> cols, axis 1 -> rows
    column_sums = p_mat.sum(axis=0)

    # divide each column entry by the column sum, softmax
    for i in range(0, len(column_sums)):
        # max_wx = max(p_mat[])
        normal_term = 1 / column_sums[i]
        p_mat[:, i] *= normal_term

    return p_mat


def get_delta_matrix(class_list, example_classes):
    # total number of examples
    m = len(example_classes)
    # total number of classes
    k = len(class_list)
    delta_mat = np.zeros(shape=(k, m))

    # classes = [c_0, c_1,..., c_k] -> class_to_ind[c_i] = i
    class_to_ind = dict()
    for ind in range(0, k):
        class_to_ind[class_list[ind]] = ind

    # ∆[i][j] = 1 if y_j = c_i else 0
    for ind in range(0, m):
        delta_col = ind
        delta_row = class_to_ind[example_classes[ind]]
        delta_mat[delta_row][delta_col] = 1

    return delta_mat


# m = num examples, n = num features, k = num classes
# examples = X (m,n), target = Y (m,1), classes = K (k,1)
def lg_fit(learning_rate, penalty, max_iterations,
           examples: npt.NDArray,
           target: npt.NDArray,
           classes: any,
           show_time=False):
    k = len(classes)
    n = len(examples[0])

    start = time.time()
    x = get_padded_examples(examples)
    w = np.random.rand(k, n + 1)
    delta = get_delta_matrix(class_list=classes, example_classes=target)
    p_mat = get_prob_matrix(samples=x, weights=w)
    iterations = 0

    while iterations < max_iterations:
        if show_time and iterations % (max_iterations / 100) == 0:
            print('.', end="")
        p_mat = get_prob_matrix(samples=x, weights=w)
        error_mat = np.add(delta, (-1 * p_mat))
        sample_error_mat = np.matmul(error_mat, x)
        penalty_mat = -penalty * w
        weight_update_mat = learning_rate * (np.add(sample_error_mat, penalty_mat))
        w = np.add(w, weight_update_mat)
        iterations += 1

    end = time.time()
    if show_time:
        print('')

    return [w, end - start]


def lg_test(x_test: npt.NDArray,
            y_test: npt.NDArray,
            weights: npt.NDArray,
            classes: any):
    m = len(y_test)
    k = len(classes)

    if len(x_test[0]) != len(weights[0]):
        x_test = get_padded_examples(x_test)

    p_mat = get_prob_matrix(samples=x_test, weights=weights)
    confusion = np.zeros(shape=(k, k))
    correct = 0

    class_to_ind = dict()
    for ind in range(0, k):
        class_to_ind[classes[ind]] = ind

    for instance in range(0, m):
        predicted = -1
        max_prob = -1
        actual = y_test[instance]
        for cls in range(0, k):
            if p_mat[cls][instance] > max_prob:
                predicted = classes[cls]
                max_prob = p_mat[cls][instance]

        if predicted == -1:
            print(f'error, no max prediction found for class, instance={instance}')
            continue

        if predicted == actual:
            correct += 1
        else:
            confusion[class_to_ind[predicted]][class_to_ind[actual]] += 1

    accuracy = correct / m

    return [accuracy, confusion]


def lg_predict(x: npt.NDArray,
               weights: npt.NDArray,
               classes: any):
    m = len(x)
    k = len(classes)

    if len(x[0]) != len(weights[0]):
        x = get_padded_examples(x)

    p_mat = get_prob_matrix(samples=x, weights=weights)
    pred = np.ones(shape=(m, 1))

    for instance in range(0, m):
        predicted = -1
        max_prob = -1
        for cls in range(0, k):
            if p_mat[cls][instance] > max_prob:
                predicted = classes[cls]
                max_prob = p_mat[cls][instance]
        if predicted == -1:
            print(f'error with prediction for example {instance}')
            continue
        pred[instance] = predicted

    return pred


def lg_gen_predictions(pred_file_name: str, x: npt.NDArray, w: npt.NDArray, c: npt.NDArray):
    y = lg_predict(x=x, weights=w, classes=c)
    np.savetxt(fname=pred_file_name,
               X=y, delimiter=',',
               header='pred',
               fmt='%i', comments='')
