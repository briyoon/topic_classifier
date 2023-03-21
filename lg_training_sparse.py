import math

from data_processing import *
from logistic_regression import *
import numpy.typing as npt


def log_loss(y_pred: npt.NDArray, y_actual: npt.NDArray):
    log_cost = 0
    for col in range(0, len(y_actual)):
        actual_class = int(y_actual[col])
        class_index = actual_class - 1
        predicted_prob = y_pred[class_index][col]
        temp = 0 if predicted_prob <= 0 else math.log(predicted_prob)
        log_cost += temp
    return -(log_cost / len(y_actual))


ex_bin_path = '/Users/estefan/Desktop/cs429529-project-2-topic-categorization/examples.npy'
ex_bin_class_path = '/Users/estefan/Desktop/cs429529-project-2-topic-categorization/example_classes.npy'
ex_sparse_path = 'binaries/sparse_norm_p_ex.npz'
ex_sparse_trans_path = 'binaries/sparse_norm_p_ex_trans.npz'

[x, x_t] = get_sparse_train_data(x_path=ex_sparse_path, x_trans_path=ex_sparse_trans_path)
c = get_classes()
y = np.load(ex_bin_class_path)
delta = get_delta_matrix(c, y)
penalties = [0.001]  # [0.0075, 0.005, 0.0025, 0.001]
learning_rates = [0.01]  # [0.005, 0.0025, 0.001]
k = len(c)
n = len(x.toarray()[0])

for learning_rate in learning_rates:
    for penalty_term in penalties:
        iterations = 0
        w = np.random.rand(k, n)
        start_loss = 20
        end_loss = 0
        for i in range(22_000, 30_000, 1_000):
            epoch_start = time.time()
            for j in range(1, 1_001):
                prb = prob_mat(x_t, w)
                error_mat = np.subtract(delta, prb)
                loss = log_loss(y_pred=prb, y_actual=y)
                max_err = np.max(error_mat)
                error_mat = error_mat @ x
                update = learning_rate * (np.add(error_mat, -penalty_term * w))
                max_update = np.max(update)
                w = np.add(w, update)
                print(f'{iterations}. lr={learning_rate}, loss={loss}, max_err={max_err},')
                iterations += 1
                end_loss = loss

            epoch_end = time.time()
            time_1000 = epoch_end - epoch_start
            result_str = get_result_str(learning_rate=learning_rate,
                                        penalty=penalty_term,
                                        iterations=i,
                                        accuracy=0,
                                        test_size=len(y),
                                        train_size=len(y),
                                        t=time_1000)

            if abs(end_loss - start_loss) < 0.00001:
                learning_rate = 0.001
            start_loss = end_loss
            record_test_result(result_str, w, conf=None, notes='abs-max scaled data')
