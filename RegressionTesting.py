import time
from datetime import datetime

import pandas as pd

from LogisticRegression import *

COLUMN_WISE_AXIS = 1

path = "/Users/estefan/Desktop/cs429529-project-2-topic-categorization/training.csv"
examples_taken = 10_000

start = time.time()
df = pd.read_csv(path, delimiter=',')
end = time.time()

print(f'read_time={end - start}s')

np_arr = df.to_numpy(dtype='float', na_value=np.NAN)
num_features = len(np_arr[0]) - 2
y = np_arr[:, len(np_arr[0]) - 1]
classes = np.unique(y)
x = np.delete(np_arr, axis=COLUMN_WISE_AXIS, obj=0)  # delete the doc id col
x = np.delete(x, axis=COLUMN_WISE_AXIS, obj=num_features)  # delete the class col
x = get_padded_examples(x)
w = np.random.rand(len(classes), num_features + 1)
delta = get_delta_matrix(class_list=classes, example_classes=y)
prob_mat = get_prob_matrix(samples=x, weights=w)
print(f'time_after_matrix_processing={time.time() - start}s')

penalty_term = 0.01
learning_rate = 0.01
total_iterations = 1_000
iterations = 0
while iterations < total_iterations:
    if iterations % (total_iterations / 100) == 0:
        print('.', end="")
    prob_mat = get_prob_matrix(samples=x, weights=w)
    error_mat = np.add(delta, (-1 * prob_mat))
    sample_error_mat = np.matmul(error_mat, x)
    penalty_mat = -penalty_term * w
    weight_update_mat = learning_rate * (np.add(sample_error_mat, penalty_mat))
    w = np.add(w, weight_update_mat)
    iterations += 1

total = len(y)
correct = 0
for sample_ind in range(0, len(y)):
    argmax_prob = max(prob_mat[:, sample_ind])
    predicted = -1
    for class_ind in range(0, len(prob_mat)):
        if prob_mat[class_ind][sample_ind] == argmax_prob:
            predicted = classes[class_ind]
            break
    if predicted == y[sample_ind]:
        correct += 1

version = 1
date = datetime.today().strftime('%Y-%m-%d')
np.savetxt(f'ClassifierWeightsV{version}_'
           f'{date}_a={learning_rate}_'
           f'p={penalty_term}_iter={iterations}.csv',
           w, delimiter=',')
print(f'accuracy={correct / total}')
print(f'total_time_{total_iterations}_iterations={time.time() - start}')
