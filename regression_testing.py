from data_processing import *
from logistic_regression import *

train_path = "/Users/estefan/Desktop/cs429529-project-2-topic-categorization/training.csv"

penalty_term = 0.01
learning_rate = 0.01
total_iterations = 1_000

# can specify rows and label path
[x, y, c] = get_training_data(training_path=train_path, rows=1_000)

x_train = x
y_train = y

x_test = x
y_test = y

[w, t] = lg_fit(learning_rate=learning_rate,
                penalty=penalty_term,
                max_iterations=total_iterations,
                examples=x_train, target=y_train, classes=c,
                show_time=True)

[acc, confusion_mat] = lg_test(x_test, y_test, w, c)
result_str = get_result_str(learning_rate=learning_rate,
                            penalty=penalty_term,
                            iterations=total_iterations,
                            accuracy=acc,
                            test_size=len(y_test),
                            train_size=len(y_train),
                            t=t)

# can ignore weights or confusion matrix, default is none
record_test_result(result_str, w, confusion_mat)
