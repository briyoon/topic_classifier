import time
from datetime import datetime
import pandas as pd
from logistic_regression import *
from data_processing import *

train_path = "/Users/estefan/Desktop/cs429529-project-2-topic-categorization/training.csv"

penalty_term = 0.01
learning_rate = 0.01
total_iterations = 1_000

# can specify rows and label path
[x, y, c] = get_training_data(training_path=train_path)

[w, t] = lg_fit(learning_rate=learning_rate,
                penalty=penalty_term,
                max_iterations=total_iterations,
                examples=x, target=y, classes=c,
                show_time=True)

x_test = x[3_000:5_000, :]
y_test = y[3_000:5_000, :]
[acc, confusion_mat] = lg_test(x_test, y_test, w, c)
result_str = get_result_str(learning_rate=learning_rate,
                            penalty=penalty_term,
                            iterations=total_iterations,
                            accuracy=acc,
                            test_size=len(x),
                            train_size=len(x_test),
                            t=t)

# can ignore w, default is none
record_test_result(result_str, w)
