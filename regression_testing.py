from data_processing import *
from logistic_regression import *


# penalty_term = 0.15
ex_path = '/Users/estefan/Desktop/cs429529-project-2-topic-categorization/examples.npy'
ex_class_path = '/Users/estefan/Desktop/cs429529-project-2-topic-categorization/example_classes.npy'

[x, y, c] = get_training_data_bin(example_path=ex_path, example_class_path=ex_class_path)
epochs = [1_000, 2_000, 3_000, 4_000, 5_000]
penalties = [0.01, 0.0075, 0.005, 0.0025, 0.001]
learning_rates = [0.01, 0.0075, 0.005, 0.0025, 0.001]

for learning_rate in learning_rates:
    for penalty_term in penalties:
        for epoch in epochs:

            [w, t] = lg_fit(learning_rate=learning_rate,
                            penalty=penalty_term,
                            max_iterations=epoch,
                            examples=x, target=y, classes=c,
                            show_time=True)

            [acc, confusion_mat] = lg_test(x, y, w, c)
            result_str = get_result_str(learning_rate=learning_rate,
                                        penalty=penalty_term,
                                        iterations=epoch,
                                        accuracy=acc,
                                        test_size=len(y),
                                        train_size=len(y),
                                        t=t)

            # can ignore weights or confusion matrix, default is none
            record_test_result(result_str, w, confusion_mat, notes='full dataset, for final submission')

