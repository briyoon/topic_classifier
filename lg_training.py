from data_processing import *
from logistic_regression import *

# set to true to save numpy binaries from training data
# if you already have them saved replaces the paths for ex_bin_...
run_initialize = False

path_to_training_csv = ''
ex_bin_path = '/Users/estefan/Desktop/cs429529-project-2-topic-categorization/examples.npy'
ex_bin_class_path = '/Users/estefan/Desktop/cs429529-project-2-topic-categorization/example_classes.npy'

if run_initialize:
    # save the csv as numpy binaries
    save_training_data_bin(example_path=path_to_training_csv)

    # default file names, can change in call to save_training_data_bin
    ex_bin_path = 'sample_numpy_bin.npy'
    ex_bin_class_path = 'sample_class_numpy_bin.npy'

# if you don't want to use numpy binaries:
# replace call with get_training_data(path_to_training_csv)
[x, y, c] = get_training_data_bin(example_path=ex_bin_path, example_class_path=ex_bin_class_path)

# check results.txt for completed training params
# saved weights and confusion matrices are in 'results' directory
penalties = [0.0075, 0.005, 0.0025, 0.001]
learning_rates = [0.0075, 0.005, 0.0025, 0.001]
max_train_iterations = 5_000
k = len(c)
n = len(x[0])
for learning_rate in learning_rates:
    for penalty_term in penalties:
        iterations = 1_000
        w = np.random.rand(k, n + 1)
        while iterations <= max_train_iterations:
            [w, t] = lg_fit(learning_rate=learning_rate,
                            penalty=penalty_term,
                            max_iterations=1_000,
                            examples=x, target=y, classes=c, w=w)
            [acc, confusion_mat] = lg_test(x, y, w, c)
            result_str = get_result_str(learning_rate=learning_rate,
                                        penalty=penalty_term,
                                        iterations=iterations,
                                        accuracy=acc,
                                        test_size=len(y),
                                        train_size=len(y),
                                        t=t)
            # will save the resulting weights and confusion mat + add entry to results.txt
            record_test_result(result_str, w, confusion_mat, notes='full dataset, for final submission')
            iterations += 1_000


