# Naive Bayes and

## Training and Testing
- Run `naive_bayes.py`

## Code
- `naive_bayes.py` contains utilities for training, testing, and producing prediction files. Prediction results are saved to the `results` subdirectory. Predictions for test data are saved to `results/nb_pred.csv`. Visualizations of the confusion matrices are saved to `results/nb_confusion_matrix.png`, and the accuracy plot for beta hyper-parameter are saved to `results/nb_a_b_accuracy.png`. The list of 100 most informative words for NB are found in `results/most_informative_words.txt`.
  - `nb_fit` fits the model to training data, given vectorized word counts `x`, and classes `y`.
  - `nb_predict` takes a document `x_new` and returns the predicted class. 
# Logistic Regression 

## Training and Testing
- Running for the first time:
  - add a `binaries` folder to the working directory
  - open `lg_training.py` and set `initialize = True` if not already 
  - update `train_csv_path` to be the path to your train.csv file
- After initialization
  - maintain or update the given file names for the binaries in the `binaries` directory
  - you can reinitialize to get a different test/train split by changing `train_size` in `lg_training`
  - to run with the entire train set just set `train_size = 12_000`
- Run `lg_training.py`, adjust hyper-parameters as desired in this file

## Code
- `data_processing.py` used to process and fetch training/testing files from csv and numpy binary format
  - `save_training_data_bin` produces numpy binary files for `x` and `y`
    - requires training data csv path, assumes csv format as specified in project
  - `save_test_data_bin` produces numpy binary files for `x` and `ids`
    - requires test data csv path, assumes csv format as specified in project
- `lg_training.py` will train on a given dataset for all combinations of learning rates 
and penalty terms for up to `max_train_iterations` saving the results at each step of 1,000. 
  - All saved results will be in the `results` directory with a record of the learning rate, penalty term,
  and other information saved to `results.txt`
  - Weights and the confusion matrices are saved as `.npy` files
  - This file primarily runs on numpy binaries, see the comments for how to switch to alternatives
- `logistic_regression.py` contains utilities for training, testing, and producing prediction files 
as specified in the project outline
  - `lg_fit` returns resulting training weights and time required to train
    - Parameters 
      - `learning_rate, penalty, max_iterations` - hyper parameters
      - `examples` (m,n) numpy array with m instances and n features 
      - `target` length m numpy array with m instances, algorithm assumes target array indices
      match example row indices 1:1
      - `classes` length k numpy array with k labels/classes for the example data 
      - `w` (k, n + 1) numpy array, optional param that if excluded is replaced by random 
      initial weights from 0 to 1 
    - `lg_test` returns model accuracy and confusion matrix
      - Runs with similar parameters to `lg_fit` but requires weights 
      - Is testing against known labels, need `x` and target `y`
    - `lg_predict` returns array of m labels for test instances
      - Generates predictions based on given weights for all m instances in `x`
      - Requires test instances `x`, weights `w`, and all possible labels/classes 
    - `lg_gen_predictions` produces csv file of the form `id,class`
      - Creates prediction csv per project outline, calls `lg_predict` so needs same arguments
      - Requires test `x` instance ids, assumes `x` and `ids` have the same ordering
