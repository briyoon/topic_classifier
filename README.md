# Logistic Regression 

## Training and Testing
- Completed training is recorded in `results.txt` displaying covered hyper parameters
- Set hyper parameters in `lg_training.py`, configure training file paths and 
wait for training to conclude. Resulting weights, confusion matrices, and records are saved in
the results directory. 
- `lg_predictions.py` has an example of how to save the test data,`x` and `ids`, as numpy binaries.
  - Load `x`, `ids`, `weights`, and `classes` then call `lg_gen_predictions` with these arguments 
  and the prediction csv file name as in 
  `lg_gen_predictions('file.csv', x, w, classes, ids)`


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
