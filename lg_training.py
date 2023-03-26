from data_processing import *
from logistic_regression import *

"""
run with initialize to save binary files required for training
the first time processing and saving data is ~10 minutes
this reduces future testing and training time by a factor of 10 
!!! IMPORTANT before running, create an empty directory 'binaries' in working dir
"""

initialize = True
save_results = False
train_csv_path = ''

# total samples in train set, remainder in the test
train_size = 9_600

# saves a test train split in numpy and scipy binaries
if initialize:
    initialize_for_training(train_csv_path=train_csv_path, train_size=train_size)

x_t_alt = sparse.load_npz('/Users/estefan/Desktop/artifacts/binaries/sparse_norm_p_ex.npz')
x_train, x_test = sparse.load_npz('binaries/x_train.npz'), sparse.load_npz('binaries/x_test.npz')
y_train, y_test = np.load('binaries/y_train.npy'), np.load('binaries/y_test.npy')
train_lambda, train_eta, train_epsilon = 0.01, 0.01, 0.00001
max_iter = 1_000

classes = get_classes()
saved_weight_file = f'W_{train_eta}_{train_lambda}_{max_iter}.npy'

w, acc, iters = lg_fit_sparse(learning_rate=train_eta, penalty_term=train_lambda,
                              max_iterations=max_iter, x_train=x_train, y_train=y_train,
                              classes=classes, epsilon_termination=True,
                              x_test=x_test, y_test=y_test, epsilon=train_epsilon, verbose=True)

print(f'hyper parameters: lr={train_eta}, pen={train_lambda}, epsilon={train_epsilon}')
print(f'final accuracy in {iters} iterations: {acc}')

if save_results:
    np.save(saved_weight_file, w)
    print(f'file {saved_weight_file} saved')
