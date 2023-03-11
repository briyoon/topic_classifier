from data_processing import *
from logistic_regression import *

run_initialize = False

# this is set to run with numpy binaries for the test set and sample ids
test_file_path = ''
test_sample_path = ''
test_id_path = ''

if run_initialize:
    save_test_data_bin(test_path=test_file_path)
    test_sample_path = 'test_examples.npy'
    test_id_path = 'test_ids.npy'

index = 4
weight_ind = 37

# could load x_test and ids from a csv here
classes = get_classes()
x_test = np.load(test_sample_path)
ids = np.load(test_id_path)
weights = np.load(f'WEIGHTS_{weight_ind}.npy')

lg_gen_predictions(pred_file_name=f'pred_{index}.csv', x=x_test, w=weights, c=classes, ids=ids)
