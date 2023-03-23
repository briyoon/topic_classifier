from data_processing import *
from logistic_regression import *

run_initialize = False

# this is set to run with numpy binaries for the test set and sample ids
test_file_path = '.../testing.csv'
test_sample_path = 'binaries/test_examples_norm.npy'
test_id_path = 'binaries/test_ids.npy'

if run_initialize:
    # normalized test data is default, change with 'normalized=False'
    save_test_data_bin(test_csv_path=test_file_path)
    test_sample_path = 'binaries/test_examples.npy'
    test_id_path = 'binaries/test_ids.npy'

submission_index = 86
weight_index = 172

# could load x_test and ids from a csv here
classes = get_classes()
x_test = np.load(test_sample_path)
ids = np.load(test_id_path)
for i in range(0, 1):
    weights = np.load(f'/Users/estefan/Desktop/final/WEIGHTS_{weight_index}.npy')
    lg_gen_predictions(pred_file_name=f'pred_{submission_index}_{weight_index}.csv',
                       x=x_test, w=weights, c=classes, ids=ids)
    submission_index += 1
    weight_index += 1



