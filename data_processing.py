from datetime import datetime

import pandas as pd
import numpy as np
import re

COLUMN_WISE_AXIS = 1
TOTAL_CLASSES = 20
LABEL_PATH = 'resources/newsgrouplabels.txt'
TEST_INDEX_PATH = 'resources/DONOTMODIFY/test_index.txt'
TEST_RESULT_PATH = 'results/results.txt'


def get_test_index():
    file = open(TEST_INDEX_PATH)
    res = int(file.read())
    file.close()
    return res


def incr_test_index():
    index = get_test_index()
    file = open(TEST_INDEX_PATH, 'w')
    index = index + 1
    file.write(str(index))
    file.close()


def get_classes(path: str):
    cls = list()
    with open(path) as f:
        for line in f:
            s = re.search(r'\d+', line)
            if s:
                cls.append(int(s.group()))
    f.close()
    return np.array(cls)


def get_training_data(training_path: str, label_path=LABEL_PATH, rows=-1):
    if rows == -1:
        data = pd.read_csv(training_path, header=None)
    else:
        data = pd.read_csv(training_path, nrows=rows, header=None)

    x = data.to_numpy(dtype='float', na_value=np.NAN)
    set_len = len(x[0])

    # last column of classes
    y = x[:, set_len - 1]

    # get rid of first and last columns
    x = np.delete(x, axis=COLUMN_WISE_AXIS, obj=0)
    x = np.delete(x, axis=COLUMN_WISE_AXIS, obj=set_len - 2)

    c = get_classes(label_path)

    return [x, y, c]


def get_result_str(learning_rate, penalty, iterations, accuracy, train_size, test_size, t=0):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    res = f'{date} rate={learning_rate},' \
          f' penalty={penalty},' \
          f' iter={iterations},' \
          f' acc={round(accuracy, 2)},' \
          f' t={round(t, 2)}s, train={train_size}, test={test_size}'
    return res


def record_test_result(result: str, w=None, conf=None):
    test_index = get_test_index()
    test_record = str(test_index)
    test_record += ' - '
    test_record += result
    test_record += '\n'
    file = open(TEST_RESULT_PATH, 'a')
    file.write(test_record)
    file.close()

    if w is not None:
        np.save(f'results/WEIGHTS_{test_index}.npy', w)

    if conf is not None:
        np.save(f'results/CMAT_{test_index}.npy', conf)

    incr_test_index()


def get_saved_weight(path: str, csv=False):
    if csv:
        np_arr = np.loadtxt(path, dtype=np.float64, delimiter=',')
    else:
        np_arr = np.load(path)

    return np_arr
