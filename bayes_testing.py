import numpy as np
import pandas as pd
import csv
import pickle
import naive_bayes as nb

import os

# Read the csv file
training_df = pd.read_csv('resources/training.csv')
# testing_df = pd.read_csv('resources/testing.csv')

vocab_df = nb.get_vocab_df()
label_df = nb.get_label_df(training_df)

num_words = vocab_df.shape[0]
num_docs = training_df.shape[0]
num_labels = len(label_df)
beta = 1 / abs(num_words)
alpha = 1 + beta

label_word_freqs = nb.get_label_word_freqs(
    training_df, label_df, vocab_df, num_words)

print(label_word_freqs[1][0])

# classify doc
priors, cond_probs = nb.train_naive_bayes(label_df) 
new_doc = training_df.iloc[100, 1:-1] # just for testing
classification = nb.naive_bayes_classify(training_df.iloc[100, 1:-1], priors, cond_probs)
print("actual label: ", training_df.iloc[100][-1], "pred label: ", classification)
