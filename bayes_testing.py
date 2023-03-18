import numpy as np
import pandas as pd
import csv
import pickle
import naive_bayes as nb
from sklearn.model_selection import train_test_split

import os

# Read the csv file
training_df = pd.read_csv('resources/training.csv')
training_df, validation_df = train_test_split(training_df, test_size=0.2)
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
priors, cond_probs = nb.train_naive_bayes(
    label_df, num_docs, num_words, alpha, label_word_freqs)
new_doc = training_df.iloc[100, 1:-1]  # just for testing
test_pred_df = pd.DataFrame(columns=['id', 'pred'])

for i, doc in training_df.iterrows():
    to_classify = doc[1:-1]
    classification = nb.naive_bayes_classify(
        to_classify, priors, cond_probs, num_labels)
    test_pred_df = pd.concat([test_pred_df, pd.DataFrame(
        {'id': doc[0], 'pred': classification}, index=[doc[0]])])

test_pred_df.to_csv('nb_pred.csv', index=False, sep='\t')

# validation
validation_pred_df = pd.DataFrame(columns=['id', 'pred'])

for i, doc in validation_df.iterrows():
    to_classify = doc[1:-1]
    classification = nb.naive_bayes_classify(
        to_classify, priors, cond_probs, num_labels)
    validation_pred_df = pd.concat([validation_pred_df, pd.DataFrame(
        {'id': doc[0], 'pred': classification}, index=[doc[0]])])

print('accuracy: ', nb.find_accuracy(
    validation_df, validation_pred_df) * 100, '%')

betas = np.linspace(0.00001, 1, num=4)

for beta in betas:
    alpha = 1 + beta
    priors, cond_probs = nb.train_naive_bayes(
        label_df, num_docs, num_words, alpha, label_word_freqs)
    validation_pred_df = pd.DataFrame(columns=['id', 'pred'])

    for i, doc in validation_df.iterrows():
        to_classify = doc[1:-1]
        classification = nb.naive_bayes_classify(
            to_classify, priors, cond_probs, num_labels)
        validation_pred_df = pd.concat([validation_pred_df, pd.DataFrame(
            {'id': doc[0], 'pred': classification}, index=[doc[0]])])

betas = np.linspace(0.00001, 1, num=10)
accuracies = []

for beta in betas:
    alpha = 1 + beta
    priors, cond_probs = nb.train_naive_bayes(
        label_df, num_docs, num_words, alpha, label_word_freqs)
    validation_pred_df = pd.DataFrame(columns=['id', 'pred'])

    for i, doc in validation_df.iterrows():
        to_classify = doc[1:-1]
        classification = nb.naive_bayes_classify(
            to_classify, priors, cond_probs, num_labels)
        validation_pred_df = pd.concat([validation_pred_df, pd.DataFrame(
            {'id': doc[0], 'pred': classification}, index=[doc[0]])])

    accuracies.append(nb.find_accuracy(
        validation_df, validation_pred_df) * 100)

    print('beta: ', beta, 'accuracy: ', nb.find_accuracy(
        validation_df, validation_pred_df) * 100, '%')
