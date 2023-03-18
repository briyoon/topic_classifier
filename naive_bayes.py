import pickle
import numpy as np
import pandas as pd
import os


def get_label_df(training_df):
    # Build label df
    label_df = pd.DataFrame(columns=['id', 'title', 'doc_count'])

    with open('resources/newsgrouplabels.txt', 'r') as f:
        for row in f:
            class_id, class_title = row.split(maxsplit=1)
            # find num docs labeled Y_k
            count = training_df.iloc[:, -
                                     1].value_counts().get(int(class_id), 0)

            label_df = pd.concat([label_df, pd.DataFrame({'id': int(class_id), 'title': class_title.rstrip(
                '\n'), 'doc_count': count}, index=[int(class_id)])])

    return label_df


def get_vocab_df():
    # Build vocab df
    vocab_df = pd.DataFrame(columns=['id', 'word'])

    with open('resources/vocabulary.txt', 'r') as f:
        index = 1
        for word in f:
            vocab_df = pd.concat([vocab_df, pd.DataFrame(
                {'id': index, 'word': word.rstrip('\n')}, ignore_index=True)])
            index += 1
    return vocab_df


def get_label_word_freqs(training_df, label_df, vocab_df, vocab_size):
    label_word_freq_path = 'resources/label_word_freq.pickle'
    label_word_freqs = {}

    if (os.path.isfile(label_word_freq_path)):
        label_word_freqs = np.load(label_word_freq_path, allow_pickle=True)
    else:
        for y in label_df['id']:
            word_counts = [0] * (vocab_size + 1)
            class_docs = training_df.loc[training_df.iloc[:, -1]
                                         == y].iloc[:, 1:-1]
            for __, doc in class_docs.iterrows():
                for x in vocab_df['id']:
                    word_counts[x] += doc[x]

            label_word_freqs[y] = word_counts

        with open('label_word_freq.pickle', 'wb') as handle:
            pickle.dump(label_word_freqs, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    return label_word_freqs


def find_accuracy(original_df, pred_df):
    num_correct = 0
    index = 0
    for i, doc in pred_df.iterrows():
        if (doc['pred'] == original_df.iloc[index,-1]):
            num_correct+= 1
        index += 1
        
    return num_correct/len(original_df)

def train_naive_bayes(label_df, num_docs, num_words, alpha, label_word_freqs):

    # MLE - prior probabilities for naive bayes
    num_labels = len(label_df) + 1
    priors = np.zeros(num_labels)
    for __, label in label_df.iterrows():
        priors[label['id']] = label['doc_count'] / num_docs

    # MAP - Conditional probability for X_i given Y_k
    cond_probs = np.zeros((num_labels - 1, num_words - 1))
    for __, label in label_df.iterrows():
        label_word_counts = np.array(label_word_freqs[label['id']])
        label_total_words = np.sum(label_word_counts, axis=0)
        num = label_word_counts + (alpha - 1)
        den = label_total_words + ((alpha - 1) * num_words)
        cond_probs[label['id'] - 1] = (num / den)[1:]

    return priors, cond_probs

def naive_bayes_classify(new_doc, priors, cond_probs, num_labels):
    label_probs = np.zeros(num_labels)

    for i in range(1, num_labels):
        label_probs[i] = np.log(priors[i]) + \
            np.sum((np.array(new_doc * cond_probs[i])))

    return np.argmax(label_probs[1:])
