import numpy as np

class NaiveBayesClassifier:

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.log_prior = None
        self.log_likelihood = None

    def fit(self, X, y):
        # Estimate P(Y) using MLE
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.log_prior = np.log(class_counts / y.shape[0])
        # Estimate P(X|Y) using MAP with Dirichlet prior
        V = X.shape[1]
        self.log_likelihood = np.zeros((len(self.classes), V))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            # Add alpha to each count for smoothing
            word_counts = X_c.sum(axis=0) + self.alpha
            total_counts = word_counts.sum()
            self.log_likelihood[i, :] = np.log(word_counts / total_counts)

    def predict(self, Xnew):
        # Calculate log-probabilities for each class
        log_probs = np.zeros(len(self.classes))
        for i, c in enumerate(self.classes):
            log_probs[i] = self.log_prior[i] + \
                (self.log_likelihood[i, :] * Xnew).sum()

        # Return the class with the highest log-probability
        return self.classes[np.argmax(log_probs)]

    def find_accuracy(original_df, pred_df):
        num_correct = 0
        index = 0
        for i, doc in pred_df.iterrows():
            if (doc['pred'] == original_df.iloc[index, -1]):
                num_correct += 1
            index += 1

        return num_correct/len(original_df)
