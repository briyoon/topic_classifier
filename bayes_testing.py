import numpy as np
import pandas as pd
import csv
import naive_bayes as nb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Read the csv file
training_df = pd.read_csv('resources/training.csv')
training_df, validation_df = train_test_split(training_df, test_size=0.2)
testing_df = pd.read_csv('resources/testing.csv')

x = training_df.iloc[:, 1:-1]
y = training_df.iloc[:, -1].values

betas = np.linspace(0.00001, 1, num=10)
accuracies = []

for beta in betas:
    validation_pred_df = pd.DataFrame(columns=['id', 'pred'])
    a = 1 + beta
    model = nb.NaiveBayesClassifier(alpha=a)
    model.fit(x, y)

    for i, doc in validation_df.iterrows():
        to_classify = doc[1:-1]
        classification = model.predict(to_classify)
        validation_pred_df = pd.concat([validation_pred_df, pd.DataFrame(
            {'id': doc[0], 'pred': classification}, index=[doc[0]])])

    accuracies.append(model.find_accuracy(
        validation_df, validation_pred_df) * 100)

    print('beta: ', beta, 'accuracy: ', model.find_accuracy(
        validation_df, validation_pred_df) * 100, '%')

plt.semilogx(betas, accuracies)
plt.xlabel('beta')
plt.ylabel('accuracy')
plt.show()
plt.savefig('nb.png')


testing_pred_df = pd.DataFrame(columns=['id', 'class'])
a = 1 + 1e-05
model = nb.NaiveBayesClassifier(alpha=a)
model.fit(x, y)

for i, doc in testing_df.iterrows():
    to_classify = doc[1:]
    classification = nb.predict(to_classify)
    testing_pred_df = pd.concat([testing_pred_df, pd.DataFrame(
        {'id': doc[0], 'class': classification}, index=[doc[0]])])

testing_pred_df.to_csv('nb_pred.csv', index=False, sep=',')
