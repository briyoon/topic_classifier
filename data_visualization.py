import pandas as pd
import matplotlib.pyplot as plt

data_path = 'results/CWV1.1_2023-03-03T21:45:10.505296_0.01_0.01_1000.csv'

data = pd.read_csv(data_path, delimiter=',')
x = range(0, len(data.iloc[0, :]))

for row in range(0, 19):
    plt.scatter(x, data.iloc[row, :], alpha=0.3)
plt.show()
