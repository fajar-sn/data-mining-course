import pandas
from sklearn.ensemble import IsolationForest

data = {
    'a': [80, 70, 60, 60, 60, 95, 55, 50, 62],
    'b': [60, 84, 40, 65, 35, 98, 70, 53, 64],
    'c': [75, 88, 55, 60, 40, 85, 53, 57, 53],
    'd': [73, 90, 58, 70, 20, 87, 64, 63, 58],
    'e': [81, 65, 47, 68, 56, 93, 74, 58, 40],
    'f': [66, 60, 49, 72, 57, 95, 77, 40, 45]
}

dataframe = pandas.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f'])
clf = IsolationForest(contamination=0.3)
pred = clf.fit_predict(dataframe)

dataframe["Outlier"] = pred.reshape(-1, 1)

print(dataframe, "\n")
print(dataframe[dataframe["Outlier"] == -1])
