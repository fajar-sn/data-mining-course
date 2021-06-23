import numpy
import pandas
import math

from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors._nearest_centroid import NearestCentroid
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

dataset = pandas.read_csv("assets/transaction.csv")
data = pandas.DataFrame(dataset, columns=["InvoiceNo", "Country", "InvoiceDate"])
months = []
years = []

for label, content in data["InvoiceDate"].items():
    date = datetime.strptime(data["InvoiceDate"][label], "%m/%d/%Y %H:%M")
    months.append(date.month)
    years.append(date.year)

data["Month"] = months
data["Year"] = years
data = data.drop(columns=["InvoiceDate"])
data = data[data["Year"] == 2011]
high_transaction = []

for i in range(11):
    month_data = pandas.DataFrame(data[data["Month"] == i + 1])
    transaction = pandas.DataFrame(month_data, columns=["InvoiceNo", "Country", "Year"])
    transaction = transaction.drop_duplicates(subset=["InvoiceNo"])
    countries = transaction.drop_duplicates(subset=["Country"])["Country"]
    countries_map = {}

    for label, content in countries.items():
        count = len(transaction[transaction["Country"] == content])
        countries_map[content] = count

    transaction = pandas.DataFrame(countries_map.keys())
    transaction = transaction.rename(columns={0: "Country"})
    transaction["TotalTransaction"] = countries_map.values()
    temp_transaction = transaction.drop(columns=["Country"])
    temp_transaction["Index"] = temp_transaction.index
    clustering = AgglomerativeClustering(n_clusters=3, linkage='average')
    cluster = clustering.fit_predict(temp_transaction)
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(temp_transaction, cluster)
    centroid = nearest_centroid.centroids_
    sorted_centroid = sorted(centroid, key=lambda k: math.sqrt(k[0]**2 + k[1]**2))
    high_transaction.append(sorted_centroid.pop())

x = []
y = []

for i in range(len(high_transaction)):
    x.append(i + 1)
    y.append(math.sqrt(high_transaction[i][0]**2 + high_transaction[i][1]**2))

plt.scatter(x, y)
plt.plot(x, y)
plt.xlabel("Month")
plt.ylabel("Jarak ke centroid transaksi tinggi")

linear_regression = LinearRegression()
x = numpy.array(x).reshape(-1, 1)
linear_regression.fit(x, y)
next_x = 13
next_x = numpy.array(next_x).reshape(-1, 1)
predicted_x = linear_regression.predict(next_x)
plt.scatter(next_x, predicted_x, c='red')
predicted_y = linear_regression.predict(x)
plt.plot(x, predicted_y)
plt.show()
