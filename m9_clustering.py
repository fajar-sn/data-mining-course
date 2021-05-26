import pandas
import matplotlib.pyplot as plot
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid

dataset = pandas.read_csv("assets/transaction.csv")
country = pandas.DataFrame(dataset, columns=["Country"])
country = country.value_counts()
transaction = pandas.DataFrame(dataset, columns=["InvoiceNo", "Country"])
transaction = transaction.drop_duplicates(subset=["InvoiceNo"])
transaction = transaction.drop(columns=["InvoiceNo"])
transaction = transaction.value_counts()

new_transaction = {
    "Index": [],
    "Country": [],
    "Transaction Count": []
}

for i in range(len(transaction)):
    new_transaction["Index"].append(i)
    new_transaction["Country"].append(transaction.keys()[i][0])
    new_transaction["Transaction Count"].append(transaction[i])

transaction = pandas.DataFrame(new_transaction)
clustering = AgglomerativeClustering(n_clusters=3, linkage='average')
cluster = clustering.fit_predict(transaction.drop(columns="Country"))
nearest_centroid = NearestCentroid()
nearest_centroid.fit(transaction.drop(columns="Country"), cluster)
transaction["Cluster"] = cluster
country_cluster = [[], [], []]

for label, content in transaction["Cluster"].items():
    if content == 0:
        country_cluster[0].append(transaction["Country"][label])
    elif content == 1:
        country_cluster[1].append(transaction["Country"][label])
    elif content == 2:
        country_cluster[2].append(transaction["Country"][label])

transaction.plot.scatter(x='Country', y='Cluster', c='Cluster', colormap='gnuplot')
plot.show()
