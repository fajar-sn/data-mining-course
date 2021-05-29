import pandas

dataset = pandas.read_csv("assets/transaction.csv")
country = pandas.DataFrame(dataset, columns=["Country"])
country = country.value_counts().sort_values()

temp_data = {"InvoiceNo": [], "QtyTotal": [], "Country": []}

qty_total = 0
current_invoice = dataset["InvoiceNo"][0]

for i in range(len(dataset)):
    qty_total += dataset["Qty"][i]

    if current_invoice != dataset["InvoiceNo"][i]:
        temp_data["InvoiceNo"].append(current_invoice)
        temp_data["QtyTotal"].append(qty_total)
        temp_data["Country"].append(dataset["Country"][i - 1])
        current_invoice = dataset["InvoiceNo"][i]
        qty_total = dataset["Qty"][i]

temp_data = pandas.DataFrame(temp_data)
temp_data = temp_data.sort_values(by=["Country"], ignore_index=True)

transaction_dict = {"Country": [], "AvgQtyPerTransaction": []}

temp_qty_sum = 0
current_country = temp_data["Country"][0]

for i in range(len(temp_data["InvoiceNo"])):
    if current_country == temp_data["Country"][i]:
        temp_qty_sum += temp_data["QtyTotal"][i]
    else:
        avg_qty_per_transaction = temp_qty_sum / country[temp_data["Country"][i - 1]]
        transaction_dict["Country"].append(temp_data["Country"][i - 1])
        transaction_dict["AvgQtyPerTransaction"].append(avg_qty_per_transaction)
        temp_qty_sum = temp_data["QtyTotal"][i]
        current_country = temp_data["Country"][i]

transaction = pandas.DataFrame(transaction_dict)

from sklearn.cluster import KMeans

transaction["CountryIndex"] = transaction.index.values
cluster_i = []
cluster_val = []
cluster_centers = []

for i in range(10):
    clustering = KMeans(n_clusters=3, init="random", n_init=10)
    clusters = clustering.fit_predict(transaction.drop(columns="Country"))
    cluster_i.append(clusters)
    cluster_val.append(clustering.inertia_)
    cluster_centers.append(clustering.cluster_centers_)

print("\nHasil 10x clustering:\n", cluster_i)
print("\nNilai SSE 10x clustering:\n", cluster_val)

cluster_val = pandas.DataFrame(cluster_val)
cluster = cluster_i[cluster_val.idxmin()[0]]

print("\nCluster terpilih:\n", cluster)

centroid = cluster_centers[cluster_val.idxmin()[0]]

print("\nCentroid:\n", centroid)

import numpy

idx = numpy.argsort(centroid.sum(axis=1))
lut = numpy.zeros_like(idx)

print(idx)
