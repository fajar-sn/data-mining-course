import pandas
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset = pandas.read_csv("assets/transaction.csv")
data = dataset[dataset["Country"] == "Portugal"]
transaction = data.groupby(["InvoiceNo", "StockCode"])["Qty"].sum()
transaction = transaction.unstack().reset_index().fillna(0).set_index("InvoiceNo")
transaction[transaction > 0] = 1

print('\nTabel Transaksi:\n', transaction)

frequent_itemsets = apriori(transaction, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("\nAssociation rules:\n", rules[["antecedents", "consequents", "confidence"]])
