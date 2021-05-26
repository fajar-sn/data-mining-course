import pandas

dataset = pandas.read_csv("assets/transaction.csv")
country = pandas.DataFrame(dataset, columns=["Country"])
country = country.value_counts()

temp_dict = {
    "InvoiceNo": [],
    "QtyTotal": [],
    "Country": []
}

qty_total = 0
current_invoice = dataset["InvoiceNo"][0]

for i in range(len(dataset)):
    if current_invoice == dataset["InvoiceNo"][i]:
        qty_total += dataset[""]

# print(dataset[0])
