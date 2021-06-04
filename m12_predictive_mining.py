import numpy
import pandas
from datetime import datetime
from matplotlib import pyplot as plt

dataset = pandas.read_csv("assets/transaction.csv")
data = pandas.DataFrame(dataset[dataset["Country"] == "Germany"], columns=["Qty", "Country", "InvoiceDate"])
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

tempData = pandas.DataFrame(data, columns=["Qty", "Month", "Year"])
tempData = tempData.groupby(["Month"])["Qty"].sum()
totalQty = pandas.DataFrame(tempData)

x = totalQty.index
y = totalQty.values

plt.scatter(x, y)
plt.plot(x, y)
plt.xlabel('Month')
plt.ylabel('TotalQty')

from sklearn.linear_model import LinearRegression

linearRegression = LinearRegression()
x = numpy.array(x).reshape(-1, 1)
linearRegression.fit(x, y)

next_x = 13
next_x = numpy.array(next_x).reshape(-1, 1)
predicted_x = linearRegression.predict(next_x)

print("\nPrediksi x \n", predicted_x.item())

plt.scatter(next_x, predicted_x, c='red')
predicted_y = linearRegression.predict(x)
plt.plot(x, predicted_y)
plt.show()
