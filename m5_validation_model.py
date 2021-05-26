import pandas
import numpy
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def holdout():
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, stratify=y1)
    print(x_train)
    # model = LogisticRegression()
    # model.fit(x_train, y_train)
    # result = model.score(x_test, y_test)
    # print(result)


dataset = pandas.read_csv('assets/titanic.csv')
sorted_dataset = dataset.rename_axis("index").sort_values(by=["Survived", "index"])
x1 = pandas.DataFrame(sorted_dataset, columns=["Sex", "Age", "Pclass", "Fare"])
# x1 = x1.drop("Sex", axis=1)
y1 = sorted_dataset["Survived"]
holdout()
# pandas.set_option('display.max_rows', None)