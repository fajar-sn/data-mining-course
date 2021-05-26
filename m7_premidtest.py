import numpy
import pandas
from sklearn.tree import DecisionTreeClassifier

def process_data(input_data):
    # Replace sex value
    data = input_data.replace("male", 1)
    data = data.replace("female", 0)

    # Replace embarked value
    data = data.replace("C", 1)
    data = data.replace("Q", 2)
    data = data.replace("S", 3)

    data = data.fillna(dataset.groupby("Survived").transform("mean"))

    # Convert to float32 data type
    # data["Fare"] = numpy.nan_to_num(data["Fare"].astype(numpy.float32))
    data["Embarked"] = numpy.nan_to_num(data["Embarked"].astype(numpy.float32))
    return data

# pandas.set_option('display.max_rows', None)

def classify_dtc(data, label):
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(data, label)
    accuracy = decision_tree_classifier.score(test_data, test_label)
    error = round((1 - accuracy) * 100, 2)
    return error

dataset = pandas.read_csv("assets/titanic.csv")
train_label = pandas.DataFrame(dataset, columns=["Survived"])

test_label = pandas.read_csv("assets/titanic_testlabel.csv")
test_label = test_label["Survived"]

test_data = pandas.read_csv("assets/titanic_test.csv")
# train_data = dataset.drop(["PassengerId", "Survived", "Name", "Cabin", "Ticket"], axis=1)
# test_data = test_data.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)
# train_data = pandas.DataFrame(dataset, columns=["Sex", "Age", "Pclass", "Fare"])
# test_data = pandas.DataFrame(test_data, columns=["Sex", "Age", "Pclass", "Fare"])
# train_data = pandas.DataFrame(dataset, columns=["Pclass", 'Sex', "Embarked"])
# test_data = pandas.DataFrame(test_data, columns=["Pclass", 'Sex', "Embarked"])
train_data = pandas.DataFrame(dataset, columns=["Sex", "Embarked"])
train_data = process_data(train_data)

test_data = pandas.DataFrame(test_data, columns=["Sex", "Embarked"])
test_data = process_data(test_data)

print("Error:", classify_dtc(train_data, train_label), "%")
