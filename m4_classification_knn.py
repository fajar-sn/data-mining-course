import pandas
import math

# Call and show dataset
dataset = pandas.read_csv("assets/titanic.csv")

# Call and show data test
test_dataset = pandas.read_csv("assets/titanic_test.csv")

train_data = pandas.DataFrame(dataset, columns=["Age", "Fare"])

missing_age = []
missing_fare = []

for i in range(len(train_data)):
    if math.isnan(train_data["Age"][i]):
        missing_age.append(i)
    if math.isnan(train_data["Fare"][i]):
        missing_fare.append(i)

for i in range(len(missing_age)):
    missing_fare.append(float('nan'))

pos_missing_train = pandas.DataFrame(missing_age, columns=["Age"])
pos_missing_train["Fare"] = missing_fare

train_data = train_data.dropna()

# pandas.set_option('display.max_rows', None)

test_data = pandas.DataFrame(test_dataset, columns=["Age", "Fare"])

pos_missing_test = {
    "Age": [],
    "Fare": []
}

for i in range(len(test_data)):
    if math.isnan(test_data["Age"][i]):
        pos_missing_test["Age"].append(i)
    if math.isnan(test_data["Fare"][i]):
        pos_missing_test["Fare"].append(i)

test_data = test_data.dropna()

train_label = pandas.DataFrame(dataset, columns=["Age", "Fare", "Survived"])
train_label = train_label.dropna(subset=["Age", "Fare"])
train_label = pandas.DataFrame(train_label, columns=["Survived"])

test_label = pandas.read_csv("assets/titanic_testlabel.csv")
test_label = pandas.DataFrame(test_label, columns=["Survived"])
test_label = test_label.drop(pos_missing_test["Age"])
test_label = test_label.drop(pos_missing_test["Fare"])


def to_min_max(value, min_data, max_data):
    new_min = 0
    new_max = 1
    new_value = ((value - min_data) * (new_max - new_min) / (max_data - min_data))
    return new_value


train_data_min = train_data.min()
train_data_max = train_data.max()
temp_age_train_data = train_data["Age"].drop_duplicates()
temp_fare_train_data = train_data["Fare"].drop_duplicates()

for value in temp_age_train_data:
    train_data["Age"] = train_data["Age"].replace(
        value, to_min_max(value, train_data_min["Age"], train_data_max["Age"]))

for value in temp_fare_train_data:
    train_data["Fare"] = train_data["Fare"].replace(
        value, to_min_max(value, train_data_min["Fare"], train_data_max["Fare"]))

test_data_min = test_data.min()
test_data_max = test_data.max()
temp_age_test_data = test_data["Age"].drop_duplicates()
temp_fare_test_data = test_data["Fare"].drop_duplicates()

for value in temp_age_test_data:
    test_data["Age"] = test_data["Age"].replace(
        value, to_min_max(value, test_data_min["Age"], test_data_max["Age"])
    )

for value in temp_fare_test_data:
    test_data["Fare"] = test_data["Fare"].replace(
        value, to_min_max(value, test_data_min["Fare"], test_data_max["Fare"])
    )

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=10, weights='distance')
kNN.fit(train_data, train_label)
class_result = kNN.predict(test_data)
precision_ratio = kNN.score(test_data, test_label)
error_ratio = 1 - precision_ratio

print("train data:\n", train_data, "\n")
print("train label:\n", train_label, "\n")
print("test data:\n", test_data, "\n")
print(class_result, "\n")
print("error ratio:", error_ratio)
