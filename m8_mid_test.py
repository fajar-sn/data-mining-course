import math

import pandas

dataset = pandas.read_csv("assets/titanic.csv")
data = pandas.DataFrame(dataset, columns=["Sex", "Age", "Pclass", "Fare", "Survived"])
train_data = data.dropna(subset=["Age"])
train_data = train_data.drop(columns="Age")
train_label = data["Age"].dropna()
test_data = pandas.DataFrame()

for index, rows in data.iterrows():
    if math.isnan(rows["Age"]):
        test_data = test_data.append(rows, ignore_index=True)


def to_min_max(current_value, min_data, max_data):
    new_min = 0
    new_max = 1
    new_value = ((current_value - min_data) * (new_max - new_min) / (max_data - min_data))
    return new_value


def normalize_data(datas):
    datas["Sex"] = datas["Sex"].replace("female", 0)
    datas["Sex"] = datas["Sex"].replace("male", 1)
    datas_constraint = {
        "Pclass": {"min": data["Pclass"].min(), "max": data["Pclass"].max()},
        "Fare": {"min": data["Fare"].min(), "max": data["Fare"].max()},
        "Sex": {"min": data["Sex"].min(), "max": data["Sex"].max()},
        "Survived": {"min": data["Survived"].min(), "max": data["Survived"].max()}
    }
    # print(datas_constraint, "\n")
    temp_data = {
        "Pclass": datas["Pclass"].drop_duplicates(),
        "Fare": datas["Fare"].drop_duplicates()
    }

    for value in temp_data["Pclass"].sort_values():
        datas["Pclass"] = datas["Pclass"].replace(value, to_min_max(
            value, datas_constraint["Pclass"]["min"], datas_constraint["Pclass"]["max"]))

    for value in temp_data["Fare"].sort_values():
        datas["Fare"] = datas["Fare"].replace(value, to_min_max(
            value, datas_constraint["Fare"]["min"], datas_constraint["Fare"]["max"]))

    return datas


train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

train_label = train_label.astype(int)
test_data = test_data.drop(columns="Age")

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
kNN.fit(train_data, train_label)
class_result = kNN.predict(test_data)
class_result_index = 0

for index, rows in data.iterrows():
    if math.isnan(rows["Age"]):
        data["Age"][index] = class_result[class_result_index]
        class_result_index += 1

print(data)
