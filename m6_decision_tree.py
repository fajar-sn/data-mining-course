import pandas
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

dataset = pandas.read_csv("assets/titanic.csv")
test_dataset = pandas.read_csv("assets/titanic_test.csv")
train_data = pandas.DataFrame(dataset, columns=["Sex", "Age", "Pclass", "Fare"])
test_data = pandas.DataFrame(test_dataset, columns=["Sex", "Age", "Pclass", "Fare"])
train_label = pandas.DataFrame(dataset, columns=["Survived"])
test_label = pandas.read_csv("assets/titanic_testlabel.csv")
test_label = test_label.drop(columns="PassengerId")

train_data["Sex"] = train_data["Sex"].replace("female", 0)
train_data["Sex"] = train_data["Sex"].replace("male", 1)
test_data["Sex"] = test_data["Sex"].replace("female", 0)
test_data["Sex"] = test_data["Sex"].replace("male", 1)
train_data["Survived"] = train_label
test_data["Survived"] = test_label
train_data = train_data.fillna(train_data.groupby("Survived").transform("mean"))
test_data = test_data.fillna(test_data.groupby("Survived").transform("mean"))
train_data = train_data.drop(columns="Survived")
test_data = test_data.drop(columns="Survived")
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(train_data, train_label)
accuracy = decision_tree_classifier.score(train_data, train_label)
error = round((1 - accuracy) * 100, 2)
print("Error ratio =", error, "%")

dot_data = tree.export_graphviz(decision_tree_classifier, out_file=None, feature_names=train_data.columns.values)
graph = graphviz.Source(dot_data, format="png")
graph.render(view=True)
graph.view()
