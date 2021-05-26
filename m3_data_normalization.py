import pandas
import math

# pandas.set_option('display.max_rows', None)
# Call and show dataset
dataset = pandas.read_csv('assets/titanic.csv')
rows, columns = dataset.shape
data = pandas.DataFrame(dataset, columns=["Age", "Fare"])

# Get data of survived column
survived_class = pandas.DataFrame(dataset, columns=["Survived"])

# Assign Survived column into data
data["Survived"] = survived_class

# Fill NaN data by using mean from each class
data = data.fillna(data.groupby("Survived").transform("mean"))


# Convert value to normalization format
def min_max_normalization(value, min_data, max_data):
    new_min = 0
    new_max = 1
    new_value = ((value - min_data) * (new_max - new_min) / (max_data - min_data))
    return new_value


min_max_data = data
min_age_data = data["Age"].min()
max_age_data = data["Age"].max()
min_fare_data = data["Fare"].min()
max_fare_data = data["Fare"].max()

for i in range(len(min_max_data)):
    # Check if data has been replaced or not
    if min_max_data["Age"][i] > 1:
        # Replace current value with normalization format value
        min_max_data["Age"] = min_max_data["Age"].replace(
            min_max_data["Age"][i], min_max_normalization(
                min_max_data["Age"][i], min_age_data, max_age_data))
    # Check if data has been replaced or not
    if min_max_data["Fare"][i] > 1:
        # Replace current value with normalization format value
        min_max_data["Fare"] = min_max_data["Fare"].replace(
            min_max_data["Fare"][i], min_max_normalization(
                min_max_data["Fare"][i], min_fare_data, max_fare_data))


# Convert value to normalization format
def z_score_normalization(value, mean, std):
    new_value = (value - mean) / std
    return new_value


z_score_data = data  # Create duplicate data
z_score_data_mean = z_score_data.mean()  # Find mean of each column
z_score_data_std = z_score_data.std()  # Find standard deviation of each column

for i in range(len(z_score_data)):
    if z_score_data["Age"][i] > 1:  # Check if data has been replaced or not
        # Replace current value with normalization format value
        z_score_data["Age"] = z_score_data["Age"].replace(z_score_data["Age"][i], z_score_normalization(
            z_score_data["Age"][i],
            z_score_data_mean["Age"],
            z_score_data_std["Age"]))
    if z_score_data['Fare'][i] > 1:  # Check if data has been replaced or not
        # Replace current value with normalization format value
        z_score_data["Fare"] = z_score_data["Fare"].replace(z_score_data["Fare"][i], z_score_normalization(
            z_score_data["Fare"][i],
            z_score_data_mean["Fare"],
            z_score_data_std["Fare"]))


# Convert value to sigmoidal method value
def to_sigmoidal(value):
    new_value = (1 - math.exp(-value)) / (1 + math.exp(-value))
    return new_value


sigmoidal_data = data
# Delete duplicates on age & fare data
temp_age_data = sigmoidal_data["Age"].drop_duplicates()
temp_fare_data = sigmoidal_data["Fare"].drop_duplicates()

for value in temp_age_data:
    sigmoidal_data["Age"] = sigmoidal_data["Age"].replace(value, to_sigmoidal(value))

for value in temp_fare_data:
    sigmoidal_data["Fare"] = sigmoidal_data["Fare"].replace(value, to_sigmoidal(value))

print(z_score_data)
