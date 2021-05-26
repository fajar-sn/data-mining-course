import pandas
import matplotlib.pyplot as plot

# To show all column without ellipsis
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

# Call and show dataset
dataset = pandas.read_csv('assets/titanic.csv')
data = dataset.loc[:, ['Name', 'Sex', 'Age', 'Pclass', 'Fare']]

# Getting SibSp & Parch data
sibSp_data = dataset.loc[:, ['SibSp']]
parCh_data = dataset.loc[:, ['Parch']]

# Convert to array
sibSp_data = sibSp_data.to_numpy()
parCh_data = parCh_data.to_numpy()

# Sum SibSp & Parch
relatives_data = sibSp_data + parCh_data

# Add new column called 'Relatives'
data['Relatives'] = relatives_data

pclass_data = data.loc[:, ['Pclass']]  # Get Pclass data
pclass_data = pclass_data.to_numpy()  # Convert to numpy format

new_pclass_data = {
    1: 0,
    2: 0,
    3: 0
}  # Create new dict of pclass data

for value in pclass_data:
    if value[0] == 1:
        new_pclass_data[1] += 1
    elif value[0] == 2:
        new_pclass_data[2] += 1
    elif value[0] == 3:
        new_pclass_data[3] += 1

sex_data = pandas.DataFrame(data, columns=['Sex'])  # Get sex data
indexes = []

for i in range(len(sex_data)):
    indexes.append(i)

sex_data['x'] = indexes
colors = ['Red', 'Blue']
# sex_data.plot(x='x', y='Sex', kind='scatter', colormap='Paired')
# plot.show()

sex_data = sex_data.to_numpy()  # Convert to numpy format

new_sex_data = {
    'male': 0,
    'female': 0
}  # Create new dict of sex data

for value in sex_data:
    if value[0] == 'male':
        new_sex_data['male'] += 1
    elif value[0] == 'female':
        new_sex_data['female'] += 1

survived_pclass_data = pandas.DataFrame(dataset, columns=["Survived", "Pclass"])
survived_pclass_data = survived_pclass_data.to_numpy()
survived_pclass_dict = {0: {1: 0, 2: 0, 3: 0}, 1: {1: 0, 2: 0, 3: 0}}

for survived in survived_pclass_data:
    if survived[0] == 0:
        if survived[1] == 1:
            survived_pclass_dict[0][1] += 1
        elif survived[1] == 2:
            survived_pclass_dict[0][2] += 1
        else:
            survived_pclass_dict[0][3] += 1
    else:
        if survived[1] == 1:
            survived_pclass_dict[1][1] += 1
        elif survived[1] == 2:
            survived_pclass_dict[1][2] += 1
        else:
            survived_pclass_dict[1][3] += 1

age_data = pandas.DataFrame(dataset, columns=['Age'])
age_data['x'] = indexes
age_data.plot.scatter(x='x', y='Age', c='Age', colormap='gnuplot')
plot.show()
