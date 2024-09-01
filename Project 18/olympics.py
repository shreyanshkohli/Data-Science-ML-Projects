import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data = pd.read_csv('Data/olympics_dataset.csv')

# PREPROCESSING
# name = data.Name.value_counts()
# lowCountNames = name[name<=10].index
# filteredNames = data[data.Name.isin(lowCountNames)]
# filteredNames = filteredNames.Name.value_counts().index

team = data.Team.value_counts()
lowCountTeams = team[team<=3000].index
filteredTeams = data[data.Team.isin(lowCountTeams)]
filteredTeams = filteredTeams.Team.value_counts().index

sport = data.Sport.value_counts()
lowCountSports = sport[sport<=3000].index
filteredSports = data[data.Sport.isin(lowCountSports)]
filteredSports = filteredSports.Sport.value_counts().index

city = data.City.value_counts()
lowCountCity = city[city<=5000].index
filteredCity = data[data.City.isin(lowCountSports)]
filteredCity = filteredCity.City.value_counts().index

event = data.Event.value_counts()
lowCountEvent = event[event<=1500].index
filteredEvent = data[data.Event.isin(lowCountEvent)]
filteredEvent = filteredEvent.Event.value_counts().index


data.Team = data.Team.apply(lambda x: 'Other' if x in filteredTeams else x)
data.Sport = data.Sport.apply(lambda x: 'Other' if x in filteredSports else x)
data.City = data.City.apply(lambda x: 'Other' if x in filteredCity else x)
data.Event = data.Event.apply(lambda x: 'Other' if x in filteredEvent else x)

# print(data.Event.value_counts())

data.drop(['NOC','Season', 'Name'], axis=1, inplace=True)

# print(data.info())

ordinal = []
nominal = ['Sex', 'Team', 'City', 'Event', 'Sport']

encodedData = pd.get_dummies(data[nominal], drop_first=True)
encodedData = encodedData.astype(int)
# print(encodedData.head())
data = pd.concat([data, encodedData], axis=1)
data.drop(['Sex', 'Team', 'City', 'Event', 'Sport'], axis=1, inplace=True)
data.columns = data.columns.astype(str)


# DATA 
x = data.drop('Medal', axis=1)
y = data.Medal

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)


# MODEL
model = RandomForestClassifier()
model.fit(xtrain,ytrain)
predicted = model.predict(xtest)
score = model.score(xtest, ytest)
print(score, predicted)


# PLOTS
cm = confusion_matrix(ytest, predicted)

sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
# plt.show()

# joblib.dump(model, 'Olympics Model')

#score: 85%