import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

data = pd.read_csv('Data/healthcare_noshows.csv')

# PREPROCESSING
region_counts = data.iloc[:, 6].value_counts()
rare_regions = region_counts[region_counts <= 10].index
filtered_data = data[data.iloc[:, 6].isin(rare_regions)]
removal = filtered_data.Neighbourhood.value_counts().index

data.drop('AppointmentID', axis=1, inplace=True)

data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])

data.Neighbourhood = data.Neighbourhood.apply(lambda x: 'OTHER' if x in removal else x)
# print(data.Neighbourhood.value_counts())

ordinal = ['Scholarship', 'Hipertension', 'Diabetes','Alcoholism', 'Handcap', 'SMS_received', 'Showed_up']
nominal = ['Gender', 'Neighbourhood']

encoder = OrdinalEncoder()

for i in ordinal:
    data[i] = encoder.fit_transform(data[[i]])
    # print(i, "encoded")
data[ordinal] = data[ordinal].astype(int)


hotData = pd.get_dummies(data[['Gender', 'Neighbourhood']], drop_first=True) 
# print(hotData)
data = pd.concat([data, hotData], axis=1)
data.drop(['Gender', 'Neighbourhood'], axis=1, inplace=True)

data['ScheduledDayYear'] = data['ScheduledDay'].dt.year
data['ScheduledDayMonth'] = data['ScheduledDay'].dt.month
data['ScheduledDayDay'] = data['ScheduledDay'].dt.day
data['AppointmentDayYear'] = data['AppointmentDay'].dt.year
data['AppointmentDayMonth'] = data['AppointmentDay'].dt.month
data['AppointmentDayDay'] = data['AppointmentDay'].dt.day

data.drop(['ScheduledDay', 'AppointmentDay', 'AppointmentDayYear', 'ScheduledDayYear'], axis=1, inplace=True)


# TRAIN-TEST 
x = data.drop('Showed_up', axis=1)
y = data.Showed_up

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.1, random_state=10)


# MODEL
model = RandomForestClassifier(n_estimators=150, max_depth=30)
model.fit(xtrain, ytrain)
predicted = model.predict(xtest)
score = model.score(xtest, ytest)
# score: 80%


#MATRIX
cm  = confusion_matrix(ytest, predicted)

sns.heatmap(cm, annot=True)
plt.ylabel('Truth')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
# plt.show()

# joblib.dump(model, 'No-Show Model')