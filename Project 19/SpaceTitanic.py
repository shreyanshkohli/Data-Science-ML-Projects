import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

train = pd.read_csv('Data/spaceship-titanic/train.csv')
test = pd.read_csv('Data/spaceship-titanic/test.csv')


# MISSING VALUES
pd.set_option('future.no_silent_downcasting', True)

train.fillna({
    'HomePlanet' : train['HomePlanet'].mode()[0],
    'CryoSleep' : train['CryoSleep'].mode()[0],
    'Cabin' : train['Cabin'].mode()[0],
    'Destination' : train['Destination'].mode()[0],
    'Age' : train['Age'].median(),
    'VIP' : train['VIP'].mode()[0],
    'RoomService' : train['RoomService'].median(),
    'FoodCourt' : train['FoodCourt'].median(),
    'ShoppingMall' : train['ShoppingMall'].median(),
    'Spa' : train['Spa'].median(),
    'VRDeck' : train['VRDeck'].median()
}, inplace=True)
# train.dropna(subset=['Name'], inplace=True)
# train.drop(['PassengerId', 'Name'], axis=1, inplace=True)

test.fillna({
    'HomePlanet' : test['HomePlanet'].mode()[0],
    'CryoSleep' : test['CryoSleep'].mode()[0],
    'Cabin' : test['Cabin'].mode()[0],
    'Destination' : test['Destination'].mode()[0],
    'Age' : test['Age'].median(),
    'VIP' : test['VIP'].mode()[0],
    'RoomService' : test['RoomService'].median(),
    'FoodCourt' : test['FoodCourt'].median(),
    'ShoppingMall' : test['ShoppingMall'].median(),
    'Spa' : test['Spa'].median(),
    'VRDeck' : test['VRDeck'].median()
}, inplace=True)
# test.dropna(subset=['Name'], inplace=True)
# test.drop(['PassengerId', 'Name'], axis=1, inplace=True)

# print(train.columns)

train['Deck'] = train.Cabin.apply(lambda x: x.split('/')[0])
train['Cabin_Number'] = train.Cabin.apply(lambda x: x.split('/')[1])
train['Side'] = train.Cabin.apply(lambda x: x.split('/')[2])

test['Deck'] = test.Cabin.apply(lambda x: x.split('/')[0])
test['Cabin_Number'] = test.Cabin.apply(lambda x: x.split('/')[1])
test['Side'] = test.Cabin.apply(lambda x: x.split('/')[2])

# print(test.Cabin_Number.value_counts())
# cabinNum = train.Cabin_Number.value_counts()
# lowCountCabins = cabinNum[cabinNum<=3].index
# filteredCabins = train[train.Cabin_Number.isin(lowCountCabins)]
# filteredCabins = filteredCabins.Cabin_Number.value_counts().index
# train.Cabin_Number = train.Cabin_Number.apply(lambda x: 'OTHER' if x in filteredCabins else x)

# cabinNum = test.Cabin_Number.value_counts()
# lowCountCabins = cabinNum[cabinNum<=3].index
# filteredCabins = test[test.Cabin_Number.isin(lowCountCabins)]
# filteredCabins = filteredCabins.Cabin_Number.value_counts().index
# test.Cabin_Number = test.Cabin_Number.apply(lambda x: 'OTHER' if x in filteredCabins else x)
# print(test.Cabin_Number.value_counts())


categorical = ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP']
nominal = ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP'] # no order
ordinal = [] # order

encodedData = pd.get_dummies(train[nominal], drop_first=True)
encodedData = encodedData.astype(float)
train = pd.concat([train, encodedData], axis=1)
train.drop(nominal, axis=1, inplace=True)
train.drop('Cabin', axis=1, inplace=True)

encodedData = pd.get_dummies(test[nominal], drop_first=True)
encodedData = encodedData.astype(float)
test = pd.concat([test, encodedData], axis=1)
test.drop(nominal, axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

# train.reset_index(drop=True, inplace=True)
# test.reset_index(drop=True, inplace=True)

# MODEL
x = train.drop(['Transported','PassengerId','Name'], axis=1)
y = train['Transported']
x = x.astype(int)
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.1, random_state=42)

# xtrain = train.drop(['Transported','PassengerId','Name'], axis=1)
# ytrain = train['Transported']
# xtest = test.drop(['PassengerId','Name'], axis=1)

model = RandomForestClassifier(n_estimators=170, max_depth=50)
model.fit(xtrain, ytrain)
predicted = model.predict(xtest)
predicted = pd.DataFrame(predicted)
print(train.Transported.value_counts())
print(predicted.value_counts())
print(xtrain.isna().sum())

spaceTitanic = pd.DataFrame()
spaceTitanic['PassengerId'] = test['PassengerId']
spaceTitanic['Transported'] = predicted

spaceTitanic.to_csv('MySubmission.csv', index=False)
joblib.dump(model, 'spaceTitanic')
