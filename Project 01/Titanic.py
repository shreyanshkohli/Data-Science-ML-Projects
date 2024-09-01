import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('Python\Data\\titanic.csv')

newData = data.drop(['PassengerId', 'Name', 'SibSp','Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

x = newData[['Pclass', 'Sex', 'Age', 'Fare']]
y = newData['Survived']

encoder = OneHotEncoder()

sex = encoder.fit_transform(x[['Sex']]).toarray()
sex = pd.DataFrame(sex)
x = pd.concat([x,sex], axis=1)
x = x.drop(['Sex',1], axis=1)
x = x.fillna({
    'Pclass': x['Pclass'].median(),
    'Age': x['Age'].median(),
    'Fare': x['Fare'].median()
})
y = y.fillna(y.median())
x.columns = x.columns.astype('str')

xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.8, random_state=11)

model = tree.DecisionTreeClassifier()

model.fit(xtrain, ytrain)
p = model.predict(xtest)
score = model.score(xtest,ytest)
print(p, score)

#score: 83%