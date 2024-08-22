import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder
import joblib

data = pd.read_csv('Python\Data\Melbourne_housing_FULL.csv')

cols = ['Suburb', 'Rooms', 'Type', 'Price', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'YearBuilt', 'Regionname']
data = data[cols]

data.fillna({
    'Price' : data.Price.mean(),
    'Bedroom2' : data.Bedroom2.mode()[0],
    'Bathroom' : data.Bathroom.mode()[0],
    'Car' : data.Car.mode()[0],
    'Landsize' : data.Landsize.mean(),
    'YearBuilt' : data.YearBuilt.mode()[0],
    'Regionname' : data.Regionname.mode()[0],
}, inplace=True)

encoder = OneHotEncoder(sparse_output=True)
encodedData = encoder.fit_transform(data[['Suburb', 'Regionname', 'Type']]).toarray()
encodedData = pd.DataFrame(encodedData)
encodedData = encodedData.drop([0,100,361],axis=1)

data = pd.concat((data, encodedData), axis=1)
data = data.drop(['Suburb','Regionname','Type'], axis=1)

x = data.drop('Price', axis=1)
x.columns = x.columns.astype(str)
y = data.Price

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1)

model1 = LinearRegression()
model1.fit(xtrain, ytrain)
p1 = model1.predict(xtest)
score1 = model1.score(xtest, ytest)

model2 = Lasso()
model2.fit(xtrain, ytrain)
p2 = model2.predict(xtest)
score2 = model2.score(xtest, ytest)

model3 = Ridge()
model3.fit(xtrain, ytrain)
p3 = model3.predict(xtest)
score3 = model3.score(xtest, ytest)

joblib.dump(model1, 'Linear')
joblib.dump(model2, 'Lasso')
joblib.dump(model3, 'Ridge')