import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv('Python\Data\cancerData.csv')
data = data.drop(['id','Unnamed: 32'], axis=1)

x = data.drop('diagnosis', axis=1)
y = data.diagnosis

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1)

# model1 = cross_val_score(LogisticRegression(),xtrain,ytrain).mean()
# model2 = cross_val_score(SVC(),xtrain,ytrain).mean()
# model3 = cross_val_score(DecisionTreeClassifier(),xtrain,ytrain).mean()
# model4 = cross_val_score(RandomForestClassifier(),xtrain,ytrain).mean()

# print(model1, model2, model3, model4, sep='\n')

model = RandomForestClassifier()
model.fit(xtrain, ytrain)
p = model.predict(xtest)
score = model.score(xtest, ytest)
print(p, score)

# clf = RandomizedSearchCV(RandomForestClassifier(),{
#     'n_estimators' : range(1,100)
# }, n_iter=10)
# clf.fit(xtrain, ytrain)
# scores = pd.DataFrame(clf.cv_results_)
# scores = scores.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_n_estimators', 'params', 'std_test_score'], axis=1)
# print(scores)

#score: 98.2%