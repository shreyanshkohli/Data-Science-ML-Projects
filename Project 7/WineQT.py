import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# import matplot.pyplot as plt
# import seaborn as sns

data = pd.read_csv('Python\Data\WineQT.csv')
# print(data.isna().any()) output: False

x = data.drop(['Id', 'quality'], axis=1)
y = data['quality']

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1,random_state=93)

# kf = StratifiedKFold(n_splits=10)
# model1 = cross_val_score(LogisticRegression(),xtrain,ytrain).mean()
# model2 = cross_val_score(SVC(),xtrain,ytrain).mean()
# model3 = cross_val_score(DecisionTreeClassifier(),xtrain,ytrain).mean()
# model4 = cross_val_score(RandomForestClassifier(),xtrain,ytrain).mean() score:64%
# print(model1, model2, model3, model4, sep='\n')

model = RandomForestClassifier(n_estimators=85)

model.fit(xtrain,ytrain)
p = model.predict(xtest)
score = model.score(xtest,ytest)
print(p,score)

# fig, axes = plt.subplots(3, 4, figsize=(15, 10))

# for i, ax in enumerate(axes.flatten()):
#     if i < len(data.columns) - 1:  
#         ax.scatter(x=data['quality'], y=data.iloc[:, i])
#         ax.set_xlabel('quality')
#         ax.set_ylabel(data.columns[i])

# plt.tight_layout()
# plt.style.use('ggplot')
# plt.show()


#score: 75.6%
