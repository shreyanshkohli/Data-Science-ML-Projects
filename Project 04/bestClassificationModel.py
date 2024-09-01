import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['type_num'] = iris.target
data['type'] = data.type_num.apply(lambda x: iris.target_names[x])
# data = data.drop('type_num', axis=1)

x = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)']]
y = data['type_num']

kf = StratifiedKFold(n_splits=10)

lrScore = cross_val_score(LogisticRegression(), x, y).mean()
svcScore = cross_val_score(SVC(), x,y).mean()
treeScore = cross_val_score(tree.DecisionTreeClassifier(), x,y).mean()
rfScore = cross_val_score(RandomForestClassifier(n_estimators=200), x,y).mean()

scores = [lrScore,svcScore,treeScore,rfScore]
for i in scores:
    print(i)

#best score for iris dataset: Logitis Regression = 97.3%