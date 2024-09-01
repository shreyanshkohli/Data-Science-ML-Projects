import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['type_num'] = iris.target
data['type'] = data.type_num.apply(lambda x: iris['target_names'][x])

x = data.iloc[:,:4]
y = data.type

xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.9)

model = RandomForestClassifier()
model.fit(xtrain, ytrain)
p = model.predict(xtest)

score = model.score(xtest,ytest)

cm = confusion_matrix(ytest,p)
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#score: 100%