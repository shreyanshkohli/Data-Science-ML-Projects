import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

digits = load_digits()
# print(dir(digits))

data = pd.DataFrame(digits.data, columns=digits.feature_names)
data['target'] = digits.target
data['values'] = data.target.apply(lambda x: digits.target_names[x])

x = data.drop(['values'], axis=1)
y = data['values']

xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.9)

model = SVC()
model.fit(xtrain,ytrain)
p = model.predict(xtest)
score = model.score(xtest, ytest)

cm = confusion_matrix(ytrain[:180], ytest)
sns.heatmap(cm, annot=True)
plt.show()

#score: 99.4%