import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()
data = pd.DataFrame(digits.data, columns=digits.feature_names)
data['type_num'] = digits.target
data['type'] = data.type_num.apply(lambda x: digits['target_names'][x])
data.drop('type_num', axis=1, inplace=True)

x = data.drop('type', axis=1)
y = data['type']

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1, random_state=1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(xtrain, ytrain)
p = model.predict(xtest)
score = model.score(xtest, ytest)

cm = confusion_matrix(ytest, p)
plt.style.use('ggplot')
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()

# joblib.dump(model, 'DigitsClassifier')

#score: 100%