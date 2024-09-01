import pandas as pd
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['type'] = wine.target

x = data.drop('type', axis=1)
y = data['type']

kf = StratifiedGroupKFold(n_splits=3)
model2 = cross_val_score(GaussianNB(),x,y).mean()
model3 = cross_val_score(MultinomialNB(),x,y).mean()

print(model2, model3)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1, random_state=15)

model2 = GaussianNB()
model2.fit(xtrain, ytrain)
p2 = model2.predict(xtest)
score1 = model2.score(xtest,ytest)
model3 = MultinomialNB()
model3.fit(xtrain, ytrain)
p3 = model3.predict(xtest)
score2 = model3.score(xtest,ytest)

print(score1, score2)
#Gaussian: 100%; Multinomial: 77%

cm2 = confusion_matrix(ytest, p2)
cm3 = confusion_matrix(ytest, p3)

sns.heatmap(cm3, annot=True)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.title('MultinomialNB')
# plt.show()