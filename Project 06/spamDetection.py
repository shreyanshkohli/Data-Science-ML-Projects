import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn. naive_bayes import MultinomialNB

data = pd.read_csv('Python\Data\spam_ham_dataset.csv')
data = data.drop('Unnamed: 0', axis=1)

x = data['text']
y = data['label']

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1, random_state=16)

# v = CountVectorizer()
# xtrain_count = v.fit_transform(xtrain.values).toarray()

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb',MultinomialNB())
])

model.fit(xtrain,ytrain)
mail = ['Automation is among our core values at Web. How are you using automation to improve efficiency in your company?'] 
realSpam = model.predict(mail)
p = model.predict(xtest)
score = model.score(xtest, ytest)
print(score)

#score: 98.8