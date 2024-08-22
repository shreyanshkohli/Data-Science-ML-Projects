import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from loop import optimise

# Load data
data = pd.read_csv('Python/Data/heart.csv')

# Specify columns
ordinal = [None]  # No ordinal columns specified in your example
nominal = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Initialize and apply OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid dummy variable trap
encodedData = encoder.fit_transform(data[nominal])
encodedData = pd.DataFrame(encodedData, columns=encoder.get_feature_names_out(nominal))

# Drop original nominal columns and concatenate encoded data
data = data.drop(nominal, axis=1)
data = pd.concat([data, encodedData], axis=1)

# Separate features and target variable
x = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']  # Target variable

xtrain1, xtest1, ytrain1, ytest1 = train_test_split(x, y, test_size=0.1, random_state=87)

# Initialize models
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()

# Initialize BaggingClassifier
bag1 = BaggingClassifier(
    estimator=model1,
    n_estimators=100,
    max_samples=0.9,
    oob_score=True,
    random_state=0
)
bag2 = BaggingClassifier(
    estimator=model2,
    n_estimators=100,
    max_samples=0.9,
    oob_score=True,
    random_state=0
)

# Fit models
model1.fit(xtrain1, ytrain1)
model2.fit(xtrain1, ytrain1)
bag1.fit(xtrain1, ytrain1)
bag2.fit(xtrain1, ytrain1)

# Evaluate models
scores = {
    'Decision Tree': model1.score(xtest1, ytest1),
    'Random Forest': model2.score(xtest1, ytest1), 
    'Bagging Decision Tree': bag1.score(xtest1, ytest1),
    'Bagging Random Forest': bag2.score(xtest1, ytest1)
}

print(scores)

#{'Decision Tree': 0.7934782608695652, 'Random Forest': 0.9565217391304348, 'Bagging Decision Tree': 0.9239130434782609, 'Bagging Random Forest': 0.9565217391304348}