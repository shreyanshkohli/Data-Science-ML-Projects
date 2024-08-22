import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
# from loop import optimise
import joblib

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

# Standardize features
scaler = StandardScaler()
xScaled = scaler.fit_transform(x)

# Apply PCA to features
pca = PCA(0.95)  # Retain 95% of variance
xPCA = pca.fit_transform(xScaled)

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(xPCA, y, test_size=0.1, random_state=48)
xtrain1, xtest1, ytrain1, ytest1 = train_test_split(x, y, test_size=0.1, random_state=87)
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(xScaled, y, test_size=0.1, random_state=1)

# Initialize and train model
model = RandomForestClassifier()
model1 = RandomForestClassifier()
model2 = RandomForestClassifier()
model.fit(xtrain, ytrain)
model1.fit(xtrain1, ytrain1)
model2.fit(xtrain2, ytrain2)

# Evaluate model
score = model.score(xtest, ytest)
score1 = model1.score(xtest1, ytest1)
score2 = model2.score(xtest2, ytest2)
print(f"Accuracy: {score} & {score1} & {score2}")

#scores: 93%, 94%, 92%

# joblib.dump(model, 'PCAModel')
# joblib.dump(model1, 'regModel')
# joblib.dump(model2, 'scaledModel')