import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
train = pd.read_csv('Data/playground-series-s4e9/train.csv')
test = pd.read_csv('Data/playground-series-s4e9/test.csv')

# Define preprocessing functions
def extract_hp_value(engine_text):
    match = re.findall(r'\b\d*\.?\d+(?=HP)', engine_text)
    return match[0] if match else None  

def extract_l_value(engine_text):
    match = re.findall(r'\b\d*\.?\d+(?=L)', engine_text)
    return match[0] if match else None 

def classify_transmission(value):
    value = value.upper() 
    if any(keyword in value for keyword in ['AT', 'A/T', 'AUTOMATIC']):
        return 'Automatic'
    elif any(keyword in value for keyword in ['M/T', 'MT', 'MANUAL']):
        return 'Manual'
    elif 'TRANSMISSION' in value or 'CVT' in value:
        return 'Transmission'
    elif any(keyword in value for keyword in ['SINGLE-SPEED', 'FIXED', 'SINGLE GEAR']):
        return 'Fixed Speed'
    elif 'SCHEDULED FOR OR IN PRODUCTION' in value:
        return 'SCHEDULED FOR OR IN PRODUCTION'
    else:
        return 'Other'

def classify_col(value):
    value = value.upper() 
    if any(keyword in value for keyword in ['YELLOW']):
        return 'Yellow'
    elif any(keyword in value for keyword in ['WHITE', 'ICE']):
        return 'White'
    elif any(keyword in value for keyword in ['GREY', 'GRAY', 'GRANITE']):
        return 'Grey'
    elif any(keyword in value for keyword in ['BLACK', 'MIDNIGHT', 'DARK']):
        return 'Black'
    elif any(keyword in value for keyword in ['RED', 'VELVET', 'ROSE']):
        return 'Red'
    elif any(keyword in value for keyword in ['BROWN', 'BEIGE', 'EBONY']):
        return 'Brown'
    elif any(keyword in value for keyword in ['ORANGE']):
        return 'Orange'
    elif any(keyword in value for keyword in ['PURPLE', 'VIOLET']):
        return 'Purple'
    elif any(keyword in value for keyword in ['BLUE']):
        return 'Blue'
    elif any(keyword in value for keyword in ['GREEN', 'SEA']):
        return 'Green'
    elif any(keyword in value for keyword in ['SILVER']):
        return 'Silver'
    elif any(keyword in value for keyword in ['GOLD']):
        return 'Gold'
    else:
        return 'Other'

# Preprocess train data
train['HP'] = train['engine'].apply(extract_hp_value).astype(float)
train['engineLiters'] = train['engine'].apply(extract_l_value).astype(float)
train.replace(['�', ' ', '', 'Unknown'], np.nan, inplace=True)
train.fillna({
    'fuel_type': train.fuel_type.mode()[0],
    'transmission': train.transmission.mode()[0],
    'HP': train.HP.median(),
    'engineLiters': train.engineLiters.median(),
    'accident': train.accident.mode()[0],
    'clean_title': train.clean_title.mode()[0]
}, inplace=True)

# print(f'train shape: {train.shape}')
# print(f'test shape: {test.shape}')

train['transmission'] = train['transmission'].apply(classify_transmission)
train['ext_col'] = train['ext_col'].apply(classify_col)
train['int_col'] = train['int_col'].apply(classify_col)

models = train.model.value_counts()
lessModels = models[models <=1600].index
filtered_train = train[train.model.isin(lessModels)]
filtered_train = filtered_train.model.value_counts().index
train.model = train.model.apply(lambda x: 'Other' if x in filtered_train else x)

brand = train.brand.value_counts()
lessBrands = brand[brand <=20].index
filtered_trainBrand = train[train.brand.isin(lessBrands)]
filtered_trainBrand = filtered_trainBrand.brand.value_counts().index
train.brand = train.brand.apply(lambda x: 'Other' if x in filtered_trainBrand else x)

# Preprocess test data
test['HP'] = test['engine'].apply(extract_hp_value).astype(float)
test['engineLiters'] = test['engine'].apply(extract_l_value).astype(float)
test.replace(['�', ' ', '', 'Unknown'], np.nan, inplace=True)
test.fillna({
    'fuel_type': test.fuel_type.mode()[0],
    'transmission': test.transmission.mode()[0],
    'HP': test.HP.median(),
    'engineLiters': test.engineLiters.median(),
    'accident': test.accident.mode()[0],
    'clean_title': test.clean_title.mode()[0]
}, inplace=True)
test['transmission'] = test['transmission'].apply(classify_transmission)
test['ext_col'] = test['ext_col'].apply(classify_col)
test['int_col'] = test['int_col'].apply(classify_col)

models = test.model.value_counts()
lessModels = models[models <=1000].index
filtered_test = test[test.model.isin(lessModels)]
filtered_test = filtered_test.model.value_counts().index
test.model = test.model.apply(lambda x: 'Other' if x in filtered_test else x)

brand = test.brand.value_counts()
lessBrands = brand[brand <=10].index
filtered_trainBrand = test[test.brand.isin(lessBrands)]
filtered_trainBrand = filtered_trainBrand.brand.value_counts().index
test.brand = test.brand.apply(lambda x: 'Other' if x in filtered_trainBrand else x)


# Combine and one-hot encode
nominal = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col', 'accident', 'clean_title']
# combined = pd.concat([train[nominal], test[nominal]], axis=0, ignore_index=True)
encodedDataTrain = pd.get_dummies(train[nominal], drop_first=True)
encodedDataTest = pd.get_dummies(test[nominal], drop_first=True)

# Add encoded data to train and test DataFrames
train = pd.concat([train.reset_index(drop=True), encodedDataTrain], axis=1)
test = pd.concat([test.reset_index(drop=True), encodedDataTest], axis=1)

# Drop original nominal columns
train.drop(nominal, axis=1, inplace=True)
test.drop(nominal, axis=1, inplace=True)

train_only_cols = set(train.columns) - set(test.columns)
test_only_cols = set(test.columns) - set(train.columns)

# print("Columns only in train:", train_only_cols)
# print("Columns only in test:", test_only_cols)

# DATA PREP
xtrain = train.drop(['id', 'engine', 'transmission', 'price'], axis=1, errors='ignore')
ytrain = train['price']
xtest = test.drop(['id', 'engine', 'transmission'], axis=1, errors='ignore')


# print(f'xtrain shape: {xtrain.shape}')
# print(f'ytrain shape: {ytrain.shape}')
# print(f'xtest shape: {xtest.shape}')
# print(f'ytest shape: {ytest.shape}')

# MODEL
model = RandomForestRegressor()
model.fit(xtrain, ytrain)
predicted = model.predict(xtest)
print(predicted)

joblib.dump(model, 'CarPredicition')
submission = pd.DataFrame()
submission['id'] = test['id']
submission['price'] = predicted
submission.to_csv('CarsMySubmission')
