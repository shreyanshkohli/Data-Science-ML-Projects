import pandas as pd
import numpy as np
import re
# from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the trained model
model = joblib.load('Projects/Project 21/CarPredicition')

# Define preprocessing functions (reuse from your preprocessing steps)
def extract_hp_value(engine_text):
    match = re.findall(r'\b\d*\.?\d+(?=HP)', engine_text)
    return float(match[0]) if match else np.nan

def extract_l_value(engine_text):
    match = re.findall(r'\b\d*\.?\d+(?=L)', engine_text)
    return float(match[0]) if match else np.nan

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
    else:
        return 'Other'

def classify_col(value):
    value = value.upper()
    color_map = {
        'YELLOW': 'Yellow', 'WHITE': 'White', 'ICE': 'White', 'GREY': 'Grey', 'GRAY': 'Grey', 
        'GRANITE': 'Grey', 'BLACK': 'Black', 'MIDNIGHT': 'Black', 'DARK': 'Black', 'RED': 'Red', 
        'VELVET': 'Red', 'ROSE': 'Red', 'BROWN': 'Brown', 'BEIGE': 'Brown', 'EBONY': 'Brown',
        'ORANGE': 'Orange', 'PURPLE': 'Purple', 'VIOLET': 'Purple', 'BLUE': 'Blue', 'GREEN': 'Green', 
        'SEA': 'Green', 'SILVER': 'Silver', 'GOLD': 'Gold'
    }
    for key in color_map:
        if key in value:
            return color_map[key]
    return 'Other'

def predict_car_price(brand, model, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title):
    # Create a DataFrame from input values
    model = joblib.load('Projects/Project 21/CarPredicition')
    input_data = pd.DataFrame({
        'brand': [brand],
        'model': [model],
        'fuel_type': [fuel_type],
        'engine': [engine],
        'transmission': [transmission],
        'ext_col': [ext_col],
        'int_col': [int_col],
        'accident': [accident],
        'clean_title': [clean_title]
    })

    # Preprocess the input data
    input_data['HP'] = input_data['engine'].apply(extract_hp_value)
    input_data['engineLiters'] = input_data['engine'].apply(extract_l_value)
    input_data['transmission'] = input_data['transmission'].apply(classify_transmission)
    input_data['ext_col'] = input_data['ext_col'].apply(classify_col)
    input_data['int_col'] = input_data['int_col'].apply(classify_col)

    # One-hot encode categorical columns
    nominal = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col', 'accident', 'clean_title']
    input_data = pd.get_dummies(input_data[nominal], drop_first=True)

    # Align input data with the training data
    # xtrain_cols = model.feature_importances_.shape[0]  # Number of features the model was trained on
    # input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Predict the price
    predicted_price = model.predict(input_data)
    
    return predicted_price[0]

# Example usage:
price = predict_car_price(
    brand='Toyota',
    model='Camry',
    fuel_type='Gasoline',
    engine='268.0HP 3.5L V6',
    transmission='6-Speed Automatic',
    ext_col='Silver',
    int_col='Black',
    accident='No',
    clean_title='Yes'
)
print(f"Predicted car price: ${price:.2f}")
