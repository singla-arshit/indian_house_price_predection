# models/train_model.py

import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import sys
import os
import xgboost as xgb
import json
from sklearn.preprocessing import LabelEncoder

from preprocess import load_data, clean_data, get_features_target


def main():
    # Load the dataset
    filepath = 'Indian_House_Prices.csv'
    df = load_data(filepath)
    
    # Clean the data
    df_clean = clean_data(df)

    # Encode City and Location with LabelEncoder
    city_encoder = LabelEncoder()
    df_clean['City'] = city_encoder.fit_transform(df_clean['City'])

    location_encoder = LabelEncoder()
    df_clean['Location'] = location_encoder.fit_transform(df_clean['Location'])



    # Define the selected features
    feature_cols = [
        'Area', 
        'No. of Bedrooms', 
        'Resale', 
        'SwimmingPool', 
        'CarParking', 
        'School', 
        'LiftAvailable', 
        'MaintenanceStaff', 
        'Location', 
        'City'
    ]
    
    # Split into features and target
    X, y = get_features_target(df_clean, feature_cols, target_col='Price')
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost Regressor
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred_xgb = xgb_model.predict(X_test)

    print("XGBoost Performance:")
    print("R² Score:", r2_score(y_test, y_pred_xgb))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_xgb))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_xgb))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

    # Save model
    model_path = 'xgb_model.pkl'
    joblib.dump(xgb_model, model_path)
    print("Model saved to:", model_path)

if __name__ == "__main__":
    main()
