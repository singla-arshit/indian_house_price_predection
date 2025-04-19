# data/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Clean and preprocess the data:
    - Drop unnecessary columns
    - Handle missing values
    - Encode categorical features using label encoding
    """
    # Drop 'Unnamed: 0' if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Fill missing numeric values with the median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Label encode categorical columns (e.g., 'Location' and 'City')
    if 'Location' in df.columns:
        le = LabelEncoder()
        df['Location'] = le.fit_transform(df['Location'])
    if 'City' in df.columns:
        le = LabelEncoder()
        df['City'] = le.fit_transform(df['City'])
    
    return df

def get_features_target(df, feature_cols, target_col='Price'):
    """
    Split the DataFrame into features (X) and target (y).
    """
    X = df[feature_cols]
    y = df[target_col]
    return X, y

