"""
Data preprocessing module for Spotify song popularity prediction.
Handles data loading, cleaning, and preprocessing steps.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_and_clean_data(file_path):
    """
    Load and clean the Spotify songs dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Load the data
    data = pd.read_csv(file_path, encoding='latin-1')
    
    # Display basic info
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values:\n{data.isna().sum()}")
    
    # Drop irrelevant columns for analysis
    columns_to_drop = [
        'track_id', 'track_name', 'track_artist', 'track_album_id',
        'track_album_name', 'track_album_release_date', 'playlist_name', 
        'playlist_id', 'playlist_genre', 'playlist_subgenre'
    ]
    
    # Only drop columns that exist in the dataset
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    data.drop(existing_columns, inplace=True, axis=1)
    
    print(f"Dataset after cleaning: {data.shape}")
    return data


def standardize_features(data):
    """
    Standardize numerical features using StandardScaler.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Standardized dataset
    """
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    
    print("Data standardization completed")
    print(f"Mean after scaling: {scaled_df.mean(axis=0).mean():.6f}")
    print(f"Std after scaling: {scaled_df.std(axis=0).mean():.6f}")
    
    return scaled_df


def prepare_data_for_modeling(data, target_column='track_popularity', test_size=0.2, random_state=42):
    """
    Prepare data for machine learning modeling.
    
    Args:
        data (pd.DataFrame): Input dataset
        target_column (str): Name of the target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def create_popularity_categories(data, threshold=1):
    """
    Create binary popularity categories for classification.
    
    Args:
        data (pd.DataFrame): Input dataset
        threshold (float): Threshold for categorizing popularity
        
    Returns:
        pd.DataFrame: Dataset with popularity categories
    """
    data_copy = data.copy()
    data_copy['popularity_category'] = (data_copy['track_popularity'] > threshold).astype(int)
    
    print(f"Popularity categories created with threshold {threshold}")
    print(f"Category distribution:\n{data_copy['popularity_category'].value_counts()}")
    
    return data_copy


def get_feature_statistics(data):
    """
    Get basic statistics for all features.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Feature statistics
    """
    stats = data.describe()
    print("Feature statistics:")
    print(stats)
    return stats


if __name__ == "__main__":
    # Example usage
    data = load_and_clean_data("../data/spotify_songs.csv")
    scaled_data = standardize_features(data)
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(scaled_data) 