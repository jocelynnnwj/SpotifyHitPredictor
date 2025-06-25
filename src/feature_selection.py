"""
Feature selection module for Spotify song popularity prediction.
Implements various feature selection methods and techniques.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def select_features_sequential(X_train, y_train, n_features=5, direction='forward'):
    """
    Perform sequential feature selection.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_features (int): Number of features to select
        direction (str): 'forward' or 'backward'
        
    Returns:
        list: Selected feature names
    """
    sfs = SequentialFeatureSelector(
        LinearRegression(), 
        n_features_to_select=n_features, 
        direction=direction, 
        scoring='r2', 
        cv=5
    )
    
    sfs.fit(X_train, y_train)
    selected_features_mask = sfs.get_support()
    selected_features = X_train.columns[selected_features_mask].tolist()
    
    print(f"Sequential Feature Selection ({direction}) selected: {selected_features}")
    return selected_features


def select_features_kbest(X_train, y_train, k=10):
    """
    Perform feature selection using SelectKBest with F-regression.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        k (int): Number of features to select
        
    Returns:
        tuple: (selected_features, feature_scores)
    """
    select_k_best = SelectKBest(f_regression, k=k)
    select_k_best.fit(X_train, y_train)
    
    selected_features_mask = select_k_best.get_support()
    selected_features = X_train.columns[selected_features_mask].tolist()
    
    # Get feature scores
    scores = select_k_best.scores_
    feature_scores = pd.DataFrame({
        'Feature': X_train.columns, 
        'Score': scores
    }).sort_values(by='Score', ascending=False)
    
    print(f"SelectKBest selected top {k} features:")
    print(feature_scores.head(k))
    
    return selected_features, feature_scores


def get_feature_importance_rf(X_train, y_train, n_estimators=100):
    """
    Get feature importance using Random Forest.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees in the forest
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("Random Forest Feature Importance:")
    print(feature_importance)
    
    return feature_importance


def select_top_features_by_score(feature_scores, threshold=50):
    """
    Select features based on a score threshold.
    
    Args:
        feature_scores (pd.DataFrame): Feature scores dataframe
        threshold (float): Score threshold for selection
        
    Returns:
        list: Selected feature names
    """
    top_features = feature_scores[feature_scores['Score'] > threshold]['Feature'].tolist()
    print(f"Features with score > {threshold}: {top_features}")
    return top_features


def get_correlation_features(data, target_column='track_popularity', threshold=0.1):
    """
    Select features based on correlation with target.
    
    Args:
        data (pd.DataFrame): Input dataset
        target_column (str): Target variable name
        threshold (float): Correlation threshold
        
    Returns:
        list: Selected feature names
    """
    correlations = data.corr()[target_column].abs().sort_values(ascending=False)
    correlated_features = correlations[correlations > threshold].index.tolist()
    correlated_features.remove(target_column)  # Remove target from features
    
    print(f"Features with correlation > {threshold} with {target_column}:")
    print(correlated_features)
    
    return correlated_features


def comprehensive_feature_selection(X_train, y_train, k=7):
    """
    Perform comprehensive feature selection using multiple methods.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        k (int): Number of features to select
        
    Returns:
        dict: Results from different feature selection methods
    """
    results = {}
    
    # Sequential Feature Selection
    results['sequential_forward'] = select_features_sequential(X_train, y_train, k, 'forward')
    
    # SelectKBest
    results['selectkbest'], results['feature_scores'] = select_features_kbest(X_train, y_train, k)
    
    # Random Forest Importance
    results['rf_importance'] = get_feature_importance_rf(X_train, y_train)
    
    # Top features by score
    results['top_by_score'] = select_top_features_by_score(results['feature_scores'], 50)
    
    return results


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_and_clean_data, standardize_features, prepare_data_for_modeling
    
    # Load and prepare data
    data = load_and_clean_data("../data/spotify_songs.csv")
    scaled_data = standardize_features(data)
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(scaled_data)
    
    # Perform feature selection
    results = comprehensive_feature_selection(X_train, y_train) 