"""
Machine learning models module for Spotify song popularity prediction.
Implements regression and classification models with evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
import joblib


class RegressionModels:
    """Class for regression model training and evaluation."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_linear_regression(self, X_train, y_train, features=None):
        """Train linear regression model."""
        if features is None:
            features = X_train.columns
        
        model = LinearRegression()
        model.fit(X_train[features], y_train)
        self.models['linear_regression'] = model
        return model
    
    def train_random_forest_regressor(self, X_train, y_train, features=None, n_estimators=100):
        """Train random forest regressor."""
        if features is None:
            features = X_train.columns
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train[features], y_train)
        self.models['random_forest_regressor'] = model
        return model
    
    def train_decision_tree_regressor(self, X_train, y_train, features=None):
        """Train decision tree regressor."""
        if features is None:
            features = X_train.columns
        
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train[features], y_train)
        self.models['decision_tree_regressor'] = model
        return model
    
    def evaluate_regression_model(self, model, X_test, y_test, features=None, model_name="model"):
        """Evaluate regression model performance."""
        if features is None:
            features = X_test.columns
        
        y_pred = model.predict(X_test[features])
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        self.results[model_name] = results
        
        print(f"\n{model_name} Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"R^2 Score: {r2:.4f}")
        
        return results
    
    def train_all_regression_models(self, X_train, y_train, X_test, y_test, features=None):
        """Train and evaluate all regression models."""
        print("Training Regression Models...")
        
        # Train models
        self.train_linear_regression(X_train, y_train, features)
        self.train_random_forest_regressor(X_train, y_train, features)
        self.train_decision_tree_regressor(X_train, y_train, features)
        
        # Evaluate models
        for name, model in self.models.items():
            self.evaluate_regression_model(model, X_test, y_test, features, name)
        
        return self.results


class ClassificationModels:
    """Class for classification model training and evaluation."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_random_forest_classifier(self, X_train, y_train, features=None, n_estimators=100):
        """Train random forest classifier."""
        if features is None:
            features = X_train.columns
        
        model = RandomForestClassifier(random_state=42, n_estimators=n_estimators)
        model.fit(X_train[features], y_train)
        self.models['random_forest_classifier'] = model
        return model
    
    def train_logistic_regression(self, X_train, y_train, features=None):
        """Train logistic regression model."""
        if features is None:
            features = X_train.columns
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train[features], y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_naive_bayes(self, X_train, y_train, features=None):
        """Train naive bayes classifier."""
        if features is None:
            features = X_train.columns
        
        model = GaussianNB()
        model.fit(X_train[features], y_train)
        self.models['naive_bayes'] = model
        return model
    
    def train_decision_tree_classifier(self, X_train, y_train, features=None):
        """Train decision tree classifier."""
        if features is None:
            features = X_train.columns
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train[features], y_train)
        self.models['decision_tree_classifier'] = model
        return model
    
    def evaluate_classification_model(self, model, X_test, y_test, features=None, model_name="model"):
        """Evaluate classification model performance."""
        if features is None:
            features = X_test.columns
        
        y_pred = model.predict(X_test[features])
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC AUC score (if binary classification)
        try:
            roc_auc = roc_auc_score(y_test, y_pred)
        except:
            roc_auc = None
        
        results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'ROC_AUC': roc_auc
        }
        
        self.results[model_name] = results
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC Score: {roc_auc:.4f}")
        
        return results
    
    def train_all_classification_models(self, X_train, y_train, X_test, y_test, features=None):
        """Train and evaluate all classification models."""
        print("Training Classification Models...")
        
        # Train models
        self.train_random_forest_classifier(X_train, y_train, features)
        self.train_logistic_regression(X_train, y_train, features)
        self.train_naive_bayes(X_train, y_train, features)
        self.train_decision_tree_classifier(X_train, y_train, features)
        
        # Evaluate models
        for name, model in self.models.items():
            self.evaluate_classification_model(model, X_test, y_test, features, name)
        
        return self.results


def save_model(model, filepath):
    """Save a trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a trained model from disk."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def compare_models(results_dict):
    """Compare multiple models and return the best performing one."""
    comparison_df = pd.DataFrame(results_dict).T
    
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Find best model based on primary metric
    if 'R2' in comparison_df.columns:  # Regression
        best_model = comparison_df['R2'].idxmax()
        print(f"\nBest Regression Model: {best_model} (R² = {comparison_df.loc[best_model, 'R2']:.4f})")
    elif 'Accuracy' in comparison_df.columns:  # Classification
        best_model = comparison_df['Accuracy'].idxmax()
        print(f"\nBest Classification Model: {best_model} (Accuracy = {comparison_df.loc[best_model, 'Accuracy']:.4f})")
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_and_clean_data, standardize_features, prepare_data_for_modeling
    
    # Load and prepare data
    data = load_and_clean_data("../data/spotify_songs.csv")
    scaled_data = standardize_features(data)
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(scaled_data)
    
    # Test regression models
    reg_models = RegressionModels()
    reg_results = reg_models.train_all_regression_models(X_train, y_train, X_test, y_test)
    
    # Test classification models (with popularity categories)
    from data_preprocessing import create_popularity_categories
    data_with_categories = create_popularity_categories(scaled_data)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = prepare_data_for_modeling(
        data_with_categories, target_column='popularity_category'
    )
    
    clf_models = ClassificationModels()
    clf_results = clf_models.train_all_classification_models(X_train_clf, y_train_clf, X_test_clf, y_test_clf) 