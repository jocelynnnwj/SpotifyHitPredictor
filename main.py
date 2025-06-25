#!/usr/bin/env python3
"""
Main execution script for SpotifyHitPredictor.
Orchestrates the entire analysis pipeline from data loading to model evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_preprocessing import (
    load_and_clean_data, standardize_features, 
    prepare_data_for_modeling, create_popularity_categories
)
from feature_selection import comprehensive_feature_selection
from models import RegressionModels, ClassificationModels, compare_models
from visualization import create_summary_plots


def main():
    """Main execution function."""
    print("🎵 SpotifyHitPredictor - Song Popularity Analysis")
    print("=" * 55)
    
    # Create results directories
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Step 1: Data Loading and Preprocessing
    print("\n📊 Step 1: Loading and Preprocessing Data")
    print("-" * 40)
    
    try:
        # Load and clean data
        data = load_and_clean_data('data/spotify_songs.csv')
        
        # Standardize features
        scaled_data = standardize_features(data)
        
        # Get feature statistics
        stats = scaled_data.describe()
        print(f"\nDataset Statistics:")
        print(f"Shape: {scaled_data.shape}")
        print(f"Features: {list(scaled_data.columns)}")
        
    except FileNotFoundError:
        print("❌ Error: spotify_songs.csv not found in data/ directory")
        print("Please ensure the data file is in the correct location.")
        return
    
    # Step 2: Feature Selection
    print("\n🔍 Step 2: Feature Selection")
    print("-" * 40)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(scaled_data)
    
    # Perform comprehensive feature selection
    feature_results = comprehensive_feature_selection(X_train, y_train, k=7)
    
    # Use top features based on SelectKBest scores
    top_features = feature_results['top_by_score']
    print(f"\nSelected top features: {top_features}")
    
    # Step 3: Regression Modeling
    print("\n📈 Step 3: Regression Modeling")
    print("-" * 40)
    
    reg_models = RegressionModels()
    reg_results = reg_models.train_all_regression_models(
        X_train, y_train, X_test, y_test, features=top_features
    )
    
    # Step 4: Classification Modeling
    print("\n🎯 Step 4: Classification Modeling")
    print("-" * 40)
    
    # Create popularity categories
    data_with_categories = create_popularity_categories(scaled_data)
    
    # Prepare data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = prepare_data_for_modeling(
        data_with_categories, target_column='popularity_category'
    )
    
    clf_models = ClassificationModels()
    clf_results = clf_models.train_all_classification_models(
        X_train_clf, y_train_clf, X_test_clf, y_test_clf, features=top_features
    )
    
    # Step 5: Model Comparison
    print("\n🏆 Step 5: Model Comparison")
    print("-" * 40)
    
    # Compare regression models
    print("\nRegression Models Comparison:")
    reg_comparison = compare_models(reg_results)
    
    # Compare classification models
    print("\nClassification Models Comparison:")
    clf_comparison = compare_models(clf_results)
    
    # Step 6: Visualization
    print("\n📊 Step 6: Creating Visualizations")
    print("-" * 40)
    
    # Create summary plots
    create_summary_plots(
        data=scaled_data,
        feature_importance_df=feature_results['rf_importance'],
        results_dict=clf_results,  # Use classification results for comparison
        save_dir='results/plots/'
    )
    
    # Step 7: Save Best Models
    print("\n💾 Step 7: Saving Best Models")
    print("-" * 40)
    
    # Save best regression model
    best_reg_model_name = reg_comparison['R2'].idxmax()
    best_reg_model = reg_models.models[best_reg_model_name]
    reg_models.save_model(best_reg_model, f'results/models/best_regression_model.pkl')
    
    # Save best classification model
    best_clf_model_name = clf_comparison['Accuracy'].idxmax()
    best_clf_model = clf_models.models[best_clf_model_name]
    clf_models.save_model(best_clf_model, f'results/models/best_classification_model.pkl')
    
    # Step 8: Summary Report
    print("\n📋 Step 8: Summary Report")
    print("-" * 40)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total songs analyzed: {len(scaled_data)}")
    print(f"   • Number of features: {len(scaled_data.columns)}")
    print(f"   • Features used for modeling: {len(top_features)}")
    
    print(f"\n🎯 Top Predictive Features:")
    for i, feature in enumerate(top_features, 1):
        print(f"   {i}. {feature}")
    
    print(f"\n📈 Best Regression Model:")
    print(f"   • Model: {best_reg_model_name}")
    print(f"   • R² Score: {reg_comparison.loc[best_reg_model_name, 'R2']:.4f}")
    print(f"   • RMSE: {reg_comparison.loc[best_reg_model_name, 'RMSE']:.4f}")
    
    print(f"\n🎯 Best Classification Model:")
    print(f"   • Model: {best_clf_model_name}")
    print(f"   • Accuracy: {clf_comparison.loc[best_clf_model_name, 'Accuracy']:.4f}")
    print(f"   • F1 Score: {clf_comparison.loc[best_clf_model_name, 'F1_Score']:.4f}")
    
    print(f"\n📁 Results Saved:")
    print(f"   • Models: results/models/")
    print(f"   • Plots: results/plots/")
    
    print(f"\n✅ SpotifyHitPredictor analysis completed successfully!")
    print("=" * 55)


def run_quick_analysis():
    """Run a quick analysis for testing purposes."""
    print("🚀 Running SpotifyHitPredictor Quick Analysis...")
    
    try:
        # Load data
        data = load_and_clean_data('data/spotify_songs.csv')
        scaled_data = standardize_features(data)
        
        # Quick feature selection
        X_train, X_test, y_train, y_test = prepare_data_for_modeling(scaled_data)
        _, feature_scores = comprehensive_feature_selection(X_train, y_train, k=5)
        
        # Quick model training
        reg_models = RegressionModels()
        reg_models.train_random_forest_regressor(X_train, y_train)
        reg_models.evaluate_regression_model(
            reg_models.models['random_forest_regressor'], 
            X_test, y_test, 
            model_name="Random Forest"
        )
        
        print("✅ SpotifyHitPredictor quick analysis completed!")
        
    except Exception as e:
        print(f"❌ Error in quick analysis: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SpotifyHitPredictor - Song Popularity Prediction')
    parser.add_argument('--quick', action='store_true', 
                       help='Run a quick analysis for testing')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_analysis()
    else:
        main() 