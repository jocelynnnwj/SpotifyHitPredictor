"""
Visualization module for Spotify song popularity prediction.
Handles all plotting and visualization functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def setup_plotting_style():
    """Set up consistent plotting style for all visualizations."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_correlation_matrix(data, save_path=None):
    """
    Plot correlation matrix heatmap.
    
    Args:
        data (pd.DataFrame): Input dataset
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    corr = data.corr()
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {save_path}")
    
    plt.show()


def plot_feature_distributions(data, features=None, save_path=None):
    """
    Plot distribution of features.
    
    Args:
        data (pd.DataFrame): Input dataset
        features (list): List of features to plot
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    if features is None:
        features = data.columns
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.histplot(data[feature], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature distributions saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_importance_df, save_path=None):
    """
    Plot feature importance scores.
    
    Args:
        feature_importance_df (pd.DataFrame): Feature importance dataframe
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance Scores', fontsize=16)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results_dict, metric='Accuracy', save_path=None):
    """
    Plot model comparison based on a specific metric.
    
    Args:
        results_dict (dict): Dictionary containing model results
        metric (str): Metric to compare (e.g., 'Accuracy', 'R2', 'F1_Score')
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    # Extract the specified metric for each model
    model_names = list(results_dict.keys())
    metric_values = [results_dict[model].get(metric, 0) for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values, color=sns.color_palette("husl", len(model_names)))
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.title(f'Model Comparison - {metric}', fontsize=16)
    plt.xlabel('Models')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def plot_prediction_vs_actual(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot predicted vs actual values for regression models.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Predicted vs Actual Values', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction vs actual plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix for classification models.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Popular', 'Popular'],
                yticklabels=['Not Popular', 'Popular'])
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_popularity_distribution(data, save_path=None):
    """
    Plot distribution of song popularity.
    
    Args:
        data (pd.DataFrame): Input dataset
        save_path (str): Path to save the plot
    """
    setup_plotting_style()
    
    plt.figure(figsize=(12, 5))
    
    # Original popularity distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data['track_popularity'], bins=30, kde=True)
    plt.title('Distribution of Song Popularity')
    plt.xlabel('Popularity Score')
    plt.ylabel('Frequency')
    
    # Popularity categories
    plt.subplot(1, 2, 2)
    if 'popularity_category' in data.columns:
        sns.countplot(data=data, x='popularity_category')
        plt.title('Popularity Categories')
        plt.xlabel('Popularity Category (0=Not Popular, 1=Popular)')
        plt.ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Popularity distribution saved to {save_path}")
    
    plt.show()


def create_summary_plots(data, feature_importance_df=None, results_dict=None, save_dir='results/plots/'):
    """
    Create a comprehensive set of summary plots.
    
    Args:
        data (pd.DataFrame): Input dataset
        feature_importance_df (pd.DataFrame): Feature importance dataframe
        results_dict (dict): Model results dictionary
        save_dir (str): Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("Creating summary plots...")
    
    # Correlation matrix
    plot_correlation_matrix(data, save_path=f"{save_dir}correlation_matrix.png")
    
    # Feature distributions
    plot_feature_distributions(data, save_path=f"{save_dir}feature_distributions.png")
    
    # Popularity distribution
    plot_popularity_distribution(data, save_path=f"{save_dir}popularity_distribution.png")
    
    # Feature importance (if available)
    if feature_importance_df is not None:
        plot_feature_importance(feature_importance_df, save_path=f"{save_dir}feature_importance.png")
    
    # Model comparison (if available)
    if results_dict is not None:
        if 'Accuracy' in list(results_dict.values())[0]:
            plot_model_comparison(results_dict, 'Accuracy', save_path=f"{save_dir}model_comparison_accuracy.png")
        elif 'R2' in list(results_dict.values())[0]:
            plot_model_comparison(results_dict, 'R2', save_path=f"{save_dir}model_comparison_r2.png")
    
    print(f"All plots saved to {save_dir}")


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_and_clean_data, standardize_features
    
    # Load and prepare data
    data = load_and_clean_data("../data/spotify_songs.csv")
    scaled_data = standardize_features(data)
    
    # Create summary plots
    create_summary_plots(scaled_data) 