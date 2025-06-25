# SpotifyHitPredictor 🎵

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)
![Data Analysis](https://img.shields.io/badge/Data%20Analysis-Pandas-green.svg)

## 📊 Project Overview

SpotifyHitPredictor is a machine learning project that analyzes Spotify song data to predict song popularity using advanced data science techniques. The project implements a comprehensive analysis pipeline including exploratory data analysis, feature selection, and both regression and classification modeling approaches to identify what makes a song popular.

## 📂 Data Source

- The main dataset used in this project is from Kaggle:  
  [Top Spotify Songs 2023 - Kaggle Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)

## 🎯 Objectives

- **Data Exploration**: Analyze Spotify song features and their relationships
- **Feature Selection**: Identify the most important features for predicting song popularity
- **Regression Modeling**: Predict exact popularity scores using various algorithms
- **Classification Modeling**: Categorize songs as popular or not popular
- **Model Comparison**: Evaluate different machine learning approaches

## 📁 Project Structure

```
SpotifyHitPredictor/
├── data/
│   ├── spotify_songs.csv              # Main dataset
│   └── spotify_2023_dataset.csv       # Additional dataset
├── notebooks/
│   └── spotify_analysis.ipynb         # Jupyter notebook with analysis
├── src/
│   ├── data_preprocessing.py          # Data cleaning and preprocessing
│   ├── feature_selection.py           # Feature selection methods
│   ├── models.py                      # Machine learning models
│   └── visualization.py               # Plotting and visualization functions
├── results/
│   ├── models/                        # Saved model files
│   └── plots/                         # Generated visualizations
├── requirements.txt                   # Python dependencies
├── main.py                            # Main execution script
├── README.md                          # Project documentation
├── SpotifyHitPredictor_Report.pdf     # Complete project report
├── SpotifyHitPredictor_Presentation.pdf # Project presentation slides
├── spotify_hit_predictor_original.py  # Original analysis script
└── spotify_hit_predictor_original.ipynb # Original Jupyter notebook
```

## 📚 Documentation & Presentations

### 📄 Project Report
- **`SpotifyHitPredictor_Report.pdf`**: Comprehensive project report including methodology, results, and conclusions
- Contains detailed analysis of the Spotify dataset and machine learning approaches
- Includes visualizations, model comparisons, and business insights

### 🎤 Presentation Slides
- **`SpotifyHitPredictor_Presentation.pdf`**: PowerPoint presentation slides for the project
- Summarizes key findings and methodology
- Suitable for academic presentations and stakeholder briefings

### 📓 Interactive Analysis
- **`notebooks/spotify_analysis.ipynb`**: Jupyter notebook with step-by-step analysis
- Interactive version of the analysis with detailed explanations
- Perfect for learning and understanding the methodology

### 📜 Original Files
- **`spotify_hit_predictor_original.py`**: Original Python analysis script
- **`spotify_hit_predictor_original.ipynb`**: Original Jupyter notebook
- These files contain the initial analysis and serve as reference

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SpotifyHitPredictor.git
cd SpotifyHitPredictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python main.py
```

## 📈 Key Features

### Data Features Analyzed
- **Audio Features**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- **Track Information**: duration, popularity, key, mode
- **Metadata**: track name, artist, album, playlist information

### Machine Learning Models
- **Regression Models**:
  - Linear Regression
  - Random Forest Regressor
  - Decision Tree Regressor

- **Classification Models**:
  - Random Forest Classifier
  - Logistic Regression
  - Naive Bayes
  - Decision Tree Classifier

### Feature Selection Methods
- Sequential Feature Selection
- SelectKBest with F-regression
- Correlation analysis

## 📊 Results Summary

The project demonstrates that:
- **Top predictive features**: instrumentalness, duration_ms, energy, acousticness, danceability, loudness, liveness
- **Classification approach** performs better than regression for popularity prediction
- **Random Forest** shows the best performance among tested models

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

## 📝 Usage Examples

### Basic Data Loading
```python
import pandas as pd
from src.data_preprocessing import load_and_clean_data

# Load and clean the data
data = load_and_clean_data('data/spotify_songs.csv')
```

### Feature Selection
```python
from src.feature_selection import select_features

# Select top features
selected_features = select_features(X_train, y_train, method='selectkbest', k=7)
```

### Model Training
```python
from src.models import train_classification_model

# Train a random forest classifier
model = train_classification_model(X_train, y_train, model_type='random_forest')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**Group 48 (USC DSCI 551):**
- Ying Yang (Leader)
- Zhiqian Li
- Yixin Qu
- Wenjing Huang

## 🙏 Acknowledgments

- Dataset from [Kaggle: Top Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)
- Dataset from [TidyTuesday](https://github.com/rfordatascience/tidytuesday)
- Spotify API for providing the audio features
- DSCI 550 course at USC for project guidance

## 📞 Support

For questions or support, please contact the development team or create an issue in the repository.


**Built with ❤️💛 for USC DSCI 550 (Group 2)**
