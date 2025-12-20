# Earnomly - Adult Income Analysis Platform

A modern, fintech-styled machine learning dashboard for analyzing and predicting income levels based on the UCI Adult Income Dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

Earnomly is an interactive web application that provides comprehensive analysis of socioeconomic factors affecting income levels. It combines exploratory data analysis with machine learning models to predict whether an individual's income exceeds $50K annually.

## Features

### Data Overview
- Dataset statistics and sample data viewer
- Data quality metrics with missing value analysis
- Interactive data exploration

### Data Analysis (EDA)
- Income distribution visualization
- Age vs Income correlation analysis
- Interactive correlation heatmaps
- Workclass distribution charts

### Supervised Learning
- **Random Forest Classifier** - Ensemble learning with configurable trees (50-300) and depth
- **Logistic Regression** - Linear classification with regularization tuning
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization

### Unsupervised Learning
- **K-Means Clustering** - Pattern discovery with configurable clusters (2-10)
- **PCA Visualization** - Dimensionality reduction with explained variance ratio
- Elbow method for optimal cluster selection

### Income Predictor
- Real-time income prediction based on user inputs
- Visual confidence scores with probability gauges
- Premium result display with gradient styling

## Tech Stack

- **Frontend**: Streamlit with custom CSS
- **Visualization**: Plotly Express
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/earnomly.git
cd earnomly

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Requirements

```
streamlit
pandas
numpy
plotly
scikit-learn
```

## Dataset

The project uses the **UCI Adult Income Dataset** containing 32,561 records with 15 attributes including:
- Demographics: Age, Sex, Race, Native Country
- Education: Education Level, Education Years
- Employment: Workclass, Occupation, Hours per Week
- Financial: Capital Gain, Capital Loss
- Target: Income (<=50K or >50K)

## Project Structure

```
MLProject/
├── app.py                 # Main Streamlit application
├── ui_config.py           # UI styling and SVG components
├── Logo.svg               # Project logo
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
└── adultData/
    └── adult.data         # UCI Adult Income Dataset
```

## Usage

1. **Train a Model**: Navigate to "Supervised Learning" and click "Start Training"
2. **Explore Data**: Use "Data Overview" and "Data Analysis" for EDA
3. **Discover Patterns**: Use "Unsupervised Learning" for clustering
4. **Make Predictions**: Go to "Income Predictor" and enter details

## Design

The UI features a minimalist fintech aesthetic with:
- Dark theme with subtle gradients
- Custom SVG icons (no emojis)
- Inter & Outfit typography
- Glassmorphism card effects
- Responsive layout

## License

MIT License - feel free to use and modify for your projects.

## Author

Built with Streamlit and Scikit-learn.
