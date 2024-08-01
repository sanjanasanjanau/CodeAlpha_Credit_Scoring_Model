# Credit Score Prediction Project

# Table of Contents
* Introduction
* Features
* Requirements
* Usage
* Contributing

## Introduction
The Credit Score Prediction project aims to predict credit scores using machine learning models. This project utilizes Python libraries such as Pandas, NumPy, Matplotlib, and scikit-learn to preprocess the data, analyze correlations, select relevant features, scale the data, build regression models, and evaluate their performance.

## Features
* Loading and preprocessing credit score datasets
* Exploratory data analysis (EDA) and data visualization using Matplotlib and seaborn
* Handling missing data and dropping irrelevant features
* Scaling numerical data for model training
* Building various regression models for credit score prediction
* Evaluating model performance using mean absolute error (MAE) and plotting results

## Requirements
* Python 3.x
* Pandas
* NumPy
* Matplotlib
* seaborn
* scikit-learn
* XGBoost

## Usage
* Loading and Preprocessing Data: Load the CreditScore_train.csv and CreditScore_test.csv datasets, concatenate them, and handle missing values.

* Exploratory Data Analysis: Explore data statistics, correlations, and visualize distributions and relationships using Matplotlib and seaborn.

* Feature Selection and Scaling: Drop columns with high missing percentages, select relevant features based on correlation analysis, and scale numerical data using MinMaxScaler.

* Model Building and Evaluation: Build regression models (e.g., Linear Regression, Random Forest, XGBoost) using scikit-learn. Evaluate models using cross-validation and plot results to compare performance.

* Prediction: Train the final model (e.g., XGBoost) on the scaled training data. Predict credit scores for the test data and visualize predictions versus actual values.

## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

### Dataset:https://www.kaggle.com/datasets/prasy46/credit-score-prediction