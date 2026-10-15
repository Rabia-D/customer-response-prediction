# Customer Response Prediction Using Machine Learning

A machine learning project exploring whether customer demographics, purchasing activity, and engagement history can help predict responses to marketing campaigns.

## Overview

Using a supermarket customer dataset containing **2,240 records**, I explored data quality, engineered behavioural features, visualized purchasing patterns, and compared three classification models to predict `Campaign_Response`.

## Tools

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Google Colab

## Approach

1. **Data profiling:** Examined feature types, missing values, categorical distributions, and outliers.
2. **Cleaning:** Handled missing income values, invalid marital-status categories, implausible birth years, an extreme income outlier, and constant columns.
3. **Feature engineering:** Created `Total_Purchases` and `Customer_Age`.
4. **Exploratory analysis:** Visualized customer spending, campaign response patterns, recency, purchase channels, and feature correlations.
5. **Classification:** Trained and evaluated K-Nearest Neighbors, Support Vector Machine, and Logistic Regression against a majority-class baseline.

## Results

| Model | F1 score |
| --- | ---: |
| Logistic Regression | **0.379** |
| SVM | 0.353 |
| KNN | 0.347 |

Logistic Regression achieved the highest F1 score of the three models. Its reported test accuracy was approximately **86.8%**, compared with a majority-class baseline of approximately **85.1%**. Because relatively few customers responded to the campaign, F1 score, precision, and recall provide important context beyond accuracy.

## Files

- [`customer_response_prediction.ipynb`](customer_response_prediction.ipynb) — data preparation, exploratory analysis, model training, and evaluation.

## Running the notebook

Open the notebook in Google Colab or Jupyter. Install the packages used in the notebook and provide the dataset file `Supermarket_Customer_Dataset.csv`.

The notebook currently reads the dataset from `/content/Supermarket_Customer_Dataset.csv` (a Google Colab path). Update that path if running locally. The dataset is **not included** in this repository; only share it if you have permission to redistribute it.

## Limitations

This is an exploratory classification study rather than a deployed production model. The target is imbalanced, and results are specific to this dataset and evaluation split. Further work could explore cross-validation, threshold tuning, class imbalance strategies, and more robust calibration.
