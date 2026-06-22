# Week 2 – Tesla Deliveries Forecasting and Time Series Analysis

## Overview

This project analyzes Tesla vehicle deliveries data from 2015–2025 using Python for Exploratory Data Analysis (EDA), feature engineering, machine learning, forecasting, and time series stationarity testing.

The objective is to understand delivery trends, build predictive models, evaluate forecasting performance, and interpret statistical results.

---

## Dataset

**File:** `tesla_deliveries_dataset_2015_2025.csv`

The dataset contains Tesla vehicle production and delivery information across different regions, models, and years.

---

## Tasks Completed

### 1. Data Loading and Inspection

* Loaded dataset into Pandas DataFrame
* Displayed:

  * Dataset shape
  * Column names
  * DataFrame information (`.info()`)
  * Statistical summary (`.describe()`)

### 2. Data Cleaning

* Checked for missing values
* Checked for duplicate records
* Verified data quality before modeling

### 3. Exploratory Data Analysis (EDA)

Created the following visualizations:

1. Deliveries by Model
2. Deliveries by Region
3. Correlation Heatmap
4. Production vs Deliveries Scatter Plot
5. Tesla Deliveries Time Trend

All charts include:

* Appropriate titles
* X-axis labels
* Y-axis labels

### 4. Feature Engineering

Applied Label Encoding on:

* Region
* Model
* Source_Type

Created additional features:

* `Deliveries_Lag1`
* `Rolling_Mean_3`

Missing values generated during feature creation were filled using column mean values.

### 5. Linear Regression Model

Performed chronological train-test split:

* Training Data: First 80%
* Testing Data: Last 20%

Model Evaluation Metrics:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score

Generated:

* Actual vs Predicted visualization

### 6. Cross Validation

Performed:

* 5-Fold Cross Validation

Reported:

* Individual Fold R² Scores
* Mean Cross-Validation R² Score

### 7. Random Forest Optimization

Used GridSearchCV with:

```python
n_estimators = [50, 100]
max_depth = [5, 10, None]
```

Outputs:

* Best Parameters
* Best Estimator
* MAE
* RMSE
* R² Score

Visualized:

* Top 10 Feature Importances

### 8. Stationarity Testing

Applied Augmented Dickey-Fuller (ADF) Test on:

`Estimated_Deliveries`

Interpretation:

* p-value < 0.05 → Stationary Series
* p-value ≥ 0.05 → Non-Stationary Series

### 9. Forecast Comparison Table

Generated forecast table containing:

* Actual Deliveries
* Predicted Deliveries
* Error Percentage

Displayed first 20 test records.

### 10. Documentation

Completed all 12 explanation markdown cells with:

* Methodology
* Feature engineering explanation
* Model interpretation
* Results discussion
* Forecasting insights

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Statsmodels

---

## Machine Learning Models

### Linear Regression

Used as baseline forecasting model.

### Random Forest Regressor

Optimized using GridSearchCV for improved predictive performance.

---

## Key Concepts Covered

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Time Series Features
* Regression Modeling
* Cross Validation
* Hyperparameter Tuning
* Feature Importance Analysis
* Forecasting
* Stationarity Testing

---

## Output Files

* `week2_Sakshi_Kekan.ipynb`
* EDA Visualizations
* Model Evaluation Results
* Forecast Comparison Table

---

## Author

**Sakshi Kiran Kekan**
Data Science Internship – Week 2 Assignment
