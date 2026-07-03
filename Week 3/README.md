# Week 3: Country Clustering Analysis using K-Means and DBSCAN

## Overview

This project performs unsupervised machine learning on a country socio-economic dataset to identify groups of countries with similar development characteristics. The notebook applies data preprocessing, feature scaling, K-Means clustering, DBSCAN clustering, and Principal Component Analysis (PCA) for visualization.

---

## Objectives

- Clean and preprocess the dataset.
- Scale numerical features using StandardScaler.
- Determine the optimal number of clusters using the Elbow Method.
- Train a K-Means clustering model.
- Evaluate clustering performance using the Silhouette Score.
- Compare results with DBSCAN clustering.
- Visualize clusters using PCA.
- Analyze socio-economic characteristics of each cluster.

---

## Dataset

**File:** `Country-data.csv`

The dataset contains socio-economic and health-related indicators for different countries, including:

- Country
- Child Mortality
- Exports
- Health
- Imports
- Income
- Inflation
- Life Expectancy
- Total Fertility
- GDP
- And other economic indicators

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Workflow

### 1. Import Required Libraries

The notebook imports:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

### 2. Load Dataset

The CSV dataset is loaded into the notebook for analysis.

---

### 3. Data Preprocessing

The following preprocessing steps are performed:

- Remove extra spaces from column names
- Remove duplicate records
- Convert numerical columns to numeric format
- Handle missing values using median imputation

---

### 4. Feature Scaling

The Country column is excluded, and all numerical features are standardized using StandardScaler.

---

### 5. Elbow Method

K-Means is trained for cluster values ranging from **2 to 10**, and inertia values are plotted to determine the optimal number of clusters.

---

### 6. K-Means Clustering

The optimal cluster count is selected as:

```
best_k = 3
```

The K-Means model assigns each country to one of three clusters.

---

### 7. Model Evaluation

The clustering quality is evaluated using the **Silhouette Score**.

---

### 8. DBSCAN Clustering

A second clustering algorithm is implemented for comparison using:

- eps = 1.5
- min_samples = 5

---

### 9. PCA Visualization

Principal Component Analysis (PCA) reduces the data to two dimensions, and the K-Means clusters are visualized using a color-coded scatter plot.

---

## Results

The notebook generates:

- Cleaned dataset
- Elbow Method graph
- K-Means clusters
- DBSCAN clusters
- Silhouette Score
- PCA visualization

---

## Observations

The analysis identifies:

- Countries with high child mortality and lower development.
- Economically strong countries with high GDP and income.
- Moderately developed countries between the two extremes.
- Relationships between health, income, fertility rate, and life expectancy.
- Distinct socio-economic clusters across countries.

---

## Output Files

- `Week3_Sakshi_Kekan.ipynb`
- `Country-data.csv`
- PCA Cluster Visualization
- Elbow Method Plot

---

## Author

**Sakshi Kiran Kekan**

B.Tech Computer Engineering  
Sanjivani College of Engineering, Kopargaon

---

## Conclusion

This project demonstrates the application of unsupervised machine learning techniques to analyze country-level socio-economic indicators. By combining K-Means, DBSCAN, and PCA, meaningful clusters are identified, enabling insights into global development patterns and economic conditions.
