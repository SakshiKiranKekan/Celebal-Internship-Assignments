Wee 2 Task :
"1. Load tesla_deliveries_dataset_2015_2025.csv into a DataFrame. Print shape, columns, .info(), and .describe().
2. Check for missing values and duplicate rows.
3. Create 5 EDA charts: deliveries by Model, by Region, correlation heatmap, Production vs Deliveries scatter, and time-trend line. All charts need titles and axis labels.
4. Encode Region, Model, Source_Type using LabelEncoder. Create Deliveries_Lag1 (shift by 1, fill NaN with mean) and Rolling_Mean_3 (3-period rolling mean, fill NaN with mean).
5. Split data chronologically 80/20 using index slicing — NOT train_test_split. Train Linear Regression, print MAE, RMSE, R², and plot Actual vs Predicted.
6. Run 5-Fold Cross Validation on Linear Regression. Print per-fold R² and mean.
7. Run GridSearchCV on RandomForestRegressor with n_estimators=[50,100], max_depth=[5,10,None]. Train best model, evaluate, and plot top 10 feature importances.
8. Run ADF stationarity test on Estimated_Deliveries. Interpret the p-value.
9. Build a forecast table comparing actual vs predicted for first 20 test records with error %.
10. Fill all 12 explanation markdown cells. Save as week2_<your_name>.ipynb — zero errors top to bottom."
