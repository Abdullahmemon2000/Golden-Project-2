import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset
laptop_data = pd.read_csv("laptopPrice.csv")

# Data Exploration
print("Dataset Overview:")
print(laptop_data.head())
print("\nDataset Information:")
print(laptop_data.info())
print("\nDataset Description:")
print(laptop_data.describe())

# Feature Engineering
# Convert RAM, SSD, HDD, and Graphic Card to numerical values
laptop_data['ram_gb'] = laptop_data['ram_gb'].str.replace(' GB', '').astype(int)
laptop_data['ssd'] = laptop_data['ssd'].str.replace(' GB', '').astype(int)
laptop_data['hdd'] = laptop_data['hdd'].str.replace(' GB', '').astype(int)
laptop_data['graphic_card_gb'] = laptop_data['graphic_card_gb'].str.replace(' GB', '').astype(int)

# Encode categorical variables
label_encoders = {}
for column in ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_type', 
               'os', 'os_bit', 'weight', 'warranty', 'Touchscreen', 'msoffice', 'rating']:
    le = LabelEncoder()
    laptop_data[column] = le.fit_transform(laptop_data[column])
    label_encoders[column] = le

# Define features (X) and target (y)
X = laptop_data.drop(columns=['Price'])
y = laptop_data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Train models and evaluate performance
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[model_name] = {"MAE": mae, "RMSE": rmse}

# Display the model evaluation results
print("\nModel Evaluation Results:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")

# Cross-validation
cv_results = {}
for model_name, model in models.items():
    # Perform 5-fold cross-validation for MAE and RMSE
    cv_mae = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    
    # Store results
    cv_results[model_name] = {
        "CV MAE": -cv_mae.mean(),
        "CV RMSE": -cv_rmse.mean()
    }

# Display the cross-validation results
print("\nCross-Validation Results:")
for model_name, metrics in cv_results.items():
    print(f"{model_name}:")
    print(f"  Cross-Validated MAE: {metrics['CV MAE']:.2f}")
    print(f"  Cross-Validated RMSE: {metrics['CV RMSE']:.2f}")
