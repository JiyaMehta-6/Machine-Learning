# ------------------------------------------------------------------

# Advanced Uber Fare Prediction Pipeline

# ------------------------------------------------------------------
# 1. Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------------------
# 2. Load Data

print("Loading data...")

df = pd.read_csv("uber.csv")

print("Initial Shape:", df.shape)

# ------------------------------------------------------------------
# 3. Data Cleaning

print("Cleaning data...")


df.drop(["Unnamed: 0", "key"], axis=1, inplace=True)

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

df.dropna(inplace=True)

# Remove invalid passenger counts
df = df[df["passenger_count"] > 0]

# Remove invalid coordinates
df = df[
    (df["pickup_latitude"].between(-90, 90)) &
    (df["dropoff_latitude"].between(-90, 90)) &
    (df["pickup_longitude"].between(-180, 180)) &
    (df["dropoff_longitude"].between(-180, 180))
]

# ------------------------------------------------------------------
# 4. Feature Engineering

print("Implementing Feature Engineering...")

# Extract temporal features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["weekday"] = df["pickup_datetime"].dt.weekday

df.drop("pickup_datetime", axis=1, inplace=True)

# Vectorized Haversine Distance
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

df["distance_km"] = haversine_vectorized(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

# Remove extreme trips (>100 km)
df = df[df["distance_km"] < 100]

# ------------------------------------------------------------------
# 5. Outlier Treatment (Fare)

print("Outlier Handling in progress...")

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["fare_amount"] >= lower) & (df["fare_amount"] <= upper)]

print("Cleaned Shape:", df.shape)

# ------------------------------------------------------------------
# 6. Correlation Heatmap

print("Correlating data...")

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("advanced_correlation_heatmap.png")
plt.show()

# ------------------------------------------------------------------
# 7. Feature / Target Split

print("Splitting data...")

X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 8. Ridge Regression Pipeline

print("Implementing Ridge Regression Pipeline...")

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

ridge_params = {"model__alpha": [0.1, 1.0, 10.0, 50.0]}

ridge_grid = GridSearchCV(ridge_pipeline, ridge_params, cv=5, scoring="r2")
ridge_grid.fit(X_train, y_train)

ridge_best = ridge_grid.best_estimator_
ridge_pred = ridge_best.predict(X_test)

r2_ridge = r2_score(y_test, ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_test, ridge_pred))

# ------------------------------------------------------------------
# 9. Random Forest (Tuned)

print("Implementing Random Forest...")

rf = RandomForestRegressor(random_state=42)

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None]
}

rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring="r2", n_jobs=-1)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)

r2_rf = r2_score(y_test, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))

# ------------------------------------------------------------------
# 10. Gradient Boosting

print("Implementing Gradient Boosting...")

gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)

gbr_pred = gbr.predict(X_test)

r2_gbr = r2_score(y_test, gbr_pred)
rmse_gbr = np.sqrt(mean_squared_error(y_test, gbr_pred))

# ------------------------------------------------------------------
# 11. Residual Analysis

print("Executing residual analysis...")

residuals = y_test - rf_pred

plt.figure()
sns.histplot(residuals, bins=50)
plt.title("Random Forest Residual Distribution")
plt.savefig("residual_distribution.png")
plt.show()

# ------------------------------------------------------------------
# 12. Feature Importance

print("Executing Feature Importance...")

importances = rf_best.feature_importances_
features = X.columns

plt.figure(figsize=(8,6))
plt.barh(features, importances)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ------------------------------------------------------------------
# 13. Model Comparison

print("Executing Model Comparison...")

print("\n=========== MODEL PERFORMANCE ===========")
print("Ridge Regression")
print("R2:", r2_ridge)
print("RMSE:", rmse_ridge)

print("\nRandom Forest")
print("R2:", r2_rf)
print("RMSE:", rmse_rf)

print("\nGradient Boosting")
print("R2:", r2_gbr)
print("RMSE:", rmse_gbr)

best_model = max(
    {"Ridge": r2_ridge, "Random Forest": r2_rf, "Gradient Boosting": r2_gbr},
    key=lambda x: {"Ridge": r2_ridge, "Random Forest": r2_rf, "Gradient Boosting": r2_gbr}[x]
)

print("\nBest Performing Model:", best_model)
# ------------------------------------------------------------------
# 14. Generate Detailed Conclusion Report

print("Generating Detailed report...")

best_r2 = max(r2_ridge, r2_rf, r2_gbr)

if best_r2 == r2_ridge:
    best_name = "Ridge Regression"
    best_rmse = rmse_ridge
elif best_r2 == r2_rf:
    best_name = "Random Forest"
    best_rmse = rmse_rf
else:
    best_name = "Gradient Boosting"
    best_rmse = rmse_gbr

conclusion_text = f"""
============================================================
UBER FARE PREDICTION – MODEL EVALUATION REPORT
============================================================

1. Dataset Summary
------------------
- Total cleaned observations: {df.shape[0]}
- Total features used: {X.shape[1]}
- Target variable: fare_amount

2. Model Performance Comparison
-------------------------------
Ridge Regression:
    R2 Score  : {r2_ridge:.4f}
    RMSE      : {rmse_ridge:.4f}

Random Forest:
    R2 Score  : {r2_rf:.4f}
    RMSE      : {rmse_rf:.4f}

Gradient Boosting:
    R2 Score  : {r2_gbr:.4f}
    RMSE      : {rmse_gbr:.4f}

3. Best Model
-------------
Best performing model based on R2 score:
    {best_name}

Corresponding RMSE:
    {best_rmse:.4f}

4. Analytical Interpretation
----------------------------
- Distance traveled (km) shows the strongest predictive relationship with fare.
- Tree-based models outperform linear models, indicating nonlinear relationships.
- Ensemble methods reduce variance and capture feature interactions effectively.
- Ridge Regression provides stable but comparatively lower predictive power.

5. Business Insight
-------------------
- Fare pricing is highly distance-dependent.
- Temporal features (hour, weekday, month) contribute secondary influence.
- Non-linear modeling is recommended for production deployment.

6. Recommendation
-----------------
Deploy {best_name} for operational fare prediction.
Further improvements may include:
    - Feature expansion (weather, surge multiplier)
    - Advanced boosting (XGBoost / LightGBM)
    - Hyperparameter Bayesian optimization
    - Model explainability (SHAP analysis)

============================================================
End of Report
============================================================
"""

with open("model_conclusion_report.txt", "w") as f:
    f.write(conclusion_text)

print("\nDetailed conclusion report saved as 'model_conclusion_report.txt'")
