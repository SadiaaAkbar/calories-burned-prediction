"""
Calories Burned Predictor — full ML pipeline
Clean -> EDA -> Feature Engineering -> Train multiple models -> Evaluate -> Save best
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------------
df = pd.read_csv("data/gym_data.csv")
print(f"Raw shape: {df.shape}")

before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

print("\nMissing values before imputation:")
print(df.isna().sum()[df.isna().sum() > 0])

# Fat_Percentage: impute with median within Gender group (more accurate than global median)
df["Fat_Percentage"] = df.groupby("Gender")["Fat_Percentage"].transform(
    lambda x: x.fillna(x.median())
)

# Sanity-range filter (removes any physiologically impossible rows)
df = df[(df["Age"].between(15, 90)) & (df["BMI"].between(10, 60))]
df = df.reset_index(drop=True)
print(f"\nClean shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. EDA (saved as figures for the README / notebook)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.histplot(df["Calories_Burned"], kde=True, ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Distribution of Calories Burned")

sns.scatterplot(
    data=df, x="Session_Duration (hours)", y="Calories_Burned",
    hue="Workout_Type", alpha=0.6, ax=axes[0, 1]
)
axes[0, 1].set_title("Session Duration vs Calories Burned")

sns.boxplot(data=df, x="Workout_Type", y="Calories_Burned", ax=axes[1, 0])
axes[1, 0].set_title("Calories Burned by Workout Type")

corr_cols = ["Age", "Weight (kg)", "Avg_BPM", "Resting_BPM", "Session_Duration (hours)",
             "Fat_Percentage", "Water_Intake (liters)", "BMI", "Calories_Burned"]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("Correlation Matrix")

plt.tight_layout()
plt.savefig("figures/eda_overview.png", dpi=120)
plt.close()
print("\nSaved figures/eda_overview.png")

print("\nTop correlations with Calories_Burned:")
print(df[corr_cols].corr()["Calories_Burned"].sort_values(ascending=False))

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["HR_Reserve"] = df["Max_BPM"] - df["Resting_BPM"]
df["HR_Intensity"] = (df["Avg_BPM"] - df["Resting_BPM"]) / df["HR_Reserve"]
df["Calories_per_Minute"] = df["Calories_Burned"] / (df["Session_Duration (hours)"] * 60)
df["Weight_x_Duration"] = df["Weight (kg)"] * df["Session_Duration (hours)"]

feature_cols_num = [
    "Age", "Weight (kg)", "Height (m)", "Max_BPM", "Avg_BPM", "Resting_BPM",
    "Session_Duration (hours)", "Fat_Percentage", "Water_Intake (liters)",
    "Workout_Frequency (days/week)", "Experience_Level", "BMI",
    "HR_Reserve", "HR_Intensity", "Weight_x_Duration",
]
feature_cols_cat = ["Gender", "Workout_Type"]
target_col = "Calories_Burned"

X = df[feature_cols_num + feature_cols_cat]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), feature_cols_num),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), feature_cols_cat),
])

# ---------------------------------------------------------------------------
# 4. TRAIN MULTIPLE MODELS
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.5),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
    "SVR (RBF)": SVR(kernel="rbf", C=100, epsilon=10),
}

results = []
fitted_pipelines = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        "model": name,
        "cv_r2_mean": round(cv_r2.mean(), 4),
        "cv_r2_std": round(cv_r2.std(), 4),
        "test_mae": round(mae, 1),
        "test_rmse": round(rmse, 1),
        "test_r2": round(r2, 4),
    })
    fitted_pipelines[name] = pipe
    print(f"{name:20s}  CV R2={cv_r2.mean():.3f}±{cv_r2.std():.3f}  Test R2={r2:.3f}  MAE={mae:.1f}  RMSE={rmse:.1f}")

results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
results_df.to_csv("models/model_comparison.csv", index=False)
print("\n=== Model comparison (sorted by test R2) ===")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. PICK BEST MODEL + SAVE
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["model"]
best_pipe = fitted_pipelines[best_name]
print(f"\nBest model: {best_name}")

joblib.dump(best_pipe, "models/best_model.joblib")
joblib.dump(feature_cols_num, "models/feature_cols_num.joblib")
joblib.dump(feature_cols_cat, "models/feature_cols_cat.joblib")

with open("models/metadata.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "metrics": results_df.iloc[0].to_dict(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }, f, indent=2)

# Feature importance plot for the winning tree-based model (if applicable)
if best_name in ("Random Forest", "Gradient Boosting"):
    ohe_names = list(best_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(feature_cols_cat))
    all_names = feature_cols_num + ohe_names
    importances = best_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": all_names, "importance": importances}).sort_values("importance", ascending=False).head(12)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", color="#55A868")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig("figures/feature_importance.png", dpi=120)
    plt.close()
    print("Saved figures/feature_importance.png")

print("\nDone. Best model saved to models/best_model.joblib")
