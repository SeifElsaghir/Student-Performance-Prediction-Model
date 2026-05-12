import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------
# CONFIG
# ---------------------------

TARGET_COL = "Performance Index"
RANDOM_STATE = 120


# ---------------------------
# 1) LOAD DATA & SHOW MISSING VALUES BEFORE HANDLING
# ---------------------------
df = pd.read_csv("Student_Perfo.csv")
print("Data loaded. Shape:", df.shape)
print("\nSUM of missing values BEFORE handling:")
print(df.isnull().sum())
print("\nTOTAL missing values (all columns):", df.isnull().sum().sum())


# ---------------------------
# 2) HANDLE MISSING VALUES
# ---------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# numeric
if len(num_cols) > 0:
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# categorical
if len(cat_cols) > 0:
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

# ---------------------------
# 3) REMOVE DUPLICATES
# ---------------------------
print("\nDuplicates before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after:", df.duplicated().sum())

# ---------------------------
# 4) ENCODING CATEGORICAL VARIABLES
# ---------------------------
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
print("\nEncoding completed for categorical columns")

# ---------------------------
# 5) SPLIT: Train/Test
# ---------------------------
X = df.drop("Performance Index", axis=1)
y = df["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)


print(f"\nSplit done. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---------------------------
# 6) OUTLIER REMOVAL (TRAINING SET ONLY) - IQR method
# ---------------------------
train_data = pd.concat([X_train, y_train], axis=1)

def remove_outliers_iqr(data, columns):
    data_clean = data.copy()
    for col in columns:
        Q1 = data_clean[col].quantile(0.25)
        Q3 = data_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data_clean = data_clean[(data_clean[col] >= lower) & (data_clean[col] <= upper)]
    return data_clean

all_cols = X_train.columns.tolist() + ["Performance Index"]
train_data_clean = remove_outliers_iqr(train_data, all_cols)

X_train = train_data_clean.drop("Performance Index", axis=1)
y_train = train_data_clean["Performance Index"]

print("Outliers removed.")



# ---------------------------
# 7) BALANCING: bin-based oversampling of training set
# ---------------------------

bins = 5
y_binned = pd.cut(y_train, bins=bins)

df_train = X_train.copy()
df_train["target"] = y_train
df_train["bin"] = y_binned

max_count = df_train["bin"].value_counts().max()

balanced_data = []
for b in df_train["bin"].unique():
    group = df_train[df_train["bin"] == b]
    group_resampled = resample(group, replace=True, n_samples=max_count, random_state=RANDOM_STATE)
    balanced_data.append(group_resampled)

df_balanced = pd.concat(balanced_data).reset_index(drop=True)

X_train = df_balanced.drop(["target", "bin"], axis=1)
y_train = df_balanced["target"]

print("Balancing completed.")

# ---------------------------
# 8) FEATURE SCALING
# ---------------------------
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("Feature scaling applied.")

# ---------------------------
# 9) TRAIN Multiple Linear Regression
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training completed.")

# ---------------------------
# 10) PREDICT & EVALUATE
# ---------------------------
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\nEvaluation on TEST set:")
print(f" R^2   = {r2_test:.4f}")
print(f" MAE   = {mae_test:.4f}")
print(f" RMSE  = {rmse_test:.4f}")

# ---------------------------
# 11) VISUALIZATION: Only FIRST plot - Actual vs Predicted (Test Set)
# ---------------------------
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, s=40, label="Predictions")
mn = min(y_test.min(), y_pred_test.min())
mx = max(y_test.max(), y_pred_test.max())
plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label="Ideal (y = x)")
plt.xlabel("Actual " + TARGET_COL)
plt.ylabel("Predicted " + TARGET_COL)
plt.title("Actual vs Predicted (Test Set)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------
# 12) TABLE: First 10 Actual vs Predicted (aligned)
# ---------------------------

results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_test
}, index=y_test.index)


results_first20 = results_df.sort_index().head(20).reset_index(drop=True)
print("\nFirst 20 Actual vs Predicted (test set):")
print(results_first20)

