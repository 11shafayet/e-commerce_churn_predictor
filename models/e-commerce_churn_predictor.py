import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from xgboost import XGBClassifier


# ----------------------------
# 1. Load data
# ----------------------------
features_df = pd.read_csv("data/raw/ecommerce_customer_features.csv")
churn_df = pd.read_csv("data/raw/ecommerce_customer_targets.csv")


# ----------------------------
# 2. Merge datasets
# ----------------------------
df = pd.merge(features_df, churn_df, on="Customer_ID")

print("Dataset shape:", df.shape)
print(df.head())


# ----------------------------
# 3. Drop unnecessary columns
# ----------------------------
if "Customer_ID" in df.columns:
    df = df.drop(columns=["Customer_ID"])


# ----------------------------
# 4. Clean and encode categorical variables
# ----------------------------
df["churned"] = df["churned"].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})

df["loyalty_member"] = (
    df["loyalty_member"].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
)


# ----------------------------
# 5. Handle missing values
# ----------------------------
print("\nMissing values before handling:")
print(df.isnull().sum())

# Separate features and target
X = df.drop("churned", axis=1)
y = df["churned"]


# ----------------------------
# 6. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ----------------------------
# 7. Logistic Regression (baseline)
# ----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)


# ----------------------------
# 8. Logistic Regression (balanced)
# ----------------------------
log_balanced_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_balanced_model.fit(X_train, y_train)

y_pred_log_balanced = log_balanced_model.predict(X_test)


# ----------------------------
# 9. Random Forest
# ----------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# ----------------------------
# 10. XGBoost model
# ----------------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    scale_pos_weight=5,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)


# ----------------------------
# 11. Evaluation function
# ----------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))


# ----------------------------
# 12. Evaluate all models
# ----------------------------
evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Logistic Regression (Balanced)", y_test, y_pred_log_balanced)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)


# ----------------------------
# 13. Feature importance (Random Forest)
# ----------------------------
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))
