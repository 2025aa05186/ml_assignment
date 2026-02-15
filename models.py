import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ----------------------------
# Load dataset (MANUAL)
# ----------------------------
df = pd.read_csv("data/train.csv")
print("Dataset loaded!")


# ----------------------------
# Split X and y
# ----------------------------
X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()

feature_columns = list(X.columns)


# ----------------------------
# Encode categorical columns
# ----------------------------
encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le


# ----------------------------
# Handle missing values
# ----------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


# ----------------------------
# Scale
# ----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ----------------------------
# Encode target (important for XGBoost)
# ----------------------------
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


# ----------------------------
# Train test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# Initialize models
# ----------------------------
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
nb = GaussianNB()
rf = RandomForestClassifier()
xgb = XGBClassifier(eval_metric="logloss")


# ----------------------------
# Train
# ----------------------------
print("Training...")

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

print("Training completed!")


# ----------------------------
# Save separately (Method 1)
# ----------------------------
joblib.dump(lr, "logistic_regression.pkl")
joblib.dump(dt, "decision_tree.pkl")
joblib.dump(knn, "knn.pkl")
joblib.dump(nb, "naive_bayes.pkl")
joblib.dump(rf, "random_forest.pkl")
joblib.dump(xgb, "xgboost.pkl")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")


print("All models saved!")