import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import numpy as np

# LOAD DATA
df = pd.read_csv("pose_dataset.csv")

# =========================
# ORIGINAL LABEL MAP
# =========================
label_map = {
    "FRONT": 0,
    "SIDE": 1,
    "WRONG": 2
}

df["label"] = df["label"].map(label_map)

# DROP NaN
df = df.dropna(subset=["label"])

# =========================
# 🔥 FIX: REMAP TO CONTIGUOUS LABELS
# =========================
unique_classes = sorted(df["label"].unique())
remap = {old: new for new, old in enumerate(unique_classes)}

df["label"] = df["label"].map(remap)

print("Original classes:", unique_classes)
print("Remapped classes:", remap)

# =========================
# FEATURES / TARGET
# =========================
X = df.drop("label", axis=1)
y = df["label"].astype(int)

num_classes = len(remap)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
if num_classes == 2:
    print("Binary classification mode")
    model = XGBClassifier(
        n_estimators=200,
        objective="binary:logistic"
    )
else:
    print("Multiclass classification mode")
    model = XGBClassifier(
        n_estimators=200,
        objective="multi:softmax",
        num_class=num_classes
    )

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# =========================
# SAVE EVERYTHING
# =========================
joblib.dump(model, "pose_model.pkl")
joblib.dump(remap, "label_remap.pkl")         # 🔥 IMPORTANT
joblib.dump(unique_classes, "original_labels.pkl")

print("Model saved!")