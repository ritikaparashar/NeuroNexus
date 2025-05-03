import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("tested.csv")  # Use train.csv from Kaggle if tested.csv is edited

# Drop known leakage or irrelevant columns
leaky_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
df.drop(columns=[col for col in leaky_cols if col in df.columns], inplace=True)

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical columns
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Drop any rows with NaNs (if any remain)
df.dropna(inplace=True)

# Keep only a minimal set of clean features
features = ['Pclass', 'Sex', 'Age', 'Fare']
if 'Survived' not in df.columns:
    raise ValueError("Dataset must contain 'Survived' column.")

X = df[features]
y = df['Survived']

# Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# Train XGBoost with intentionally reduced power
model = XGBClassifier(
    n_estimators=20,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.5,
    reg_alpha=1.0,
    reg_lambda=1.5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n Accuracy: {acc * 100:.2f}% (Expected ~90-94%)")
print("\n Classification Report:")
print(classification_report(y_test, y_pred))
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', color='skyblue')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(True)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
