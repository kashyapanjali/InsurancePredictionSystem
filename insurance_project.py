import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ----------------------
# Data Preparation
# ----------------------
data = pd.read_csv("insurance.csv")

print("✅ First 5 rows of data:\n", data.head())
print("\n✅ Dataset Info:")
print(data.info())
print("\n✅ Missing values:")
print(data.isnull().sum())

# ----------------------
# Exploratory Data Analysis (EDA)
# ----------------------

# Summary statistics
print("\n✅ Summary Statistics:")
print(data.describe())

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(data['age'], bins=5, kde=True)
plt.title("Age Distribution")
plt.show()

# Charges vs Age
plt.figure(figsize=(6,4))
sns.scatterplot(x='age', y='charges', data=data, hue='smoker')
plt.title("Charges vs Age (Smoker/Non-Smoker)")
plt.show()

# Average charges by region
plt.figure(figsize=(6,4))
sns.barplot(x='region', y='charges', data=data, estimator=np.mean, ci=None)
plt.title("Average Charges by Region")
plt.show()

# ----------------------
# Feature Engineering
# ----------------------

# Copy dataset
df = data.copy()

# Encode categorical variables
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

print("\n✅ After Encoding:")
print(df.head())

# Create target column (Approved = 1 if charges > 10000 else 0)
df['approved'] = np.where(df['charges'] > 10000, 1, 0)

print("\n✅ Target Column Added:")
print(df[['age','charges','approved']])

# Drop unused column 'charges'
X = df.drop(['charges','approved'], axis=1)  # Features
y = df['approved']                           # Target

print("\n✅ Features (X):")
print(X.head())

print("\n✅ Target (y):")
print(y.head())

# ==============================
# Model Training
# ==============================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✅ Training set size:", X_train.shape)
print("✅ Test set size:", X_test.shape)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# ==============================
# Random Forest + Model Comparison
# ==============================
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score

print("\n✅ Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# Compare both models
log_acc = accuracy_score(y_test, y_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n📊 Model Comparison:")
print("Logistic Regression Accuracy:", log_acc)
print("Random Forest Accuracy:", rf_acc)

best_model = "Random Forest" if rf_acc > log_acc else "Logistic Regression"
print("\n🔥 Best Performing Model:", best_model)

# ==============================
# Save Best Model + Predict New Data
# ==============================
import joblib

# Save the best model
if best_model == "Random Forest":
    joblib.dump(rf_model, "best_insurance_model.pkl")
    final_model = rf_model
else:
    joblib.dump(model, "best_insurance_model.pkl")
    final_model = model

print("\n✅ Best model saved as best_insurance_model.pkl")

# ----------------------
# Predict on New Data
# ----------------------

# Example new customer
new_customer = pd.DataFrame({
    'age': [29],
    'sex': [1],        
    'bmi': [27.5],
    'children': [2],
    'smoker': [0],     
    'region': [2]     
})

print("\n🆕 New Customer Data:\n", new_customer)

prediction = final_model.predict(new_customer)

if prediction[0] == 1:
    print("\n💡 Prediction: Claim Approved (High Charges Expected)")
else:
    print("\n💡 Prediction: Claim Not Approved (Low Charges Expected)")
