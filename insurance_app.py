# ==============================
# Insurance Claim Prediction App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------------------
# Load Dataset
# ----------------------
data = pd.read_csv("insurance.csv")

st.title("🏥 Insurance Claim Prediction System")
st.write("Enter customer details to check if claim will be approved (High Charges) or not.")

# ----------------------
# EDA / Analytics Section
# ----------------------
st.subheader("📊 Data Analytics")

chart_type = st.selectbox(
    "Select chart type",
    ["Age vs Charges", "Charges Distribution", "Average Charges by Region"]
)

if chart_type == "Age vs Charges":
    fig, ax = plt.subplots()
    sns.scatterplot(x='age', y='charges', hue='smoker', data=data, ax=ax)
    ax.set_title("Charges vs Age (Smoker/Non-Smoker)")
    st.pyplot(fig)

elif chart_type == "Charges Distribution":
    fig, ax = plt.subplots()
    sns.histplot(data['charges'], bins=30, kde=True, ax=ax)
    ax.set_title("Charges Distribution")
    st.pyplot(fig)

elif chart_type == "Average Charges by Region":
    fig, ax = plt.subplots()
    sns.barplot(x='region', y='charges', data=data, estimator=np.mean, ci=None, ax=ax)
    ax.set_title("Average Charges by Region")
    st.pyplot(fig)

# ----------------------
# User Inputs
# ----------------------
age = st.slider("Age", 18, 80, 30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode inputs
sex_val = 1 if sex == "male" else 0
smoker_val = 1 if smoker == "yes" else 0
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_val = region_map[region]

new_data = pd.DataFrame({
    "age": [age],
    "sex": [sex_val],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker_val],
    "region": [region_val]
})

# ----------------------
# Train / Load Model
# ----------------------
# Encode full dataset
df = data.copy()
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])
df['approved'] = np.where(df['charges'] > 10000, 1, 0)

X = df.drop(['charges','approved'], axis=1)
y = df['approved']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate & choose best model
if accuracy_score(y_test, rf_model.predict(X_test)) >= accuracy_score(y_test, lr_model.predict(X_test)):
    final_model = rf_model
else:
    final_model = lr_model

# Save best model
joblib.dump(final_model, "best_insurance_model.pkl")

# ----------------------
# Prediction
# ----------------------
if st.button("Predict Claim Approval"):
    prediction = final_model.predict(new_data)
    if prediction[0] == 1:
        st.success("✅ Claim Approved (High Charges Expected)")
    else:
        st.error("❌ Claim Not Approved (Low Charges Expected)")
