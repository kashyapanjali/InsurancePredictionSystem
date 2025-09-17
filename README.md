````markdown
# Insurance Claim Prediction System

**A web-based application to predict insurance claim approval using machine learning and interactive analytics.**

---

## Project Overview

This project predicts whether an insurance claim is likely to be approved based on customer details such as age, BMI, smoker status, number of children, and region. It also provides an **interactive dashboard** for data analytics and visualization.

---

## Features

- **Claim Prediction:**  
  Enter customer details and get a prediction whether the claim will be approved (high charges expected) or not.

- **Probability Score:**  
  Shows the likelihood of claim approval for better decision-making.

- **Analytics Dashboard:**  
  Visualizes historical insurance data with charts like:
  - Age vs Charges (with smoker status)
  - Charges Distribution
  - Average Charges by Region

---

## Technologies Used

- Python
- Pandas & NumPy (Data handling)
- Scikit-learn (Machine Learning: Random Forest & Logistic Regression)
- Streamlit (Web app & dashboard)
- Matplotlib & Seaborn (Data visualization)
- Joblib (Model saving & loading)

---

## How to Run

1. Clone the repository.
2. Make sure you have Python 3.x installed.
3. Install required packages:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn joblib
```
````

4. Run the app:

```bash
streamlit run insurance_app.py
```

```bash
Local URL: http://localhost:8502
```

5. Open the link in your browser and use the interactive dashboard.

---

## Dataset

- The app uses the `insurance.csv` dataset with the following columns:
  `age, sex, bmi, children, smoker, region, charges`

- Target variable (`approved`) is created based on **charges > 10,000**.

---

## Outcome

- Predicts claim approval based on customer data.
- Provides analytics and insights about insurance charges.
- Demonstrates end-to-end **data science workflow** from data prep to web deployment.
