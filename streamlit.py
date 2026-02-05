import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess data (for demonstration, you can comment this out if using saved models)
@st.cache_data
def load_data():
    df = pd.read_csv('Brain_Tumor.csv')
    if 'Image' in df.columns:
        df = df.drop('Image', axis=1)
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, y, scaler

X, y, scaler = load_data()

# Train models (for demo; in production, load pre-trained models)
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(scaler.transform(X), y)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(scaler.transform(X), y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(scaler.transform(X), y)

# Streamlit UI
st.title("Brain Tumor Classification App")
st.write("Enter the features below to predict brain tumor class using different ML algorithms.")

feature_names = list(X.columns)
user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    user_input.append(val)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred_logreg = logreg.predict(input_scaled)[0]
    pred_dt = dt.predict(input_scaled)[0]
    pred_knn = knn.predict(input_scaled)[0]

    st.subheader("Predictions:")
    st.write(f"**Logistic Regression:** {'Tumor' if pred_logreg else 'No Tumor'}")
    st.write(f"**Decision Tree:** {'Tumor' if pred_dt else 'No Tumor'}")
    st.write(f"**K-Nearest Neighbors:** {'Tumor' if pred_knn else 'No Tumor'}")