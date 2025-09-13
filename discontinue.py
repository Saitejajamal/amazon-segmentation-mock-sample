# student_dropout_predictor_rule.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("📚 Student Dropout Predictor with Attendance Rule")

# Generate synthetic dataset
def generate_data(n=300):
    np.random.seed(42)
    attendance = np.random.rand(n)
    wellness = np.random.rand(n)
    digital = np.random.rand(n)
    y = (attendance + wellness + digital < 1.2).astype(int)  # dropout=1 if overall low
    df = pd.DataFrame({
        "attendance": attendance,
        "wellness": wellness,
        "digital_footprint": digital,
        "discontinued": y
    })
    return df

# Load or generate data
st.sidebar.header("Data Options")
use_synthetic = st.sidebar.checkbox("Use sample data", value=True)
if use_synthetic:
    df = generate_data(500)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

st.write("### Data Preview")
st.dataframe(df.head())

# Train model
X = df.drop("discontinued", axis=1)
y = df["discontinued"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"✅ Model trained. Test Accuracy: **{acc:.2f}**")

# Prediction section with attendance rule
st.write("### Try a new prediction")
attendance = st.slider("Attendance (0=low, 1=high)", 0.0, 1.0, 0.5)
wellness = st.slider("Wellness (0=low, 1=high)", 0.0, 1.0, 0.5)
digital = st.slider("Digital footprint (0=low, 1=high)", 0.0, 1.0, 0.5)

new_data = np.array([[attendance, wellness, digital]])
pred = model.predict(new_data)[0]
prob = model.predict_proba(new_data)[0][1]

# Rule-based override: always flag if attendance < 0.4
if attendance < 0.4:
    pred = 1
    prob = max(prob, 0.95)  # optional: boost probability

if pred == 1:
    st.error(f"⚠️ High Risk of Discontinuation (prob={prob:.2f}) — flagged due to low attendance" 
             if attendance < 0.4 else f"⚠️ High Risk of Discontinuation (prob={prob:.2f})")
else:
    st.success(f"👍 Likely to Continue (prob={prob:.2f})")
