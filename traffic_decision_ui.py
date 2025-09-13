# filename: traffic_signal_advanced.py

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt

# -----------------------
# Synthetic Training Data
# -----------------------
data = pd.DataFrame({
    'num_cars':    [5, 15, 25, 40, 60, 80, 100, 120, 10, 50, 90, 30],
    'wait_time':   [5, 10, 15, 20, 25, 35, 45, 60, 5, 30, 40, 15],
    'emergency':   [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    'pedestrian':  [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    'time_of_day': [0, 1, 1, 2, 2, 3, 3, 1, 0, 2, 3, 1],
    'extend_green':[0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
})

X = data.drop(columns=['extend_green'])
y = data['extend_green']

# Train decision tree
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="🚦 Smart Traffic Controller", layout="wide")
st.title("🚦 Smart Traffic Signal Decision System")
st.write("This system uses a **Decision Tree Model** plus safety rules "
         "to decide whether to extend a green light or switch lanes.")

# Sidebar input
st.sidebar.header("🔧 Traffic Input Controls")
num_cars = st.sidebar.slider("Number of Cars in Lane", 0, 200, 50)
wait_time = st.sidebar.slider("Average Waiting Time (seconds)", 0, 120, 20)
emergency = st.sidebar.selectbox("Emergency Vehicle Present?", [0, 1],
                                 format_func=lambda x: "Yes" if x==1 else "No")
pedestrian = st.sidebar.selectbox("Pedestrian Waiting?", [0, 1],
                                  format_func=lambda x: "Yes" if x==1 else "No")
time_of_day = st.sidebar.selectbox("Time of Day", [0, 1, 2, 3],
                                   format_func=lambda x: ["Night","Morning","Afternoon","Evening"][x])

# -----------------------
# Prediction with Rules
# -----------------------
input_features = [[num_cars, wait_time, emergency, pedestrian, time_of_day]]
prediction = model.predict(input_features)[0]
proba = model.predict_proba(input_features)[0]

# Override Rules
if pedestrian == 1 and emergency == 0:
    # Pedestrian safety rule
    prediction = 0
    proba = [1.0, 0.0]
elif emergency == 1:
    # Emergency override
    prediction = 1
    proba = [0.0, 1.0]

# -----------------------
# Display Result
# -----------------------
st.subheader("📊 Decision Result")
if emergency == 1:
    st.error("🚑 Emergency vehicle detected! Priority given.")
    st.success("✅ Extend Green Light (Emergency override applied)")
elif pedestrian == 1:
    st.error("🚶 Pedestrian waiting! Cars must STOP.")
    st.warning("⛔ Switch Lane (Pedestrian safety rule applied)")
elif prediction == 1:
    st.success(f"✅ Extend Green Light (Confidence: {proba[1]*100:.1f}%)")
else:
    st.warning(f"⛔ Switch Lane (Confidence: {proba[0]*100:.1f}%)")

# -----------------------
# Visuals
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Traffic Volume Analysis")
    traffic_data = pd.DataFrame({
        'Cars': [num_cars, 200-num_cars],
        'Type': ['This Lane', 'Other Lanes']
    })
    st.bar_chart(traffic_data.set_index("Type"))

with col2:
    st.subheader("🌳 Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(8,6))
    tree.plot_tree(model, feature_names=X.columns, class_names=["Switch","Extend"],
                   filled=True, fontsize=8)
    st.pyplot(fig)

# Show decision rules
st.subheader("📜 Decision Tree Rules")
rules = export_text(model, feature_names=list(X.columns))
st.code(rules)
