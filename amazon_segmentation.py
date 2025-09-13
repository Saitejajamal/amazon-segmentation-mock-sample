# amazon_segmentation_ui.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Amazon Customer Segmentation ", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #ffffff;}
    h1, h2, h3, h4 {color: #131921;}
    .stButton>button {background-color: #FF9900; color: black; font-weight: bold;}
    .stSlider>div>div>div>div {color: #131921;}
    .stDataFrame {border: 2px solid #FF9900; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 style='color:#FF9900;'>Amazon Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Segment customers based on **Frequency, Value, and Time on Site** to optimize recommendations and marketing campaigns.")

# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("Controls")
num_customers = st.sidebar.slider("Number of Customers", 50, 1000, 300)
max_k = st.sidebar.slider("Max K for Elbow Method", 2, 10, 6)
k = st.sidebar.slider("Choose K (Number of Clusters)", 2, max_k, 4)

# -----------------------------
# Generate Dummy Customer Data
# -----------------------------
np.random.seed(42)
data = pd.DataFrame({
    "Frequency": np.random.randint(1, 30, num_customers),
    "Value": np.random.randint(100, 5000, num_customers),
    "Time_on_Site": np.random.uniform(1, 60, num_customers)
})

st.markdown("### Sample Customer Data")
st.dataframe(data.head(10))

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []
for i in range(1, max_k+1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

st.markdown("### Elbow Method to Choose Optimal K")
fig, ax = plt.subplots()
ax.plot(range(1, max_k+1), wcss, marker='o', color='#FF9900')
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Method")
st.pyplot(fig)

# -----------------------------
# K-Means Clustering
# -----------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

st.markdown("### Clustered Customer Data")
st.dataframe(data.head(10))

# -----------------------------
# Cluster Visualization (Plotly)
# -----------------------------
st.markdown("### Customer Segments Visualization (3D)")
fig2 = px.scatter_3d(
    data, x="Frequency", y="Value", z="Time_on_Site",
    color="Cluster", symbol="Cluster",
    color_continuous_scale=px.colors.qualitative.Set1,
    title="3D Customer Segmentation"
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Cluster Summary Cards
# -----------------------------
st.markdown("### Cluster Insights")
cluster_summary = data.groupby("Cluster").agg({
    "Frequency": "mean",
    "Value": "mean",
    "Time_on_Site": "mean",
    "Cluster": "count"
}).rename(columns={"Cluster": "Count"}).reset_index()

for i, row in cluster_summary.iterrows():
    st.markdown(f"""
    <div style="border:2px solid #FF9900; border-radius:10px; padding:15px; margin-bottom:10px; background-color:#ffffff;">
        <h3 style="color:#FF9900;">Cluster {int(row['Cluster'])}</h3>
        <p style="color:#232F3E;"><b>Frequency:</b> {row['Frequency']:.2f}</p>
        <p style="color:#232F3E;"><b>Value:</b> {row['Value']:.2f}</p>
        <p style="color:#232F3E;"><b>Time on Site:</b> {row['Time_on_Site']:.2f} min</p>
        <p style="color:#232F3E;"><b>Number of Customers:</b> {row['Count']}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
### Possible Cluster Labels
- **C0 – Bargain Shoppers** → Low frequency, low value  
- **C1 – Premium Buyers** → High value, moderate frequency  
- **C2 – Occasional Buyers** → Low engagement overall  
- **C3 – Loyal Frequent Buyers** → High frequency and consistent spend
""")
