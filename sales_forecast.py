# save as sales_prediction_ui.py and run: streamlit run sales_prediction_ui.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("📈 Monthly Sales Prediction")

# Upload data
uploaded_file = st.file_uploader("Upload CSV file with 'month' and 'sales' columns", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Data")
    st.write(df.head())

    if 'month' not in df.columns or 'sales' not in df.columns:
        st.error("CSV must contain 'month' and 'sales' columns.")
    else:
        # Convert 'month' to numeric if not already
        df['month'] = pd.to_datetime(df['month'])
        df['month_num'] = df['month'].dt.month + (df['month'].dt.year - df['month'].dt.year.min())*12

        X = df[['month_num']]
        y = df['sales']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, label="Actual", color="blue")
        ax.plot(X_test, y_pred, label="Predicted", color="red")
        ax.set_xlabel("Month Number")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)

        # Future prediction
        st.subheader("🔮 Predict Future Sales")
        future_months = st.number_input("Enter number of months into future", min_value=1, max_value=24, value=3)

        last_month_num = df['month_num'].max()
        future_data = pd.DataFrame({'month_num': [last_month_num + i for i in range(1, future_months+1)]})
        future_pred = model.predict(future_data)

        result_df = pd.DataFrame({
            'Future Month #': future_data['month_num'],
            'Predicted Sales': future_pred
        })

        st.write(result_df)
