import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ----------------------------------
# Custom CSS
# ----------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 25px;
    }

    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .info-box {
        background-color: #f3f4f6;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #2563eb;
        margin-bottom: 20px;
    }

    .warning-box {
        background-color: #fff7ed;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #f97316;
        margin-bottom: 20px;
    }

    .success-box {
        background-color: #ecfdf5;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #10b981;
        margin-bottom: 20px;
    }

    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }

    .small-text {
        color: #6b7280;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# Load saved model files
# ----------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("fraud_app_files/random_forest_fraud_model.joblib")
    scaler = joblib.load("fraud_app_files/scaler.joblib")

    with open("fraud_app_files/feature_names.json", "r") as f:
        feature_names = json.load(f)

    with open("fraud_app_files/threshold.json", "r") as f:
        threshold_data = json.load(f)

    threshold = threshold_data["threshold"]

    return model, scaler, feature_names, threshold


model, scaler, feature_names, threshold = load_model_files()

# ----------------------------------
# Header
# ----------------------------------
st.markdown('<div class="main-title">💳 Credit Card Fraud Detection App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload transaction data and predict whether each transaction is legitimate or fraudulent.</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="info-box">
        <b>Selected Model:</b> Random Forest<br>
        <b>Decision Threshold:</b> {threshold}<br>
        <b>Goal:</b> Detect fraudulent credit card transactions using machine learning.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="warning-box">
        <b>Note:</b> This app only displays previews of uploaded data and prediction results.
        The full prediction file can still be downloaded after prediction.
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.title("App Information")
st.sidebar.write("This app was built for a credit card fraud detection final project.")
st.sidebar.write("The model predicts fraud using transaction features from the Kaggle credit card fraud dataset.")

st.sidebar.markdown("---")
st.sidebar.write("### Expected Columns")
st.sidebar.write("The uploaded CSV should contain:")
st.sidebar.write("`Time`, `V1` to `V28`, and `Amount`")
st.sidebar.write("The `Class` column is optional.")

st.sidebar.markdown("---")
st.sidebar.write("### Prediction Labels")
st.sidebar.write("0 = Legitimate")
st.sidebar.write("1 = Fraudulent")

# ----------------------------------
# File upload
# ----------------------------------
st.markdown('<div class="section-header">1. Upload Transaction CSV</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transaction features",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.markdown(
            """
            <div class="success-box">
                File uploaded successfully. Review the file information below before running prediction.
            </div>
            """,
            unsafe_allow_html=True
        )

        # File information
        st.markdown('<div class="section-header">2. Uploaded File Summary</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", data.shape[0])

        with col2:
            st.metric("Columns", data.shape[1])

        with col3:
            if "Class" in data.columns:
                st.metric("Actual Fraud Cases", int(data["Class"].sum()))
            else:
                st.metric("Actual Fraud Cases", "N/A")

        # Preview
        st.write("### Uploaded Data Preview")
        st.dataframe(data.head(10), use_container_width=True)

        # Prepare data
        if "Class" in data.columns:
            data_for_prediction = data.drop(columns=["Class"])
        else:
            data_for_prediction = data.copy()

        missing_cols = [
            col for col in feature_names
            if col not in data_for_prediction.columns
        ]

        extra_cols = [
            col for col in data_for_prediction.columns
            if col not in feature_names
        ]

        if missing_cols:
            st.error("The uploaded CSV is missing required columns.")
            st.write(missing_cols)

        else:
            if extra_cols:
                st.warning("The uploaded CSV has extra columns. These columns will be ignored.")
                st.write(extra_cols)

            data_for_prediction = data_for_prediction[feature_names]

            st.markdown('<div class="section-header">3. Run Prediction</div>', unsafe_allow_html=True)

            run_prediction = st.button("Run Fraud Prediction", type="primary")

            if run_prediction:

                with st.spinner("Generating predictions..."):

                    data_scaled = scaler.transform(data_for_prediction)

                    fraud_probabilities = model.predict_proba(data_scaled)[:, 1]

                    predictions = (fraud_probabilities >= threshold).astype(int)

                    results = data.copy()
                    results["Fraud Probability"] = fraud_probabilities
                    results["Prediction"] = predictions
                    results["Prediction Label"] = results["Prediction"].map(
                        {
                            0: "Legitimate",
                            1: "Fraudulent"
                        }
                    )

                st.markdown(
                    """
                    <div class="success-box">
                        Predictions completed successfully.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown('<div class="section-header">4. Prediction Summary</div>', unsafe_allow_html=True)

                total_count = len(results)
                fraud_count = int(results["Prediction"].sum())
                legit_count = total_count - fraud_count
                fraud_rate = fraud_count / total_count if total_count > 0 else 0

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Transactions", total_count)

                with col2:
                    st.metric("Predicted Fraud", fraud_count)

                with col3:
                    st.metric("Predicted Legitimate", legit_count)

                with col4:
                    st.metric("Predicted Fraud Rate", f"{fraud_rate:.2%}")

                # If Class exists, show simple actual vs predicted comparison
                if "Class" in results.columns:
                    actual_fraud_count = int(results["Class"].sum())
                    st.write("### Actual vs Predicted Fraud Count")

                    comparison_counts = pd.DataFrame(
                        {
                            "Category": ["Actual Fraud", "Predicted Fraud"],
                            "Count": [actual_fraud_count, fraud_count]
                        }
                    )

                    st.dataframe(comparison_counts, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(comparison_counts["Category"], comparison_counts["Count"])
                    ax.set_ylabel("Count")
                    ax.set_title("Actual vs Predicted Fraud Count")
                    st.pyplot(fig)

                # Prediction distribution chart
                st.write("### Prediction Distribution")

                prediction_counts = results["Prediction Label"].value_counts().reset_index()
                prediction_counts.columns = ["Prediction Label", "Count"]

                st.dataframe(prediction_counts, use_container_width=True)

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.bar(prediction_counts["Prediction Label"], prediction_counts["Count"])
                ax2.set_ylabel("Count")
                ax2.set_title("Predicted Legitimate vs Fraudulent Transactions")
                st.pyplot(fig2)

                # Results preview
                st.markdown('<div class="section-header">5. Prediction Results Preview</div>', unsafe_allow_html=True)

                st.write("Showing first 50 rows only.")
                st.dataframe(results.head(50), use_container_width=True)

                # Predicted fraud preview
                st.write("### Predicted Fraud Cases Preview")

                fraud_cases = results[results["Prediction"] == 1]

                if len(fraud_cases) > 0:
                    st.write("Showing first 50 predicted fraud cases only.")
                    st.dataframe(fraud_cases.head(50), use_container_width=True)
                else:
                    st.info("No fraudulent transactions were predicted.")

                # Download full results
                st.markdown('<div class="section-header">6. Download Results</div>', unsafe_allow_html=True)

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Full Prediction Results",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error("An error occurred while processing the file.")
        st.write(e)

else:
    st.info("Please upload a CSV file to begin.")