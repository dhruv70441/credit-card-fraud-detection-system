import streamlit as st
import pandas as pd
from prediction_pipeline import FraudModelPipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="💳 Fraud Detection App", layout="centered")

# ---------------- CSS ANIMATIONS ----------------
st.markdown("""
<style>
/* Smooth background transition */
[data-testid="stAppViewContainer"] {
    transition: background-color 1s ease;
}

/* Fraud found - red background */
.red-bg {
    background-color: #ffcccc !important;
}

/* No fraud - green background */
.green-bg {
    background-color: #ccffcc !important;
}

/* Fade-in for results */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-section {
    animation: fadeIn 0.7s ease;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI HEADER ----------------
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Upload your transaction CSV and detect potential frauds in seconds ⚡")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload a CSV file for prediction", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Initialize and run pipeline
    pipeline = FraudModelPipeline()
    result_df, preds = pipeline.predict(df)

    # Detect fraud presence
    fraud_found = (result_df["fraud_prediction"] == 1).any()

    # Select background color
    bg_color = "red-bg" if fraud_found else "green-bg"

    # Inject JavaScript to change background
    st.markdown(
        f"""
        <script>
        const app = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
        if (app) {{
            app.classList.remove('red-bg', 'green-bg');
            app.classList.add('{bg_color}');
        }}
        </script>
        """,
        unsafe_allow_html=True
    )

    # ---------------- RESULTS DISPLAY ----------------
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.subheader("🧾 Prediction Results Preview")
    st.dataframe(result_df[['amt', 'fraud_probability', 'fraud_prediction']].head(10))
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- DOWNLOAD BUTTON ----------------
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Results",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

    # ---------------- RESET BUTTON ----------------
    if st.button("✅ Done / Reset"):
        st.markdown("""
        <script>
        const app = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
        if (app) { app.classList.remove('red-bg', 'green-bg'); }
        </script>
        """, unsafe_allow_html=True)
        st.rerun()
