import streamlit as st
import pickle
import numpy as np

# ==============================
# Load the trained XGBoost model
# ==============================
model = pickle.load(open("xgb_model.sav", "rb"))

def run_app():
    # ==============================
    # App Header
    # ==============================
    st.markdown(
        """
        <style>
            .main-title {
                text-align: center;
                font-size: 60px;
                font-weight: bold;
                color: red;
            }
            .subtitle {
                text-align: center;
                font-size: 22px;
                color: #117A65;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="main-title">🩺 RENALGUARD</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A Machine Learning powered tool for early detection of Chronic Kidney Disease</p>', unsafe_allow_html=True)
    st.write("---")

    # ==============================
    # Input Section
    # ==============================
    st.subheader("🔎 Enter Patient Medical Records")
    col1, col2 = st.columns(2)

    with col1:
        sg = st.number_input("Specific Gravity (Sg)", min_value=1.0, max_value=1.05, step=0.01, format="%.2f")
        al = st.number_input("Albumin (Al)", min_value=0.0, max_value=5.0, step=0.1)
        bu = st.number_input("Blood Urea (Bu)", min_value=1.5, max_value=391.0, step=1.0)
        sc = st.number_input("Serum Creatinine (Sc)", min_value=0.4, max_value=76.0, step=0.1)

    with col2:
        sod = st.number_input("Sodium (Sod)", min_value=4.5, max_value=163.0, step=1.0)
        pot = st.number_input("Potassium (Pot)", min_value=2.5, max_value=47.0, step=0.1)
        hemo = st.number_input("Hemoglobin (Hemo)", min_value=3.1, max_value=17.8, step=0.1)
        htn = st.selectbox("Hypertension (Htn)", options=["No", "Yes"])

    # Encode categorical input
    htn_val = 1 if htn == "Yes" else 0

    # ==============================
    # Prediction Button
    # ==============================
    if st.button("🔍 Diagnosis"):
        features = np.array([[sg, al, bu, sc, sod, pot, hemo, htn_val]])
        try:
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)[0][1] * 100  # Probability of CKD

            st.write("---")
            st.write(f"**Confidence Level:** {prediction_proba:.2f}%")

            if prediction[0] == 1:
                st.error("⚠️ The patient is **likely to have Chronic Kidney Disease (CKD)**.\n\nPlease consult a doctor immediately.")
            else:
                st.success("✅ The patient is **NOT likely to have Chronic Kidney Disease (CKD)**.")

        except Exception as e:
            st.error(f"❌ Diagnosis failed: {e}")

# ==============================
# Main Entry Point
# ==============================
if __name__ == "__main__":
    run_app()