import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Custom CSS for pastel theme
st.markdown("""
<style>
/* Soft pastel card style */
.pastel-card {
    background-color: #D7EFFC;  /* light baby blue */
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 3px 10px rgba(174,225,249,0.5);
    margin-bottom: 20px;
}

/* Gradient banner style */
.pastel-banner {
    background: linear-gradient(135deg, #AEE1F9, #D7EFFC);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 20px;
}

/* Buttons in pastel gradient */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #AEE1F9, #D7EFFC);
    color: #4A4A4A;
    border-radius: 12px;
    height: 45px;
    font-size: 16px;
    border: none;
    box-shadow: 0 3px 10px rgba(174,225,249,0.5);
}
div.stButton > button:hover {
    opacity: 0.9;
}

/* Radio buttons pastel */
div.stRadio > div {
    background-color: #EFFBFF;
    padding: 10px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# Load saved files
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

df = pd.read_csv("Toddler Autism dataset July 2018 (1).csv")

# st.title("🧠 Autism Spectrum Disorder (ASD) Detection App")
# st.markdown("""
# <div class="pastel-banner">
#     <h2>🧩 Welcome to the ASD Detection Website</h2>
#     <p>Soft, calm, and supportive interface 💙</p>
# </div>
# """, unsafe_allow_html=True)


page = st.sidebar.radio("Navigation",["Home", "Predict ASD", "Support & Help"])


# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.title("🧠 Autism Spectrum Disorder (ASD) Detection App")
    st.markdown("""
    <div class="pastel-banner">
    <h2>🧩 Welcome to the ASD Detection Website</h2>
    <p>Soft, calm, and supportive interface 💙</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="pastel-card">
        <h2>About This App</h2>
        <p>This app is designed to help detect Autism Spectrum Disorder (ASD) in toddlers.</p>
        <p>If you suspect ASD, please consult a healthcare professional for a proper diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("Welcome to the ASD Detection App")

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Predict ASD":
    st.header("ASD Prediction Form")

    st.header("Fill the Screening Form")

    user_input = {}
# added for user to select yes/no for the screening questions 

    # Screening A1–A10
    screening_questions = {
        "Does your child look at you when you call their name?": "A1",
        "Does your child point to indicate that they want something?": "A2",
        "Does your child play pretend games (e.g., pretend to feed a doll)?": "A3",
        "Does your child smile in response to your face or smile?": "A4",
        "Is your child able to engage in simple pretend play (e.g., pretend drinking)?": "A5",
        "Does your child respond to their name when called?": "A6",
        "Does your child make eye contact?": "A7",
        "Does your child imitate your actions?": "A8",
        "Does your child respond when you point at something?": "A9",
        "Does your child show interest in other children or attempt to play with them?": "A10"
    }
    for question, key in screening_questions.items():
        val = st.radio(
            question,
            ["-- Select --","Yes", "No"],
            index=0,
            key=key
        )

        if val == "Yes":
            user_input[key] = 1
        elif val == "No":
            user_input[key] = 0
        else:
            user_input[key] = None

    # Age
    age = st.number_input("Age (months)", min_value=12, max_value=72)
    user_input["Age_Mons"] = age

    # Qchat Score
    if None in [user_input[f"A{i}"] for i in range(1, 11)]:
     st.warning("⚠️ Please answer all screening questions before     prediction.")
     st.stop()
    qchat_score = 0
    for i in range(1, 11):
      qchat_score = qchat_score + user_input[f"A{i}"]
    user_input["Qchat-10-Score"] = qchat_score

    # Encoded categorical fields
    for col in ["Sex", "Ethnicity", "Jaundice","Family_mem_with_ASD"]:
        val = st.selectbox(col, label_encoders[col].classes_)
        user_input[col] = label_encoders[col].transform([val])[0]

    # Predict Button
    if st.button("Predict"):

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])

        # Scale numeric values
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("✅ No ASD Traits Detected")
        else:
            st.error("⚠️ ASD Traits Detected. Please consult a professional.")

#support page
if page == "Support & Help":
    st.write("💗“Autism is not a disability — it is a different ability. With understanding, support, and love, every child can shine in their own beautiful way.”")
    