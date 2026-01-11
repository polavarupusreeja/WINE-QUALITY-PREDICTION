import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# ------------------------------
# Background Image + CSS + Animations
# ------------------------------
bg_image_url = "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?q=80&w=2070&auto=format&fit=crop"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{bg_image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Dark overlay */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.4);
        z-index: -1;
    }}

    h1, h2, h3, p {{
        color: white !important;
    }}

    /* Input Box */
    [data-testid="stVerticalBlock"] > div:nth-child(2) {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 30px;
        border-radius: 15px;
    }}

    /* Sliders */
    label, .stSlider > label {{
        color: #4a0e0e !important;
        font-weight: 700;
    }}

    /* Button */
    .stButton>button {{
        background-color: #7b1e3a;
        color: white;
        border-radius: 8px;
        width: 100%;
        height: 3em;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #a0264d;
    }}

    /* 🍷 Wine Pour Animation (GIF) */
    .wine-pour {{
        width: 180px;
        height: auto;
        animation: fadeIn 1.5s ease-in-out;
        margin: auto;
    }}

    /* Floating Grapes 🍇 */
    .grape {{
        position: absolute;
        font-size: 40px;
        animation: floating 4s infinite ease-in-out;
        opacity: 0.8;
    }}

    @keyframes floating {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-20px); }}
        100% {{ transform: translateY(0px); }}
    }}

    /* Fade In Animation */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Load saved model & scaler
# ------------------------------
try:
    scaler = pickle.load(open("scaler_model.sav", "rb"))
    model = pickle.load(open("finalized_RFmodel.sav", "rb"))
except FileNotFoundError:
    st.error("❌ Model files not found.")

# ------------------------------
# Title
# ------------------------------
st.title("🍷 Wine Quality Prediction")
st.write("Predict the *quality of red wine* using machine learning")

st.divider()

# ------------------------------
# Inputs
# ------------------------------
st.subheader("Enter Wine Properties")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.5)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.slider("Residual Sugar (log)", 0.0, 3.0, 0.6)
    chlorides = st.slider("Chlorides (log)", 0.0, 2.0, 0.9)
    free_sulfur = st.slider("Free Sulfur Dioxide (log)", 0.0, 2.0, 0.6)

with col2:
    total_sulfur = st.slider("Total Sulfur Dioxide", 6.0, 300.0, 98.0)
    density = st.slider("Density", 0.990, 1.005, 0.996)
    ph = st.slider("pH", 2.5, 4.5, 3.3)
    sulphates = st.slider("Sulphates (log)", 0.0, 3.0, 1.8)
    alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.5)

# ------------------------------
# Create DataFrame
# ------------------------------
input_data = pd.DataFrame([[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur, total_sulfur, density,
    ph, sulphates, alcohol
]], columns=[
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
])

# ------------------------------
# Prediction + ANIMATIONS
# ------------------------------
if st.button("Predict Wine Quality 🍷"):

    with st.spinner("Analyzing wine properties... 🍇"):
        time.sleep(1)
        scaled_input = scaler.transform(input_data)
        predicted_quality = int(round(model.predict(scaled_input)[0]))

    # 🔊 Wine Pouring Sound (autoplay)
    st.markdown(
        """
        <audio autoplay>
            <source src="https://www.soundjay.com/mechanical/sounds/water-pour-1.mp3" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True
    )

    # 🍷 Wine Pour Animation
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://i.pinimg.com/originals/81/14/44/811444b777c2c36d2c36fdaa1ba9ca37.gif" class="wine-pour">
        </div>
        """,
        unsafe_allow_html=True
    )

    # ⭐ Grapes animation
    st.markdown(
        """
        <div style='position:relative; height:60px;'>
            <span class='grape' style='left:20%; animation-delay:0s;'>🍇</span>
            <span class='grape' style='left:50%; animation-delay:1s;'>🍇</span>
            <span class='grape' style='left:80%; animation-delay:2s;'>🍇</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ⭐ QUALITY RESULT CARD
    if predicted_quality >= 7:
        color = "#2e7d32"
        quality_text = "Excellent Wine! 🍾"
        st.balloons()
    elif predicted_quality >= 5:
        color = "#ff9800"
        quality_text = "Average Wine 🍷"
        st.toast("Not bad!", icon="👍")
    else:
        color = "#d32f2f"
        quality_text = "Low Quality ❌"
        st.toast("Better for cooking than drinking 😅", icon="🍳")

    st.markdown(
        f"""
        <div style='
            background-color: rgba(255,255,255,0.92);
            padding: 25px;
            margin-top: 20px;
            border-radius: 15px;
            border-left: 10px solid {color};
            color: {color};
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
            animation: fadeIn 1.2s;
        '>
            Predicted Score: {predicted_quality} / 10<br>
            <small style='font-size: 20px;'>{quality_text}</small>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
st.caption("Made with ❤️ using Streamlit & Machine Learning")
