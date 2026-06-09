
# Imports
import streamlit as st
import pandas as pd
import qrcode
from io import BytesIO
from PIL import Image
import joblib

# Load trained pipeline model
pipeline = joblib.load("purchase_pipeline.pkl")

# App Title & Description
st.title("Digital Advertising CPA Prediction & Scenario Testing")

# Interactive Scenario Testing User Interface
    # -------------------------------
st.write("### Scenario Testing")

w_cpa = st.number_input("Cost of Acquisition ($)", min_value=1.0, value=50.0)
w_ctr = st.slider("Click Through Rate (%)", 0.0, 1.0, 0.05)

age_group = st.selectbox("Age Group", ["16-17","18-24","25-34","35-44","45-54","55-65"])
country = st.selectbox("Country", ["australia","brazil","canada","france","germany","india","japan","mexico","united kingdom","united states"])
ad_type = st.selectbox("Ad Type", ["carousel","image","stories","video"])
day_of_week = st.selectbox("Day of Week", ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"])

# Create Scenario Dataframe for features
user_input = pd.DataFrame({
        "w_cpa": [w_cpa],
        "w_ctr": [w_ctr],
        "age_group": [age_group],
        "country": [country],
        "ad_type": [ad_type],
        "day_of_week": [day_of_week]
    })


# Predict Purchase Probability
purchase_prob = pipeline.predict_proba(user_input)[:, 1][0]

# Expected CPA = Cost / Expected Purchases
expected_cpa = w_cpa / purchase_prob if purchase_prob > 0 else None

# Display Results
st.write("### Scenario Results")
st.metric("Predicted Purchase Probability", f"{purchase_prob:.2%}")
if expected_cpa:
    st.metric("Expected Cost per Acquisition", f"${expected_cpa:,.2f}")
else:
    st.warning("Purchase probability is zero, CPA undefined.")

# QR Code for sharing
st.write("### Share This App")
st.write("Scan the QR code below to open this app on your phone:")

# Assume local Streamlit server URL
app_url = "http://localhost:8501"

qr = qrcode.QRCode(box_size=6, border=2)
qr.add_data(app_url)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
buf = BytesIO()
img.save(buf)
buf.seek(0)

st.image(Image.open(buf), caption="Scan to open app")