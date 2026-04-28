import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# CROP INFO (COMPLETE DATA)
# -------------------------------
crop_info = {
    "rice": {"season": "Kharif", "soil": "Clayey soil", "tips": "Requires high rainfall"},
    "maize": {"season": "Kharif/Rabi", "soil": "Loamy soil", "tips": "Needs moderate rainfall"},
    "chickpea": {"season": "Rabi", "soil": "Sandy loam", "tips": "Needs low rainfall"},
    "kidneybeans": {"season": "Rabi", "soil": "Well-drained soil", "tips": "Avoid excess water"},
    "pigeonpeas": {"season": "Kharif", "soil": "Loamy soil", "tips": "Drought tolerant"},
    "mothbeans": {"season": "Kharif", "soil": "Sandy soil", "tips": "Needs less water"},
    "mungbean": {"season": "Kharif", "soil": "Loamy soil", "tips": "Short duration crop"},
    "blackgram": {"season": "Kharif", "soil": "Clay loam", "tips": "Needs warm climate"},
    "lentil": {"season": "Rabi", "soil": "Loamy soil", "tips": "Requires cool climate"},
    "pomegranate": {"season": "Annual", "soil": "Well-drained soil", "tips": "Needs dry climate"},
    "banana": {"season": "Year-round", "soil": "Loamy soil", "tips": "High humidity needed"},
    "mango": {"season": "Summer", "soil": "Loamy soil", "tips": "Needs warm climate"},
    "grapes": {"season": "Winter", "soil": "Sandy soil", "tips": "Requires pruning"},
    "watermelon": {"season": "Summer", "soil": "Sandy loam", "tips": "Needs sunlight"},
    "muskmelon": {"season": "Summer", "soil": "Sandy soil", "tips": "Warm climate"},
    "apple": {"season": "Winter", "soil": "Loamy soil", "tips": "Needs cold climate"},
    "orange": {"season": "Winter", "soil": "Loamy soil", "tips": "Moderate water"},
    "papaya": {"season": "Year-round", "soil": "Well-drained soil", "tips": "Avoid waterlogging"},
    "coconut": {"season": "Annual", "soil": "Sandy soil", "tips": "Needs coastal climate"},
    "cotton": {"season": "Kharif", "soil": "Black soil", "tips": "Warm climate"},
    "jute": {"season": "Kharif", "soil": "Alluvial soil", "tips": "High humidity"},
    "coffee": {"season": "Winter", "soil": "Loamy soil", "tips": "Shade required"}
}

# -------------------------------
# UI TITLE
# -------------------------------
st.title("🌾 Smart Agriculture AI Platform")
st.write("Enter soil and weather conditions to get crop recommendation")

# -------------------------------
# INPUT FIELDS
# -------------------------------
st.subheader("🧪 Soil Nutrients")
col1, col2, col3 = st.columns(3)

with col1:
    N = st.slider("Nitrogen (N)", 0, 140, 50)
with col2:
    P = st.slider("Phosphorus (P)", 0, 140, 50)
with col3:
    K = st.slider("Potassium (K)", 0, 140, 50)

st.subheader("🌦 Weather Conditions")
col4, col5 = st.columns(2)

with col4:
    temperature = st.slider("Temperature (°C)", 0, 50, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 60)

with col5:
    ph = st.slider("pH", 0.0, 14.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    import os

    file_path = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
    return pd.read_csv(file_path)

data = load_data()

# -------------------------------
# TRAIN MODEL
# -------------------------------
@st.cache_resource
def train_model(data):
    X = data.drop('label', axis=1)
    y = data['label']

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

model = train_model(data)

# -------------------------------
# BUTTON ACTION
# -------------------------------
st.write("Click below to get recommendation")

if st.button("🌱 Get Recommendation"):

    if N == 0 and P == 0 and K == 0:
        st.warning("⚠️ Please enter valid soil values!")
    else:
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]

        probs = model.predict_proba(input_data)[0]
        top_indices = probs.argsort()[-3:][::-1]

        # -------------------------------
        # TOP 3 RESULTS
        # -------------------------------
        st.subheader("🌱 Top Crop Recommendations")

        for i in top_indices:
            crop = model.classes_[i]
            confidence = probs[i] * 100
            st.write(f"**{crop}** — {confidence:.2f}% confidence")

        # -------------------------------
        # CHART
        # -------------------------------
        chart_data = pd.DataFrame({
            "Crop": [model.classes_[i] for i in top_indices],
            "Confidence (%)": [probs[i] * 100 for i in top_indices]
        })

        st.subheader("📊 Prediction Confidence")
        st.bar_chart(chart_data.set_index("Crop"))

        # -------------------------------
        # CROP DETAILS
        # -------------------------------
        top_crop = model.classes_[top_indices[0]].lower()

        st.subheader(f"📘 Crop Details: {top_crop.capitalize()}")

        if top_crop in crop_info:
            info = crop_info[top_crop]

            st.write(f"🌱 **Season:** {info['season']}")
            st.write(f"🌍 **Soil:** {info['soil']}")
            st.write(f"💡 **Tips:** {info['tips']}")
        else:
            st.info(f"No detailed info available for **{top_crop}** yet.")