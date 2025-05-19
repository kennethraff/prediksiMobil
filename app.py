import streamlit as st
import pickle
import pandas as pd
import datetime

# — 1) Load artifacts —
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("stacked_model.pkl", "rb") as f:
    model = pickle.load(f)

# — 2) Opsi dropdown dari encoder.classes_ —
brands        = list(encoders['Brand'].classes_)
models        = list(encoders['Model'].classes_)
locations     = list(encoders['Location'].classes_)
transmissions = list(encoders['Transmission'].classes_)
fuels         = list(encoders['Fuel'].classes_)
jeniss        = list(encoders['Jenis'].classes_)

# — 3) Konfigurasi numeric inputs —
current_year = datetime.datetime.now().year
min_mileage, max_mileage = 0, 300_000

# — 4) Layout —
st.set_page_config(layout="wide")
st.title("🔮 Prediksi Harga Jual Mobil Bekasmu")
st.markdown("Masukkan spesifikasi mobilmu di bawah untuk melihat estimasi harga jual bekasnya.")
result_box = st.empty()

# Grid layout: 4 baris x 2 kolom
col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", brands)
    model_user = st.selectbox("Model", models)
    mileage = st.slider("Mileage (km)", min_mileage, max_mileage, step=5000)
    year = st.selectbox("Tahun", list(range(1990, current_year + 1)))

with col2:
    fuel = st.selectbox("Fuel", fuels)
    location = st.selectbox("Location", locations)
    transmission = st.selectbox("Transmission", transmissions)
    jenis = st.selectbox("Jenis", jeniss)

predict_btn = st.button("🔍 Prediksikan")

# — 5) Blok prediksi —
if predict_btn:
    umur = current_year - year
    FEATURE_ORDER = [
        'Brand','Model','Location',
        'Year','Mileage','Transmission',
        'Fuel','Umur','Jenis'
    ]

    x_df = pd.DataFrame([{
        'Brand':        brand,
        'Model':        model_user,
        'Location':     location,
        'Transmission': transmission,
        'Fuel':         fuel,
        'Jenis':        jenis,
        'Year':         year,
        'Mileage':      mileage,
        'Umur':         umur
    }])

    for col, le in encoders.items():
        x_df[col] = le.transform(x_df[col])

    numerical_cols = ['Year', 'Mileage', 'Umur']
    x_df[numerical_cols] = scaler.transform(x_df[numerical_cols])
    x_df = x_df[FEATURE_ORDER]

    price_pred = model.predict(x_df)[0] * 1000

    result_box.markdown(
        f"**Harga wajar untuk mobil di tahun 2025 adalah**:  \n"
        f"<div style='font-size:1.8rem; color:#E76F51;'>Rp {price_pred:,.0f} ,-</div>",
        unsafe_allow_html=True
    )
