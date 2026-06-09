import streamlit as st
import numpy as np
import pickle, os
import tensorflow as tf

KLASOR    = r'C:\Users\merha\OneDrive\Masaüstü\24.03.2026\Bilgisayar\Yuksek_Lisans\1\veriprroeses\Proje\4\files'
MODEL_DIR = os.path.join(KLASOR, 'modeller')

st.set_page_config(page_title="Güneş Işınımı Tahmini", page_icon="☀️", layout="centered")

@st.cache_resource
def model_yukle(sehir):
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f'{sehir}_model.keras'))
    with open(os.path.join(MODEL_DIR, f'{sehir}_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f'{sehir}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return model, scaler, meta

st.title("☀️ Güneş Işınımı Tahmin Sistemi")
st.markdown("Seçtiğiniz ilin LSTM modeli ile bir sonraki saatin güneş ışınımını tahmin eder.")
st.divider()

sehir = st.selectbox("📍 Şehir seçin", ['Konya', 'Van', 'Mugla'])

try:
    model, scaler, meta = model_yukle(sehir)
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

ort = meta['ortalamalar']
ara = meta['araliklar']

c1, c2 = st.columns(2)
c1.metric("Model RMSE", f"{meta['rmse']:.1f} Wh/m²")
c2.metric("Model R²", f"{meta['r2']:.3f}")

st.divider()
st.subheader("Mevcut koşulları girin")
st.caption("Boş bıraktığınız alanlar için ilin ortalama değeri kullanılır.")

col1, col2 = st.columns(2)

with col1:
    isinim_in = st.number_input(
        f"Güneş ışınımı (Wh/m²) — ort: {ort['isinim']:.0f}",
        min_value=0.0, max_value=float(ara['isinim'][1]*1.2),
        value=None, placeholder=f"{ort['isinim']:.0f} (ortalama)"
    )
    nem_in = st.number_input(
        f"Nem (g/kg) — ort: {ort['nem']:.1f}",
        value=None, placeholder=f"{ort['nem']:.1f} (ortalama)"
    )

with col2:
    sicaklik_in = st.number_input(
        f"Sıcaklık (°C) — ort: {ort['sicaklik']:.1f}",
        value=None, placeholder=f"{ort['sicaklik']:.1f} (ortalama)"
    )
    basinc_in = st.number_input(
        f"Basınç (kPa) — ort: {ort['basinc']:.1f}",
        value=None, placeholder=f"{ort['basinc']:.1f} (ortalama)"
    )

if st.button("🔮 Tahmin Et", type="primary", use_container_width=True):

    isinim_v   = isinim_in   if isinim_in   is not None else ort['isinim']
    sicaklik_v = sicaklik_in if sicaklik_in is not None else ort['sicaklik']
    nem_v      = nem_in      if nem_in      is not None else ort['nem']
    basinc_v   = basinc_in   if basinc_in   is not None else ort['basinc']

    kullanilan = []
    if isinim_in is None:   kullanilan.append("ışınım")
    if sicaklik_in is None: kullanilan.append("sıcaklık")
    if nem_in is None:      kullanilan.append("nem")
    if basinc_in is None:   kullanilan.append("basınç")

    girdi = np.array([[isinim_v, sicaklik_v, nem_v, basinc_v]])
    girdi_norm = scaler.transform(girdi).reshape(1, 1, 4)
    pred_norm = model.predict(girdi_norm, verbose=0).flatten()
    dummy = np.zeros((1, 4)); dummy[:,0] = pred_norm
    tahmin = float(np.clip(scaler.inverse_transform(dummy)[0,0], 0, None))

    st.divider()
    st.subheader("📊 Tahmin Sonucu")
    st.metric(f"{sehir} — Bir sonraki saatin güneş ışınımı", f"{tahmin:.1f} Wh/m²")

    if kullanilan:
        st.info(f"Şu değerler için ilin ortalaması kullanıldı: {', '.join(kullanilan)}")

    maks = ara['isinim'][1]
    oran = tahmin / maks if maks > 0 else 0

    st.subheader("💡 Değerlendirme")
    if tahmin < 1:
        st.write("🌙 Gece veya güneş batmış — panel üretimi beklenmiyor.")
    elif oran < 0.2:
        st.write("☁️ Düşük ışınım — bulutlu veya erken/geç saat. Panel verimi sınırlı.")
    elif oran < 0.5:
        st.write("⛅ Orta düzey ışınım — kısmen bulutlu. Üretim makul seviyede.")
    else:
        st.write("☀️ Yüksek ışınım — açık ve güneşli. Panel üretimi için elverişli.")

    if sicaklik_v < 0:
        st.write("❄️ Donma riski — panel yüzeyinde buzlanma olabilir.")
    if nem_v > ort['nem'] * 1.5:
        st.write("💧 Yüksek nem — yağış veya sis ihtimaline dikkat.")

st.divider()
st.caption("Bu uygulama eğitilmiş LSTM modellerini kullanır. Tahminler istatistiksel olup kesin değildir.")