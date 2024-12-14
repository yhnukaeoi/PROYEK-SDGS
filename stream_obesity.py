import streamlit as st
import pandas as pd
import pickle
import textwrap
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import base64

# Fungsi untuk menambahkan gambar sebagai latar belakang
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Tambahkan gambar latar belakang
add_bg_from_local("bg_obesity.jpg")

# CSS untuk mempercantik tampilan
st.markdown("""
    <style>
        /* Mengubah tampilan kolom input */
        .stTextInput, .stNumberInput, .stSelectbox, .stSlider {
            background-color: white !important;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* Efek bayangan */
            margin-bottom: 15px;
            width: 100%;
        }
        .stTextInput > div > input,
        .stNumberInput > div > input,
        .stSelectbox > div > select,
        .stSlider > div > div {
            font-size: 14px;
        }
        /* Mengubah warna label untuk input */
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stSlider label {
            color: #333;
        }
        .main {
            background-color: #f7f9fc;
            padding: 20px;
            border-radius: 8px;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .info-card {
            background-color: #ffffff;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model
def load_model_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Fungsi untuk prediksi
def predict(data, model, label_encoders, scaler, target_encoder):
    categorical_cols = ['Gender', 'CALC', 'FAVC', 'SMOKE','SCC', 'family_history_with_overweight', 'CAEC', 'MTRANS']
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

    # Encode kategori
    for col in categorical_cols:
        try:
            data[col] = label_encoders[col].transform(data[col])
        except ValueError as e:
            if 'y contains previously unseen labels' in str(e):
             st.warning(f"Terdapat nilai yang tidak dikenali pada kolom {col}. Nilai akan diisi dengan default.")
            data[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])

    # Standardisasi numerik
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    # Prediksi
    prediction = model.predict(data)
    return target_encoder.inverse_transform(prediction)

# Fungsi untuk memberikan rekomendasi berdasarkan tingkat obesitas
def get_recommendations(prediction):
    recommendations = {
        "Insufficient_Weight": {
            "lifestyle": textwrap.fill(
                "Tingkatkan asupan kalori Anda dan konsultasikan dengan ahli gizi untuk memastikan Anda mendapatkan nutrisi yang cukup. Fokus pada istirahat yang cukup untuk mendukung peningkatan berat badan secara sehat.",
                width=75
            ),
            "diet": textwrap.fill(
                "Konsumsi makanan kaya nutrisi seperti kacang-kacangan, biji-bijian, susu penuh lemak, daging tanpa lemak, buah-buahan, dan sayur. Hindari makanan cepat saji meskipun berkalori tinggi karena rendah nutrisi.",
                width=75
            )
        },
        "Normal_Weight": {
            "lifestyle": textwrap.fill(
                "Pertahankan pola makan seimbang dan rutin berolahraga minimal 30 menit sehari, seperti jogging atau yoga, untuk menjaga berat badan tetap stabil.",
                width=75
            ),
            "diet": textwrap.fill(
                "Konsumsilah biji-bijian utuh, protein tanpa lemak, sayuran, buah-buahan, dan lemak sehat seperti alpukat atau minyak zaitun. Hindari makanan olahan dan tinggi gula.",
                width=75
            )
        },
        "Overweight_Level_I": {
            "lifestyle": textwrap.fill(
                "Tingkatkan aktivitas fisik seperti berjalan cepat selama 30-45 menit setiap hari. Kurangi waktu duduk berlebihan dengan lebih banyak bergerak di sela-sela aktivitas harian.",
                width=75
            ),
            "diet": textwrap.fill(
                "Kurangi konsumsi makanan manis dan tinggi lemak jenuh. Tambahkan lebih banyak sayuran, protein tanpa lemak (seperti ikan atau ayam tanpa kulit), dan lemak sehat (seperti kacang-kacangan atau alpukat).",
                width=75
            )
        },
        "Overweight_Level_II": {
            "lifestyle": textwrap.fill( 
                "Bergabung dengan program kebugaran yang terstruktur atau melakukan aktivitas fisik seperti bersepeda atau berenang 4-5 kali seminggu. Batasi kebiasaan duduk terlalu lama, dan biasakan bergerak setiap jam.",
                width=75
            ),
            
            "diet": textwrap.fill( 
            "Fokus pada pengendalian porsi makan. Pilih makanan tinggi serat seperti oatmeal, sayuran hijau, buah-buahan rendah gula, dan produk susu rendah lemak. Hindari makanan yang digoreng.",
            width=75
            )
        },
        "Obesity_Type_I": {
            "lifestyle": textwrap.fill(
                "Tingkatkan aktivitas aerobik, seperti jogging atau latihan interval, selama 45 menit 4-5 kali seminggu. Konsultasikan dengan pelatih kebugaran untuk membuat rutinitas latihan yang dipersonalisasi.",
                width=75
            ),
            "diet": textwrap.fill(
                "Adopsi pola makan dengan defisit kalori. Fokus pada makanan kaya protein nabati seperti tahu, tempe, dan kacang-kacangan. Batasi konsumsi lemak jenuh dan makanan tinggi karbohidrat olahan.",
                width=75
            )
        },
        "Obesity_Type_II": {
            "lifestyle": textwrap.fill(
            "Dapatkan saran dari ahli gizi atau ahli diet untuk membuat rencana yang dipersonalisasi. Fokus pada latihan kekuatan untuk meningkatkan metabolisme dan aktivitas aerobik intensitas sedang.",
            width=75
            ),
            "diet": textwrap.fill(
            "Ikuti pola makan rendah karbohidrat, tinggi protein, dan rendah lemak jenuh. Hindari minuman manis, makanan tinggi gula, dan batasi asupan kalori dengan cara mengatur porsi makan.",
            width=75
            )
        },
        "Obesity_Type_III": {
            "lifestyle": textwrap.fill(
                "Konsultasikan dengan dokter dan ahli kebugaran untuk rencana penurunan berat badan. Fokus pada aktivitas ringan seperti berjalan kaki atau berenang untuk mengurangi tekanan pada sendi.",
                width=75
            ),
            "diet": textwrap.fill(
                "Ikuti pola makan rendah kalori yang diawasi ahli gizi. Konsumsi makanan tinggi serat dan protein, seperti sayuran, kacang-kacangan, dan ikan. Hindari makanan olahan dan gorengan.",
                width=75
            )
        }

    }

    return recommendations.get(prediction, {
        "lifestyle": "Tidak ada rekomendasi khusus.",
        "diet": "Tidak ada rekomendasi khusus."
    })

    # Fungsi untuk memberikan keterangan sesuai input
def get_water_intake_description(value):
    if value == 1:
        return "(0-1 liter/hari)"
    elif value == 2:
        return "(1-2 liter/hari)"
    elif value == 3:
        return "(>2 liter/hari)"
    else:
        return "Data konsumsi air tidak valid"
    
def get_vegetable_consumtion(value):
    if value == 1:
        return "(Jarang)"
    elif value == 2:
        return "(Sedang)"
    elif value == 3:
        return "(Sering)"
    else:
        return "Data frekuensi konsumsi sayur tidak valid"
    
def get_daily_meal(value):
    if value == 1:
        return "(1 kali/hari)"
    elif value == 2:
        return "(2 kali/hari)"
    elif value == 3:
        return "(3 kali/hari)"
    elif value == 4:
        return "(>3 kali/hari)"
    else:
        return "Data frekuensi konsumsi sayur tidak valid"

def get_physical_activity(value):
    if value == 0:
        return "(Tidak Pernah)"
    elif value == 1:
        return "(Kadang-kadang)"
    elif value == 2:
        return "(Sering)"
    elif value == 3:
        return "(Selalu)"
    else:
        return "Data Aktivitas Fisik tidak valid"

def get_technology_usage(value):
    if value == 0:
        return "(0-2 jam/hari)"
    elif value == 1:
        return "(2-4 jam/hari)"
    elif value == 2:
        return "(>4 jam/hari)"
    else:
        return "Data penggunaan teknologi tidak valid"
    

# Fungsi untuk menyisipkan data ke template gambar
def insert_data_to_image(template_path, output_path, user_data):
    img = Image.open(template_path)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", size=50)
    except:
        font = ImageFont.load_default()

    coordinates = {
        "Nama": (800, 450),
        "Tanggal": (800, 540),
        "Usia": (1250, 1100),
        "Gender": (1250, 1200),
        "Tinggi": (1250, 1310),
        "Berat": (1250, 1420),
        "Alkohol": (1250, 1520),  # Start at 1700 and decrement
        "Kalori": (1250, 1630),
        "Sayur": (1250, 1690),
        "Makan": (1250, 1800),
        "SCC": (1250, 1950),
        "Merokok": (1250, 2060),
        "Minum": (1250, 2170),
        "Keluarga_Obesitas": (1250, 2280),  # Sesuaikan posisi
        "Fisik": (1250, 2340),
        "Teknologi": (1250, 2450),
        "Ngemil": (1250, 2600),
        "Transportasi": (1250, 2710),
        "Prediksi": (300, 2880),
        "Rekomendasi": (300, 3030),
    }


    # Mengambil deskripsi sesuai input
    water_description = get_water_intake_description(user_data["CH2O"])
    vegetable_description = get_vegetable_consumtion(user_data["FCVC"])
    meal_description = get_daily_meal(user_data["NCP"])
    pysical_description = get_physical_activity(user_data["FAF"])
    technology_description = get_technology_usage(user_data["TUE"])


    # Gabungkan teks
    text_to_draw = f"{user_data['CH2O']} = {water_description}" 
    text_to_draw1 = f"\n{user_data['FCVC']} = {vegetable_description}"
    text_to_draw2 = f"\n{user_data['NCP']} = {meal_description}"
    text_to_draw3 = f"\n{user_data['FAF']} = {pysical_description}"
    text_to_draw4 = f"\n{user_data['TUE']} = {technology_description}"



    draw.text(coordinates["Nama"], f"{user_data['name']}", font=font, fill="black")
    draw.text(coordinates["Tanggal"], f"{user_data['date']}", font=font, fill="black")
    draw.text(coordinates["Usia"], f"{user_data['Age']} tahun", font=font, fill="black")
    draw.text(coordinates["Gender"], f"{user_data['Gender']}", font=font, fill="black")
    draw.text(coordinates["Tinggi"], f"{user_data['Height']} cm", font=font, fill="black")
    draw.text(coordinates["Berat"], f"{user_data['Weight']} kg", font=font, fill="black")
    draw.text(coordinates["Alkohol"], f"{user_data['CALC']}", font=font, fill="black")

    draw.text(coordinates["Kalori"], f"{user_data['FAVC']}", font=font, fill="black")

    draw.text(coordinates["Sayur"], text_to_draw1, font=font, fill="black")
    draw.text(coordinates["Makan"], text_to_draw2, font=font, fill="black")
    draw.text(coordinates["SCC"], f"{user_data['SCC']}", font=font, fill="black")
    draw.text(coordinates["Merokok"], f"{user_data['SMOKE']}", font=font, fill="black")

    draw.text(coordinates["Minum"], text_to_draw, font=font, fill="black")
    draw.text(coordinates["Keluarga_Obesitas"], f"{user_data['family_history_with_overweight']}", font=font, fill="black")

    draw.text(coordinates["Fisik"], text_to_draw3, font=font, fill="black")
    draw.text(coordinates["Teknologi"], text_to_draw4, font=font, fill="black")
    draw.text(coordinates["Ngemil"], f"{user_data['CAEC']}", font=font, fill="black")
    draw.text(coordinates["Transportasi"], f"{user_data['MTRANS']}", font=font, fill="black")
    draw.text(coordinates["Prediksi"], f"{user_data['prediction']}", font=font, fill="black")
    draw.text((coordinates["Rekomendasi"][0], coordinates["Rekomendasi"][1] + 30),
            f"Gaya Hidup: {recommendations['lifestyle']}", font=font, fill="black")
    draw.text((coordinates["Rekomendasi"][0], coordinates["Rekomendasi"][1] + 200),
            f"Pola Makan: {recommendations['diet']}", font=font, fill="black")

    img.save(output_path)
    return output_path

# Fungsi untuk mengonversi gambar ke PDF
def convert_image_to_pdf(image_path):
    img = Image.open(image_path)
    pdf_bytes = BytesIO()
    img.save(pdf_bytes, format='PDF')
    pdf_bytes.seek(0)
    return pdf_bytes

# Header
st.markdown("<h1>Prediksi Tingkat Obesitas</h1>", unsafe_allow_html=True)

# Input pengguna
name = st.text_input("Nama Lengkap")
exam_date = st.date_input("Tanggal Pemeriksaan")
Age = st.number_input("Usia", min_value=1, max_value=100, value=30)
Gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
Height = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=170)
Weight = st.number_input("Berat Badan (kg)", min_value=20, max_value=200, value=60)
CALC = st.selectbox("Kebiasaan Minum Alkohol", ["no", "Sometimes", "Frequently", "Always"])
FAVC = st.selectbox("Konsumsi Makanan Tinggi Kalori", ["no", "yes"])
FCVC = st.slider("Frekuensi konsumsi sayur == 1 (Jarang), 2 (Sedang), 3 (Sering) ", min_value=1, max_value=3, value=2)
NCP = st.slider("Jumlah makanan utama dalam sehari == 1 (1 kali), 2 (2 kali), 3 (3 kali), 4 (>3 kali)", min_value=1, max_value=4, value=3)
SCC = st.selectbox("Kebiasaan memantau asupan kalori", ["no", "yes"])  # Sesuaikan dengan encoder
SMOKE = st.selectbox("Kebiasaan merokok", ["no", "yes"])
CH2O = st.slider("Konsumsi air putih harian (dalam liter) == 1 (0-1 liter), 2 (1-2 liter), 3 (>2 liter).", min_value=1, max_value=3, value=2)
family_history_with_overweight = st.selectbox("Riwayat keluarga obesitas", ["no", "yes"])
FAF = st.slider("Frekuensi aktivitas fisik == 0 (Tidak pernah), 1 (Kadang-kadang), 2 (Sering), 3(Selalu)", min_value=0, max_value=3, value=1)
TUE = st.slider("Waktu menggunakan teknologi == 0 (0-2 jam), 1 (2-4 jam), 2 (>4 jam)", min_value=0, max_value=2, value=1)
CAEC = st.selectbox("Kebiasaan Ngemil", ["no", "Sometimes", "Frequently", "Always"])
MTRANS = st.selectbox("Transportasi utama", ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])

# Simpan input dalam DataFrame
user_input = pd.DataFrame({
    'Age': [Age],
    'Gender': [Gender],
    'Height': [Height],
    'Weight': [Weight],
    'CALC': [CALC],
    'FAVC': [FAVC],
    'FCVC': [FCVC],
    'NCP': [NCP],
    'SCC': [SCC],  # Tambahkan SCC
    'SMOKE': [SMOKE],
    'CH2O': [CH2O],
    'family_history_with_overweight': [family_history_with_overweight],
    'FAF': [FAF],
    'TUE': [TUE],
    'CAEC': [CAEC],
    'MTRANS': [MTRANS]
})

# Load model
model_data = load_model_pickle('obesity_model.pkl')
model = model_data['model']
label_encoders = model_data['label_encoders']
scaler = model_data['scaler']
target_encoder = model_data['target_encoder']

# Prediksi
if st.button('Prediksi Tingkat Obesitas'):
    prediction = predict(user_input, model, label_encoders, scaler, target_encoder)
    st.markdown("<h2>Hasil Prediksi</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='info-card'><strong>Tingkat Obesitas:</strong> {prediction[0]}</div>", unsafe_allow_html=True)
    
    # Rekomendasi
    recommendations = get_recommendations(prediction[0])

    st.markdown(f"<div class='info-card'><strong>Gaya Hidup:</strong> {recommendations['lifestyle']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='info-card'><strong>Pola Makan:</strong> {recommendations['diet']}</div>", unsafe_allow_html=True)

    # Sisipkan data ke template
    template_path = "Hasil.jpg"  # Ganti dengan path template Anda
    output_image_path = "Hasil_Updated.jpg"
    user_data = {
        "name": name,
        "date": exam_date,
        "Age": Age,
        "Gender": Gender,
        "Height": Height,
        "Weight": Weight,
        "CALC": CALC,
        "FAVC": FAVC,
        "FCVC": FCVC,
        "NCP": NCP,
        "SCC": SCC,
        "SMOKE": SMOKE,
        "CH2O": CH2O,
        "family_history_with_overweight": family_history_with_overweight,
        "FAF": FAF,
        "TUE": TUE,
        "CAEC": CAEC,
        "MTRANS": MTRANS,
        "prediction": prediction[0],
       "recommendation": f"Gaya Hidup: {recommendations['lifestyle']}<br>Pola Makan: {recommendations['diet']}"

    }

    image_path = insert_data_to_image(template_path, output_image_path, user_data)
    st.image(image_path, caption="Hasil Template")

    

    # Konversi ke PDF dan unduh
    pdf_bytes = convert_image_to_pdf(image_path)
    st.download_button(
        label="Unduh PDF",
        data=pdf_bytes,
        file_name="Hasil_Prediksi_Obesitas.pdf",
        mime="application/pdf"
    )

   

