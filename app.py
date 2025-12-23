import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

binary_classes = ["Jinak", "Ganas"]

disease_classes = [
    "Actinic keratosis",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Vascular lesion"
]

disease_info = {
    "Actinic keratosis": "Lesi kulit akibat paparan sinar UV, berpotensi berkembang menjadi kanker kulit.",
    "Basal cell carcinoma": "Kanker kulit paling umum, tumbuh lambat dan jarang menyebar.",
    "Benign keratosis": "Lesi kulit jinak, tidak berbahaya.",
    "Dermatofibroma": "Benjolan kulit jinak yang biasanya muncul akibat luka kecil atau gigitan serangga.",
    "Melanocytic nevus": "Tahi lalat biasa, biasanya jinak namun perlu pemantauan.",
    "Melanoma": "Kanker kulit agresif yang bisa menyebar, butuh penanganan cepat.",
    "Squamous cell carcinoma": "Kanker kulit yang bisa menyebar, sering muncul di area yang terpapar sinar matahari.",
    "Vascular lesion": "Lesi pembuluh darah kulit, biasanya jinak dan tidak berbahaya."
}

benign_diseases = [
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Vascular lesion"
]

malignant_diseases = [
    "Actinic keratosis",
    "Basal cell carcinoma",
    "Melanoma",
    "Squamous cell carcinoma"
]

@st.cache_resource
def load_models():
    # Binary
    model_binary = models.resnet50(pretrained=False)
    model_binary.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 2)
    )
    model_binary.load_state_dict(
        torch.load("Model_Binary.pth", map_location=DEVICE)
    )
    model_binary.to(DEVICE).eval()

    # Multiclass
    model_multi = models.resnet50(pretrained=False)
    model_multi.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, len(disease_classes))
    )
    model_multi.load_state_dict(
        torch.load("Model_Multi.pth", map_location=DEVICE)
    )
    model_multi.to(DEVICE).eval()

    return model_binary, model_multi


model_binary, model_multi = load_models()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(image):
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Binary
        out_bin = model_binary(img)
        prob_bin = F.softmax(out_bin, dim=1)
        idx_bin = prob_bin.argmax(1).item()

        # Multiclass
        out_multi = model_multi(img)
        prob_multi = F.softmax(out_multi, dim=1)
        idx_multi = prob_multi.argmax(1).item()

    return {
        "binary_label": binary_classes[idx_bin],
        "binary_conf": prob_bin[0][idx_bin].item() * 100,
        "disease_label": disease_classes[idx_multi],
        "disease_conf": prob_multi[0][idx_multi].item() * 100
    }

# =========================
# UI
# =========================
st.title("DETEKSI KANKER KULIT")


st.markdown("### 📸 Ambil Foto Kamera")
camera_file = st.camera_input("")


st.markdown("### 📂 Unggah Gambar")
uploaded_file = st.file_uploader(
    "Unggah citra kulit (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

image = None
if camera_file is not None:
    image = Image.open(camera_file).convert("RGB")
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

# =========================
# AUTO DETECTION + FUSION
# =========================
if image is not None:
    st.image(image, width=400)

    with st.spinner("Menganalisis citra..."):
        result = predict(image)

    # ===== DECISION FUSION =====
    if result["disease_label"] in benign_diseases:
        final_status = "Jinak"
        final_color = "success"
    else:
        final_status = "Ganas"
        final_color = "error"

    # Cek konsistensi binary vs multiclass
    binary_ok = (result["binary_label"] == final_status)

    st.markdown("## 🧾 Hasil Deteksi Akhir")

    # ===== FINAL STATUS =====
    if final_color == "error":
        st.error("🔴 **Status Kanker: GANAS**")
    else:
        st.success("🟢 **Status Kanker: JINAK**")

    # ===== DISEASE =====
    if result["disease_label"] in malignant_diseases:
        st.error(
            f"🔴 **Jenis Penyakit:** {result['disease_label']}  \n"
            f"Confidence: {result['disease_conf']:.2f}%"
        )
    else:
        st.success(
            f"🟢 **Jenis Penyakit:** {result['disease_label']}  \n"
            f"Confidence: {result['disease_conf']:.2f}%"
        )

    # ===== CEK KONSISTENSI =====
    if not binary_ok:
        st.warning(
            "⚠️ Prediksi penyakit tidak konsisten. "
            "Mohon konsultasikan ke dokter untuk pemeriksaan lebih lanjut."
        )
