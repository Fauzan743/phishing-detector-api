from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import io
import os
import zipfile
import requests
import joblib
import pandas as pd

from feature_extractor import extract_features


# ============================================================
# KONFIGURASI
# ============================================================

MODEL_PATH = "models/random_forest_url.pkl"

MODEL_DOWNLOAD_URL = (
    "https://github.com/Fauzan743/phishing-detector-api/"
    "releases/download/v1.0.0/random_forest_model_v1.0.0.zip"
)


# ============================================================
# DOWNLOAD MODEL DARI GITHUB RELEASE
# ============================================================

def download_model():
    """
    Mengunduh model Random Forest dari GitHub Release
    dan mengekstraknya ke folder models/.
    """

    os.makedirs("models", exist_ok=True)

    print("Model tidak ditemukan atau bukan model asli.")
    print("Mengunduh model dari GitHub Release...")
    print(f"URL: {MODEL_DOWNLOAD_URL}")

    response = requests.get(
        MODEL_DOWNLOAD_URL,
        timeout=120
    )

    response.raise_for_status()

    print("Download model berhasil.")
    print("Mengekstrak model...")

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:

        model_filename = "random_forest_url.pkl"

        if model_filename not in zip_file.namelist():
            raise FileNotFoundError(
                f"{model_filename} tidak ditemukan di dalam ZIP."
            )

        zip_file.extract(
            model_filename,
            "models"
        )

    print(f"Model berhasil diekstrak ke: {MODEL_PATH}")


# ============================================================
# CEK MODEL
# ============================================================

def model_file_is_valid():
    """
    Mengecek apakah file model benar-benar model Random Forest
    atau hanya file pointer Git LFS.
    """

    if not os.path.exists(MODEL_PATH):
        return False

    try:

        with open(MODEL_PATH, "rb") as f:

            header = f.read(100)

        # ----------------------------------------------------
        # File Git LFS pointer biasanya diawali dengan:
        #
        # version https://git-lfs.github.com/spec/v1
        #
        # Jika ditemukan, berarti file tersebut bukan model asli.
        # ----------------------------------------------------

        if header.startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):

            print("File model terdeteksi sebagai Git LFS pointer.")

            return False

        return True

    except Exception as e:

        print(f"Gagal memeriksa file model: {e}")

        return False


# ============================================================
# LOAD MODEL
# ============================================================

model = None
MODEL_LOADED = False
MODEL_LOAD_ERROR = None


try:

    # --------------------------------------------------------
    # Jika model tidak ada atau merupakan Git LFS pointer,
    # download model asli dari GitHub Release.
    # --------------------------------------------------------

    if not model_file_is_valid():

        download_model()

    # --------------------------------------------------------
    # Load Random Forest
    # --------------------------------------------------------

    print("Memuat model Random Forest...")

    model = joblib.load(MODEL_PATH)

    MODEL_LOADED = True

    print("Model Random Forest berhasil dimuat.")


except Exception as e:

    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)

    print("Gagal memuat model.")
    print(f"Error: {MODEL_LOAD_ERROR}")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Phishing URL Detection API",
    description="API untuk mendeteksi URL phishing menggunakan Random Forest.",
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class URLRequest(BaseModel):
    url: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Phishing URL Detection API",
        "model": "Random Forest",
        "model_loaded": MODEL_LOADED,
        "classification": {
            "0": "PHISHING",
            "1": "LEGITIMATE"
        }
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    response = {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "model": "Random Forest",
        "classification": {
            "0": "PHISHING",
            "1": "LEGITIMATE"
        }
    }

    if not MODEL_LOADED:

        response["error"] = MODEL_LOAD_ERROR

    return response


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: URLRequest):

    # --------------------------------------------------------
    # Pastikan model tersedia
    # --------------------------------------------------------

    if not MODEL_LOADED:

        raise HTTPException(
            status_code=500,
            detail="Model Random Forest tidak berhasil dimuat."
        )

    # --------------------------------------------------------
    # Bersihkan URL
    # --------------------------------------------------------

    url = request.url.strip()

    # --------------------------------------------------------
    # Validasi URL kosong
    # --------------------------------------------------------

    if not url:

        raise HTTPException(
            status_code=400,
            detail="URL tidak boleh kosong."
        )

    # --------------------------------------------------------
    # Validasi panjang URL
    # --------------------------------------------------------

    if len(url) > 10000:

        raise HTTPException(
            status_code=400,
            detail="URL terlalu panjang."
        )

    try:

        # ====================================================
        # 1. EKSTRAK FITUR
        # ====================================================

        feature_dict = extract_features(url)

        # ====================================================
        # 2. UBAH FITUR MENJADI DATAFRAME
        # ====================================================

        X = pd.DataFrame([feature_dict])

        # ====================================================
        # 3. RANDOM FOREST PREDICTION
        # ====================================================
        #
        # model.predict() langsung menentukan kelas.
        #
        # 0 = PHISHING
        # 1 = LEGITIMATE
        #
        # Tidak menggunakan threshold.
        # Tidak menggunakan custom scoring.
        # Tidak menggunakan PhishDestroy.
        #
        # ====================================================

        prediction = int(model.predict(X)[0])

        # ====================================================
        # 4. UBAH HASIL MENJADI LABEL
        # ====================================================

        if prediction == 0:

            result = "PHISHING"

        else:

            result = "LEGITIMATE"

        # ====================================================
        # 5. OPTIONAL PROBABILITY
        # ====================================================
        #
        # Probability hanya sebagai informasi.
        # Tidak digunakan untuk menentukan hasil.
        #
        # ====================================================

        phishing_probability = None
        legitimate_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X)[0]

            classes = model.classes_

            for class_value, probability in zip(
                classes,
                probabilities
            ):

                if int(class_value) == 0:

                    phishing_probability = float(
                        probability
                    )

                elif int(class_value) == 1:

                    legitimate_probability = float(
                        probability
                    )

        # ====================================================
        # 6. RESPONSE
        # ====================================================

        return {
            "url": url,

            "prediction": result,

            "prediction_class": prediction,

            "phishing_probability": (
                round(
                    phishing_probability,
                    6
                )
                if phishing_probability is not None
                else None
            ),

            "legitimate_probability": (
                round(
                    legitimate_probability,
                    6
                )
                if legitimate_probability is not None
                else None
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses URL: {str(e)}"
        )