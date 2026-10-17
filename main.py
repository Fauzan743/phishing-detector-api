from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import pandas as pd

from feature_extractor import extract_features


# ============================================================
# KONFIGURASI
# ============================================================

MODEL_PATH = "models/random_forest_url.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    model = None
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)


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
        # Tidak menggunakan threshold.
        #
        # model.predict() langsung menentukan kelas:
        #
        # 0 = PHISHING
        # 1 = LEGITIMATE
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
                    phishing_probability = float(probability)

                elif int(class_value) == 1:
                    legitimate_probability = float(probability)

        # ====================================================
        # 6. RESPONSE
        # ====================================================

        return {
            "url": url,
            "prediction": result,
            "prediction_class": prediction,
            "phishing_probability": (
                round(phishing_probability, 6)
                if phishing_probability is not None
                else None
            ),
            "legitimate_probability": (
                round(legitimate_probability, 6)
                if legitimate_probability is not None
                else None
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses URL: {str(e)}"
        )