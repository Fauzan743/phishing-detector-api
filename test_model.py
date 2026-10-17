import joblib
import pandas as pd

from feature_extractor import extract_features


# ============================================================
# 1. LOAD MODEL
# ============================================================

MODEL_PATH = "models/random_forest_url.pkl"

print("Memuat model...")

model = joblib.load(MODEL_PATH)

print("Model berhasil dimuat.")


# ============================================================
# 2. FUNGSI PREDIKSI
# ============================================================

def predict_url(url):
    """
    Mengubah URL menjadi fitur kemudian
    melakukan klasifikasi menggunakan Random Forest.
    """

    features = extract_features(url)

    # Ubah dictionary fitur menjadi DataFrame
    X = pd.DataFrame([features])

    # Prediksi menggunakan Random Forest
    prediction = model.predict(X)[0]

    return prediction


# ============================================================
# 3. URL YANG DIUJI
# ============================================================

test_urls = [

    # --------------------------------------------------------
    # URL LEGITIMATE
    # --------------------------------------------------------

    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.wikipedia.org",
    "https://www.facebook.com",
    "https://www.youtube.com",


    # --------------------------------------------------------
    # URL SIMULASI PHISHING
    # --------------------------------------------------------

    "http://google-login-security.example.com/verify-account",
    "http://microsoft-account-security.example.com/login",
    "http://paypal-verification.example.com/secure/login",
    "http://facebook-security.example.com/verify",
    "http://bank-account-verification.example.com/login"
]


# ============================================================
# 4. PENGUJIAN
# ============================================================

print("\n========================================")
print("PENGUJIAN RANDOM FOREST")
print("========================================")


for url in test_urls:

    prediction = predict_url(url)

    if prediction == 0:
        result = "PHISHING"
    else:
        result = "LEGITIMATE"

    print("\nURL        :", url)
    print("Prediction :", prediction)
    print("Hasil      :", result)


# ============================================================
# 5. SELESAI
# ============================================================

print("\n========================================")
print("PENGUJIAN SELESAI")
print("========================================")