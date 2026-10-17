import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from feature_extractor import extract_features


# ============================================================
# 1. KONFIGURASI
# ============================================================

DATASET_PATH = "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = "models/random_forest_url.pkl"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Memuat dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Jumlah data awal : {len(df):,}")


# ============================================================
# 3. HAPUS URL DUPLIKAT
# ============================================================

df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

print(f"Jumlah data setelah hapus duplikat : {len(df):,}")


# ============================================================
# 4. EKSTRAK FITUR DARI URL
# ============================================================

print("\nMengekstrak fitur URL...")

features = df["URL"].apply(extract_features)

X = pd.DataFrame(features.tolist())

# Label PhiUSIIL:
# 0 = Phishing
# 1 = Legitimate
y = df["label"]


print(f"Jumlah fitur : {X.shape[1]}")
print(f"Jumlah data  : {X.shape[0]}")


# ============================================================
# 5. CEK FITUR
# ============================================================

print("\nNama fitur:")

for i, column in enumerate(X.columns, start=1):
    print(f"{i:2}. {column}")


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\nMembagi dataset menjadi train dan test...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Data training : {len(X_train):,}")
print(f"Data testing  : {len(X_test):,}")


# ============================================================
# 7. TENTUKAN FITUR NUMERIK DAN KATEGORIK
# ============================================================

# TLD merupakan data kategorik karena berisi teks,
# misalnya: com, org, net, id, dll.
categorical_features = ["TLD"]

# Semua fitur selain TLD merupakan fitur numerik.
numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


print("\nFitur kategorik:")
print(categorical_features)

print("\nJumlah fitur numerik:", len(numeric_features))


# ============================================================
# 8. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# 9. RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ============================================================
# 10. PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("random_forest", random_forest)
    ]
)


# ============================================================
# 11. TRAINING
# ============================================================

print("\n========================================")
print("TRAINING RANDOM FOREST")
print("========================================")

model.fit(X_train, y_train)

print("Training selesai.")


# ============================================================
# 12. PREDIKSI
# ============================================================

print("\nMelakukan prediksi pada data testing...")

# Keputusan klasifikasi menggunakan predict()
# tanpa threshold tambahan.
y_pred = model.predict(X_test)


# ============================================================
# 13. EVALUASI
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("HASIL EVALUASI")
print("========================================")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Phishing (0)",
            "Legitimate (1)"
        ]
    )
)


print("Confusion Matrix:")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ============================================================
# 14. SIMPAN MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)


print("\n========================================")
print("MODEL BERHASIL DISIMPAN")
print("========================================")

print(f"Lokasi model : {MODEL_PATH}")