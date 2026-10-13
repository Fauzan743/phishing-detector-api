from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import math
import re
import requests
from urllib.parse import urlparse


# ============================================================
# KONFIGURASI
# ============================================================

MODEL_PATH = "models/random_forest_engineered.pkl"
THRESHOLD = 0.343


# Load model bundle
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
FEATURES = bundle["features"]


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Phishing URL Detection API",
    description=(
        "API untuk mendeteksi URL phishing menggunakan "
        "Random Forest dan PhishDestroy Threat Intelligence."
    ),
    version="1.1.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class URLRequest(BaseModel):
    url: str


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_entropy(text: str) -> float:
    """Menghitung Shannon entropy."""

    if not text:
        return 0.0

    probabilities = [
        text.count(char) / len(text)
        for char in set(text)
    ]

    return -sum(
        p * math.log2(p)
        for p in probabilities
        if p > 0
    )


def extract_features(url: str) -> dict:
    """
    Menghasilkan 56 fitur yang sama dengan model training.
    """

    original_url = url.strip()

    # URL parsing
    parse_url = original_url

    if "://" not in parse_url:
        parse_url = "//" + parse_url

    try:
        parsed = urlparse(parse_url)
    except Exception:
        parsed = urlparse("//")

    hostname = parsed.hostname or ""
    hostname = hostname.lower()

    path = parsed.path or ""
    query = parsed.query or ""

    # --------------------------------------------------------
    # Basic URL features
    # --------------------------------------------------------

    url_len = len(original_url)
    dom_len = len(hostname)

    # IP detection
    ipv4_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    is_ip = int(bool(re.match(ipv4_pattern, hostname)))

    # TLD
    hostname_parts = hostname.split(".") if hostname else []

    if hostname_parts:
        tld = hostname_parts[-1]
    else:
        tld = ""

    tld_len = len(tld)

    # Subdomain
    subdom_cnt = max(len(hostname_parts) - 2, 0)

    # Character counts
    letter_cnt = sum(
        c.isalpha()
        for c in original_url
    )

    digit_cnt = sum(
        c.isdigit()
        for c in original_url
    )

    special_cnt = sum(
        not c.isalnum()
        for c in original_url
    )

    eq_cnt = original_url.count("=")
    qm_cnt = original_url.count("?")
    amp_cnt = original_url.count("&")
    dot_cnt = original_url.count(".")
    dash_cnt = original_url.count("-")
    under_cnt = original_url.count("_")

    # Ratios
    letter_ratio = (
        letter_cnt / url_len
        if url_len > 0
        else 0
    )

    digit_ratio = (
        digit_cnt / url_len
        if url_len > 0
        else 0
    )

    spec_ratio = (
        special_cnt / url_len
        if url_len > 0
        else 0
    )

    is_https = int(
        original_url.lower().startswith("https://")
    )

    slash_cnt = original_url.count("/")

    entropy = calculate_entropy(original_url)

    path_len = len(path)
    query_len = len(query)

    # --------------------------------------------------------
    # Engineered features
    # --------------------------------------------------------

    hostname_digit_count = sum(
        c.isdigit()
        for c in hostname
    )

    hostname_letter_count = sum(
        c.isalpha()
        for c in hostname
    )

    hostname_dash_count = hostname.count("-")

    hostname_dot_count = hostname.count(".")

    hostname_digit_ratio = (
        hostname_digit_count / len(hostname)
        if hostname
        else 0
    )

    hostname_letter_ratio = (
        hostname_letter_count / len(hostname)
        if hostname
        else 0
    )

    hostname_dash_ratio = (
        hostname_dash_count / len(hostname)
        if hostname
        else 0
    )

    fe_subdomain_count = max(
        len(hostname_parts) - 2,
        0
    )

    fe_path_url_ratio = (
        path_len / url_len
        if url_len > 0
        else 0
    )

    fe_query_url_ratio = (
        query_len / url_len
        if url_len > 0
        else 0
    )

    fe_hostname_url_ratio = (
        len(hostname) / url_len
        if url_len > 0
        else 0
    )

    fe_special_url_ratio = (
        special_cnt / url_len
        if url_len > 0
        else 0
    )

    fe_digit_url_ratio = (
        digit_cnt / url_len
        if url_len > 0
        else 0
    )

    fe_at_count = original_url.count("@")

    fe_percent_count = original_url.count("%")

    fe_encoded_count = len(
        re.findall(
            r"%[0-9a-fA-F]{2}",
            original_url
        )
    )

    fe_semicolon_count = original_url.count(";")
    fe_colon_count = original_url.count(":")
    fe_comma_count = original_url.count(",")
    fe_plus_count = original_url.count("+")
    fe_tilde_count = original_url.count("~")
    fe_underscore_count = original_url.count("_")
    fe_pipe_count = original_url.count("|")
    fe_star_count = original_url.count("*")

    fe_paren_count = (
        original_url.count("(")
        + original_url.count(")")
    )

    # --------------------------------------------------------
    # Port
    # --------------------------------------------------------

    try:
        port = parsed.port
    except ValueError:
        port = None

    fe_has_port = int(
        port is not None
    )

    fe_nonstandard_port = int(
        port is not None
        and port not in (80, 443)
    )

    # --------------------------------------------------------
    # Path segments
    # --------------------------------------------------------

    path_segments = [
        segment
        for segment in path.split("/")
        if segment
    ]

    fe_path_segment_count = len(
        path_segments
    )

    # --------------------------------------------------------
    # Query parameters
    # --------------------------------------------------------

    query_parameters = [
        parameter
        for parameter in query.split("&")
        if parameter
    ]

    fe_query_parameter_count = len(
        query_parameters
    )

    # --------------------------------------------------------
    # Double slash
    # --------------------------------------------------------

    fe_double_slash_count = original_url.count("//")

    # --------------------------------------------------------
    # Punycode
    # --------------------------------------------------------

    fe_has_punycode = int(
        "xn--" in hostname.lower()
    )

    # --------------------------------------------------------
    # IPv4-like
    # --------------------------------------------------------

    fe_ipv4_like = int(
        bool(
            re.search(
                r"(?:\d{1,3}\.){2,3}\d{1,3}",
                hostname
            )
        )
    )

    # --------------------------------------------------------
    # Long hostname label
    # --------------------------------------------------------

    hostname_labels = [
        label
        for label in hostname.split(".")
        if label
    ]

    max_label_length = max(
        (
            len(label)
            for label in hostname_labels
        ),
        default=0
    )

    fe_long_hostname_label = int(
        max_label_length >= 20
    )

    fe_max_subdomain_label_length = (
        max_label_length
    )

    # --------------------------------------------------------
    # Gabungkan semua fitur
    # --------------------------------------------------------

    features = {

        "url_len": url_len,
        "dom_len": dom_len,
        "is_ip": is_ip,
        "tld_len": tld_len,
        "subdom_cnt": subdom_cnt,
        "letter_cnt": letter_cnt,
        "digit_cnt": digit_cnt,
        "special_cnt": special_cnt,
        "eq_cnt": eq_cnt,
        "qm_cnt": qm_cnt,
        "amp_cnt": amp_cnt,
        "dot_cnt": dot_cnt,
        "dash_cnt": dash_cnt,
        "under_cnt": under_cnt,

        "letter_ratio": letter_ratio,
        "digit_ratio": digit_ratio,
        "spec_ratio": spec_ratio,
        "is_https": is_https,
        "slash_cnt": slash_cnt,
        "entropy": entropy,
        "path_len": path_len,
        "query_len": query_len,

        "fe_hostname_digit_count": (
            hostname_digit_count
        ),

        "fe_hostname_letter_count": (
            hostname_letter_count
        ),

        "fe_hostname_dash_count": (
            hostname_dash_count
        ),

        "fe_hostname_dot_count": (
            hostname_dot_count
        ),

        "fe_hostname_digit_ratio": (
            hostname_digit_ratio
        ),

        "fe_hostname_letter_ratio": (
            hostname_letter_ratio
        ),

        "fe_hostname_dash_ratio": (
            hostname_dash_ratio
        ),

        "fe_subdomain_count": (
            fe_subdomain_count
        ),

        "fe_path_url_ratio": (
            fe_path_url_ratio
        ),

        "fe_query_url_ratio": (
            fe_query_url_ratio
        ),

        "fe_hostname_url_ratio": (
            fe_hostname_url_ratio
        ),

        "fe_special_url_ratio": (
            fe_special_url_ratio
        ),

        "fe_digit_url_ratio": (
            fe_digit_url_ratio
        ),

        "fe_at_count": fe_at_count,
        "fe_percent_count": fe_percent_count,
        "fe_encoded_count": fe_encoded_count,
        "fe_semicolon_count": fe_semicolon_count,
        "fe_colon_count": fe_colon_count,
        "fe_comma_count": fe_comma_count,
        "fe_plus_count": fe_plus_count,
        "fe_tilde_count": fe_tilde_count,
        "fe_underscore_count": fe_underscore_count,
        "fe_pipe_count": fe_pipe_count,
        "fe_star_count": fe_star_count,
        "fe_paren_count": fe_paren_count,

        "fe_has_port": fe_has_port,
        "fe_nonstandard_port": (
            fe_nonstandard_port
        ),

        "fe_path_segment_count": (
            fe_path_segment_count
        ),

        "fe_query_parameter_count": (
            fe_query_parameter_count
        ),

        "fe_double_slash_count": (
            fe_double_slash_count
        ),

        "fe_has_punycode": (
            fe_has_punycode
        ),

        "fe_ipv4_like": fe_ipv4_like,

        "fe_long_hostname_label": (
            fe_long_hostname_label
        ),

        "fe_max_subdomain_label_length": (
            fe_max_subdomain_label_length
        ),
    }

    return features


# ============================================================
# PHISHDESTROY THREAT INTELLIGENCE
# ============================================================

def check_phishdestroy(url: str) -> dict:
    """
    Mengecek domain URL ke PhishDestroy.

    PhishDestroy digunakan sebagai lapisan threat intelligence
    untuk mengecek apakah domain sudah terdapat dalam blocklist.
    """

    # Pastikan URL dapat diproses oleh urlparse
    parse_url = url.strip()

    if "://" not in parse_url:
        parse_url = "//" + parse_url

    try:
        parsed = urlparse(parse_url)
        domain = parsed.hostname or ""
        domain = domain.lower()

    except Exception:
        domain = ""

    # Jika domain tidak berhasil diperoleh
    if not domain:
        return {
            "available": False,
            "domain": "",
            "threat": False,
            "message": (
                "Domain tidak dapat diambil dari URL."
            ),
        }

    try:
        response = requests.get(
            "https://api.destroy.tools/v1/check",
            params={
                "domain": domain
            },
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        return {
            "available": True,
            "domain": result.get(
                "domain",
                domain
            ),
            "threat": bool(
                result.get(
                    "threat",
                    False
                )
            ),
            "message": result.get(
                "message",
                ""
            ),
        }

    except requests.RequestException as e:

        # Jika PhishDestroy tidak dapat diakses,
        # Random Forest tetap digunakan.
        return {
            "available": False,
            "domain": domain,
            "threat": False,
            "message": (
                "PhishDestroy tidak dapat diakses: "
                f"{str(e)}"
            ),
        }

    except ValueError:

        return {
            "available": False,
            "domain": domain,
            "threat": False,
            "message": (
                "Respons PhishDestroy tidak valid."
            ),
        }


# ============================================================
# PREDICTION
# ============================================================


@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Phishing URL Detection API",
        "model": "Random Forest Engineered",
        "threat_intelligence": "PhishDestroy",
        "features": len(FEATURES),
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(FEATURES),
        "threat_intelligence": "PhishDestroy",
    }


@app.post("/predict")
def predict(request: URLRequest):

    url = request.url.strip()

    # --------------------------------------------------------
    # Validasi URL
    # --------------------------------------------------------

    if not url:

        raise HTTPException(
            status_code=400,
            detail="URL tidak boleh kosong."
        )

    if len(url) > 10000:

        raise HTTPException(
            status_code=400,
            detail="URL terlalu panjang."
        )

    try:

        # ====================================================
        # 1. RANDOM FOREST
        # ====================================================

        feature_dict = extract_features(url)

        # Gunakan URUTAN fitur persis dari model
        X = pd.DataFrame(
            [
                [
                    feature_dict[f]
                    for f in FEATURES
                ]
            ],
            columns=FEATURES,
        )

        # Probability class
        probabilities = model.predict_proba(X)[0]

        # Random Forest:
        # class 0 = legitimate
        # class 1 = phishing

        phishing_probability = float(
            probabilities[1]
        )

        legitimate_probability = float(
            probabilities[0]
        )

        # ====================================================
        # 2. PHISHDESTROY
        # ====================================================

        phishdestroy_result = (
            check_phishdestroy(url)
        )

        phishdestroy_available = bool(
            phishdestroy_result.get(
                "available",
                False
            )
        )

        phishdestroy_threat = bool(
            phishdestroy_result.get(
                "threat",
                False
            )
        )

        # ====================================================
        # 3. FINAL DECISION
        # ====================================================

        # URL dianggap PHISHING apabila:
        #
        # A. Random Forest memberikan probability
        #    phishing >= threshold
        #
        # ATAU
        #
        # B. PhishDestroy mendeteksi domain sebagai threat.
        #
        # Jika PhishDestroy sedang tidak tersedia,
        # keputusan tetap menggunakan Random Forest.

        if phishing_probability >= THRESHOLD:

            prediction = "PHISHING"

        elif (
            phishdestroy_available
            and phishdestroy_threat
        ):

            prediction = "PHISHING"

        else:

            prediction = "LEGITIMATE"

        # ====================================================
        # 4. RESPONSE
        # ====================================================

        return {

            "url": url,

            "prediction": prediction,

            "phishing_probability": round(
                phishing_probability,
                6
            ),

            "legitimate_probability": round(
                legitimate_probability,
                6
            ),

            "threshold": THRESHOLD,

            "phishdestroy": {

                "available": (
                    phishdestroy_available
                ),

                "domain": (
                    phishdestroy_result.get(
                        "domain",
                        ""
                    )
                ),

                "threat": (
                    phishdestroy_threat
                ),

                "message": (
                    phishdestroy_result.get(
                        "message",
                        ""
                    )
                ),
            },
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Gagal memproses URL: {str(e)}"
            )
        )