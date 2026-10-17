import re
from urllib.parse import urlparse


def extract_features(url):
    """
    Mengubah URL menjadi fitur-fitur numerik
    yang dapat digunakan oleh Random Forest.
    """

    # Pastikan URL berupa string
    url = str(url)

    # Tambahkan scheme sementara jika URL tidak memilikinya
    parse_url = url
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", parse_url):
        parse_url = "http://" + parse_url

    parsed = urlparse(parse_url)

    domain = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query

    # Hilangkan port dari domain
    domain_without_port = domain.split(":")[0]

    # ---------------------------------------------------------
    # 1. Panjang URL
    # ---------------------------------------------------------

    url_length = len(url)

    # ---------------------------------------------------------
    # 2. Panjang domain
    # ---------------------------------------------------------

    domain_length = len(domain_without_port)

    # ---------------------------------------------------------
    # 3. Apakah domain berupa IP address
    # ---------------------------------------------------------

    is_domain_ip = 1 if re.match(
        r"^(?:\d{1,3}\.){3}\d{1,3}$",
        domain_without_port
    ) else 0

    # ---------------------------------------------------------
    # 4. TLD
    # ---------------------------------------------------------

    parts = domain_without_port.split(".")

    if len(parts) >= 2:
        tld = parts[-1]
    else:
        tld = ""

    tld_length = len(tld)

    # ---------------------------------------------------------
    # 5. Jumlah subdomain
    # ---------------------------------------------------------

    if len(parts) > 2:
        no_of_subdomain = len(parts) - 2
    else:
        no_of_subdomain = 0

    # ---------------------------------------------------------
    # 6. Obfuscation
    # ---------------------------------------------------------

    obfuscated_chars = len(re.findall(r"%[0-9a-fA-F]{2}", url))

    has_obfuscation = 1 if obfuscated_chars > 0 else 0

    obfuscation_ratio = (
        obfuscated_chars / url_length
        if url_length > 0
        else 0
    )

    # ---------------------------------------------------------
    # 7. Jumlah huruf
    # ---------------------------------------------------------

    no_of_letters = sum(c.isalpha() for c in url)

    letter_ratio = (
        no_of_letters / url_length
        if url_length > 0
        else 0
    )

    # ---------------------------------------------------------
    # 8. Jumlah angka
    # ---------------------------------------------------------

    no_of_digits = sum(c.isdigit() for c in url)

    digit_ratio = (
        no_of_digits / url_length
        if url_length > 0
        else 0
    )

    # ---------------------------------------------------------
    # 9. Karakter khusus
    # ---------------------------------------------------------

    no_of_equals = url.count("=")

    no_of_qmark = url.count("?")

    no_of_ampersand = url.count("&")

    # Karakter khusus selain karakter yang umum
    special_chars = re.findall(
        r"[^a-zA-Z0-9]",
        url
    )

    no_of_other_special_chars = len(special_chars)

    special_char_ratio = (
        no_of_other_special_chars / url_length
        if url_length > 0
        else 0
    )

    # ---------------------------------------------------------
    # 10. HTTPS
    # ---------------------------------------------------------

    is_https = 1 if parsed.scheme.lower() == "https" else 0

    # ---------------------------------------------------------
    # Kembalikan fitur
    # ---------------------------------------------------------

    return {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "TLD": tld,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": obfuscated_chars,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special_chars,
        "SpacialCharRatioInURL": special_char_ratio,
        "IsHTTPS": is_https
    }