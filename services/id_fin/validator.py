import re


FIN_PATTERN = re.compile(r"^[A-Z0-9]{7}$")


def is_valid_fin(fin: str) -> bool:
    return bool(fin and FIN_PATTERN.match(fin))


def clean_ocr_text(text: str) -> str:
    return text.strip().upper() if text else ""


def clean_mrz_line(text: str) -> str:
    if not text:
        return ""
    text = text.strip().upper()
    for char in ["(", ")", "{", "}", "[", "]", ",", ".", ";", ":", "`", '"', "'"]:
        text = text.replace(char, "<")
    text = text.replace(" ", "<")
    return re.sub(r"[^A-Z0-9<]", "<", text)


def compute_mrz_check_digit(data: str) -> int:
    weights = [7, 3, 1]
    total = 0
    for index, char in enumerate(data):
        if "0" <= char <= "9":
            value = int(char)
        elif "A" <= char <= "Z":
            value = ord(char) - ord("A") + 10
        else:
            value = 0
        total += value * weights[index % 3]
    return total % 10
