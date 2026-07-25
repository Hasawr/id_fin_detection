import re


FIN_PATTERN = re.compile(r"^[A-Z0-9]{7}$")
OLD_CARD_SERIAL_PATTERN = re.compile(r"^[0-9]{7,9}$")
NEW_CARD_SERIAL_PATTERN = re.compile(r"^A[AB][0-9]{7}$")

# Common OCR-B confusions for digit-only MRZ fields (dates, check digits, numbers).
OCR_DIGIT_CONFUSIONS = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }
)


def is_valid_fin(fin: str) -> bool:
    return bool(fin and FIN_PATTERN.match(fin))


def is_valid_old_card_serial(serial: str) -> bool:
    return bool(serial and OLD_CARD_SERIAL_PATTERN.match(serial))


def is_valid_new_card_serial(serial: str) -> bool:
    return bool(serial and NEW_CARD_SERIAL_PATTERN.match(serial))


def correct_ocr_digits(value: str) -> str:
    """Map letter shapes that OCR often confuses with digits."""
    if not value:
        return ""
    return value.upper().translate(OCR_DIGIT_CONFUSIONS)


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
