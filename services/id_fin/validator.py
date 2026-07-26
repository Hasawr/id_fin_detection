import re


# Letter O never appears in a real FIN or card serial number - only the
# digit 0 does. Excluding O here is a defense-in-depth backstop; callers
# should also normalize with normalize_fin_o0() before validating.
FIN_PATTERN = re.compile(r"^[A-NP-Z0-9]{7}$")
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


def normalize_fin_o0(value: str) -> str:
    """Correct OCR readings of the letter O in a FIN or serial number.

    Real Azerbaijani FINs and card serial numbers never contain the letter
    O - only the digit 0. Unlike other OCR-B letter/digit confusions, this
    one is unconditional: there is no legitimate FIN character it could be
    mistaken for, so every O is always the digit 0.
    """
    return value.upper().replace("O", "0") if value else value


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
