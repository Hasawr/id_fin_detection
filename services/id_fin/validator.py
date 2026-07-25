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


def iter_o0_variants(value: str) -> list[str]:
    """Return O/0 substitution variants; original OCR text stays first."""
    if not value:
        return []
    normalized = value.upper()
    positions = [index for index, char in enumerate(normalized) if char in "O0"]
    if not positions:
        return [normalized]

    from itertools import product

    ordered = [normalized]
    seen = {normalized}
    for choices in product("0O", repeat=len(positions)):
        chars = list(normalized)
        for position, choice in zip(positions, choices, strict=True):
            chars[position] = choice
        candidate = "".join(chars)
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def resolve_fin_o0_ambiguity(
    value: str,
    *,
    checksum_ok=None,
) -> str | None:
    """Pick FIN among O/0 variants; trust OCR unless a checksum forces a flip.

    Letter O and digit 0 are both valid in Azerbaijani FINs, so neighbor
    heuristics are unsafe. Only rewrite when the original reading fails a
    provided checksum and exactly one O/0 variant passes.
    """
    if not value:
        return None
    variants = [item for item in iter_o0_variants(value) if is_valid_fin(item)]
    if not variants:
        return None
    original = variants[0]
    if checksum_ok is None:
        return original
    if checksum_ok(original):
        return original
    matching = [item for item in variants[1:] if checksum_ok(item)]
    if len(matching) == 1:
        return matching[0]
    return original


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
