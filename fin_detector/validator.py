import re

FIN_PATTERN = re.compile(r'^[A-Z0-9]{7}$')
ID_NUMBER_OLD = re.compile(r'^[A-Z]{3}\d{7,9}$')   # e.g., AZE14431397
ID_NUMBER_NEW = re.compile(r'^[A-Z]{2}\d{7}$')      # e.g., AA3778866

def is_valid_fin(fin: str) -> bool:
    """Returns True if fin matches the 7-char alphanumeric pattern."""
    if not fin:
        return False
    return bool(FIN_PATTERN.match(fin))

def is_valid_id_number(id_num: str) -> bool:
    """Returns True if id_num matches standard Azerbaijani ID number patterns."""
    if not id_num:
        return False
    return bool(ID_NUMBER_OLD.match(id_num) or ID_NUMBER_NEW.match(id_num))

def clean_ocr_text(text: str) -> str:
    """Strip whitespace and convert to uppercase."""
    if not text:
        return ""
    return text.strip().upper()

def clean_mrz_line(text: str) -> str:
    """Clean OCR noises from MRZ line: replace brackets/commas with '<'."""
    if not text:
        return ""
    # Standard cleanup of MRZ OCR noise
    text = text.strip().upper()
    # Replace common characters misidentified as '<'
    for char in ['(', ')', '{', '}', '[', ']', ',', '.', ';', ':', '`', '"', "'"]:
        text = text.replace(char, '<')
    # Replace spaces with '<' as MRZ lines shouldn't have spaces
    text = text.replace(' ', '<')
    return text

def compute_mrz_check_digit(data: str) -> int:
    """ICAO Doc 9303 check digit computation."""
    weights = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        if '0' <= char <= '9':
            val = int(char)
        elif 'A' <= char <= 'Z':
            val = ord(char) - ord('A') + 10
        elif char == '<':
            val = 0
        else:
            val = 0
        total += val * weights[i % 3]
    return total % 10
