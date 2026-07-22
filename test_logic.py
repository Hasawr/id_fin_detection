from fin_detector.validator import compute_mrz_check_digit, clean_mrz_line, clean_ocr_text, is_valid_fin
from fin_detector.mrz_extractor import MRZExtractor
import re

def test_check_digit():
    print("Testing check digit calculation...")
    # Test case 1: Example from search result: AB2134 -> 5
    # (A=10, B=11, 2, 1, 3, 4)
    # Weights: 7, 3, 1, 7, 3, 1
    # Products: 70, 33, 2, 7, 9, 4 -> Sum = 125 -> 125 % 10 = 5.
    cd1 = compute_mrz_check_digit("AB2134")
    assert cd1 == 5, f"Expected 5, got {cd1}"
    print("Test case 1 passed!")

    # Test case 2: Document number 1234567<
    # Weights: 7, 3, 1, 7, 3, 1, 7, 3
    # 1*7 + 2*3 + 3*1 + 4*7 + 5*3 + 6*1 + 7*7 + 0*3 = 114 -> 114 % 10 = 4.
    cd2 = compute_mrz_check_digit("1234567<")
    assert cd2 == 4, f"Expected 4, got {cd2}"
    print("Test case 2 passed!")

def test_mrz_parsing_logic():
    print("Testing MRZ FIN extraction parsing logic...")
    # Simulated Line 1
    line1 = "IDAZE1234567<0AZE7890123<<<<<<"
    # Length of line1 is 30
    assert len(line1) == 30
    
    # In our MRZExtractor:
    # optional_data is line1[14:29] -> indices 14 to 28 inclusive (length 15)
    # line1[14] is character 15.
    optional_data = line1[14:29]
    print(f"Sub-string line1[14:29]: '{optional_data}'")
    assert optional_data == "AZE7890123<<<<<"
    
    opt_clean = optional_data.rstrip('<')
    print(f"Cleaned optional data: '{opt_clean}'")
    assert opt_clean == "AZE7890123"
    
    issuing_state = line1[2:5]
    print(f"Issuing state: '{issuing_state}'")
    assert issuing_state == "AZE"
    
    fin_candidate = None
    if opt_clean.startswith(issuing_state) and len(opt_clean) >= 3 + 7:
        fin_candidate = opt_clean[3:10]
        
    print(f"Extracted FIN candidate: '{fin_candidate}'")
    assert fin_candidate == "7890123"
    assert is_valid_fin(fin_candidate)
    print("MRZ parsing logic passed!")

if __name__ == "__main__":
    test_check_digit()
    print("-" * 40)
    test_mrz_parsing_logic()
    print("-" * 40)
    print("All unit tests passed successfully!")
