from services.id_fin.validator import (
    clean_mrz_line,
    compute_mrz_check_digit,
    is_valid_fin,
)
from services.id_fin.mrz_extractor import MRZExtractor


def test_check_digit() -> None:
    assert compute_mrz_check_digit("AB2134") == 5
    assert compute_mrz_check_digit("1234567<") == 4


def test_fin_validation() -> None:
    assert is_valid_fin("7ABC123")
    assert not is_valid_fin("TOO-LONG")


def test_mrz_cleanup() -> None:
    assert clean_mrz_line("ID AZE,123") == "ID<AZE<123"
    assert clean_mrz_line("AZE$123") == "AZE<123"


def test_old_card_td2_fin_extraction() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("19<01<1980<<<<<<<<<<<<<<<<<<<<", 0.95),
        ("I<AZERASHIDOVAINARA<<<<<<<<<<", 0.98),
        ("14433235<1AZE8001195F30011901HNLXEM7", 0.99),
    ]

    pair = extractor._find_td2_pair(merged_lines)
    assert pair is not None

    result = extractor._parse_td2(pair, is_cropped=False)
    assert result.fin == "1HNLXEM"
    assert result.card_type == "older_card"
    assert result.checksum_valid is True
    assert result.line3 == ""


def test_exact_new_and_old_card_mrz_layouts() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    new_card = extractor._parse_td1(
        [
            ("IAAZEAA203982795H0H4NX<<<<<<<<", 0.99),
            ("9601226M3006162AZE<<<<<<<<<<<5", 0.99),
            ("VALIYEV<<OMAR<<<<<<<<<<<<<<<<<", 0.99),
        ],
        is_cropped=False,
    )
    old_card = extractor._parse_td2(
        (
            ("I<AZEGOJAYEV<<AYKHAN<<<<<<<<<<<<<<<<", 0.99),
            ("19205792<7AZE8210276M32102712BDLLON5", 0.99),
        ),
        is_cropped=False,
    )

    assert new_card.fin == "5H0H4NX"
    assert new_card.card_type == "new_card"
    assert all(
        len(line) == 30
        for line in (new_card.line1, new_card.line2, new_card.line3)
    )
    assert old_card.fin == "2BDLLON"
    assert old_card.card_type == "older_card"
    assert len(old_card.line1) == 36
    assert len(old_card.line2) == 36
    assert old_card.line3 == ""


def test_truncated_old_card_is_not_classified_as_new() -> None:
    extractor = MRZExtractor.__new__(MRZExtractor)
    merged_lines = [
        ("19<01<1980<<<<<<<<<<<<<<<<<<<<", 0.95),
        ("I<AZERASHIDOVAINARA<<<<<<<<<<", 0.98),
        ("14433235<1AZE8001195F30011901H", 0.99),
    ]

    pair = extractor._find_td2_pair(merged_lines)
    assert pair is not None
    old_result = extractor._parse_td2(pair, is_cropped=False)
    fallback_result = extractor._parse_td1(merged_lines, is_cropped=False)

    assert old_result.card_type == "older_card"
    assert old_result.fin is None
    assert fallback_result.card_type == "unknown"
