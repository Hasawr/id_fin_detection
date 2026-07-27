from dataclasses import dataclass, field

@dataclass
class VIZResult:
    fin: str | None
    confidence: float
    bbox: list | None   # Bounding box of the detected FIN value on VIZ
    method: str         # "label_proximity" | "blind_regex" | "not_found"

@dataclass
class MRZResult:
    fin: str | None
    confidence: float
    line1: str
    line2: str
    line3: str
    checksum_valid: bool
    method: str         # "mrz_strip" | "td2_extract" | "td2_full_image_fallback" | "full_image_fallback" | "not_found"
    id_number: str | None = None

    @property
    def is_old_card(self) -> bool:
        """True if FIN was extracted using old card (TD2) logic."""
        return self.method in ("old_card_line2", "old_card_line2_fallback", "td2_extract", "td2_full_image_fallback")

    @property
    def card_format(self) -> str:
        """Return the detected MRZ format type."""
        if self.is_old_card:
            return "TD2 (2-line, old card)"
        return "TD1 (3-line, new biometric card)"


@dataclass
class FINDetectionOutput:
    viz_fin: str | None
    mrz_fin: str | None
    viz_confidence: float
    mrz_confidence: float
    mrz_id_number: str | None = None
    viz_result: VIZResult | None = None
    mrz_result: MRZResult | None = None
    notes: list[str] = field(default_factory=list)
