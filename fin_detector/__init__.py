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
    method: str         # "mrz_strip" | "full_image_fallback" | "not_found"

@dataclass
class FINDetectionOutput:
    viz_fin: str | None
    mrz_fin: str | None
    viz_confidence: float
    mrz_confidence: float
    viz_result: VIZResult | None = None
    mrz_result: MRZResult | None = None
    notes: list[str] = field(default_factory=list)
