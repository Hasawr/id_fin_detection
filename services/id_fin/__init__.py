from dataclasses import dataclass, field


@dataclass
class MRZResult:
    fin: str | None
    confidence: float
    line1: str
    line2: str
    line3: str
    checksum_valid: bool
    method: str
    card_type: str
    card_serial_number: str | None


@dataclass
class FINDetectionOutput:
    fin: str | None
    confidence: float
    mrz_result: MRZResult | None = None
    notes: list[str] = field(default_factory=list)
