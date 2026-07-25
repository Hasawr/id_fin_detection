from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class ServiceResponse(BaseModel):
    service: str
    version: str = "v1"
    data: dict[str, Any] | None = None
    error: ErrorDetail | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    services: list[str] = Field(default_factory=list)
    ocr_max_concurrency: int | None = Field(
        None,
        description="Max ID images that can run on the GPU at once.",
        examples=[2],
    )
    ocr_available_workers: int | None = Field(
        None,
        description="Idle OCR workers ready to accept work right now.",
        examples=[2],
    )


class MRZDetails(BaseModel):
    fin: str | None = Field(
        None,
        description="Personal FIN extracted from the MRZ when available.",
        examples=["7ABC123"],
    )
    confidence: float = Field(
        0.0,
        description="MRZ parse confidence from 0 to 1.",
        examples=[0.96],
    )
    line1: str = Field("", description="Reconstructed MRZ line 1.")
    line2: str = Field("", description="Reconstructed MRZ line 2.")
    line3: str = Field("", description="Reconstructed MRZ line 3 for TD1 cards.")
    checksum_valid: bool = Field(
        False,
        description="Whether the document-number checksum validated.",
    )
    method: str = Field(
        "",
        description="Detection method used, for example td1_direct or td2_direct.",
        examples=["td1_direct"],
    )
    card_type: Literal["new_card", "older_card", "unknown"] = Field(
        "unknown",
        description="new_card (TD1, 3-line MRZ), older_card (TD2, 2-line MRZ), or unknown.",
    )
    card_serial_number: str | None = Field(
        None,
        description=(
            "Document serial from the MRZ. New TD1 cards use AA or AB "
            "followed by seven digits. Older TD2 cards use the numeric "
            "document number (typically 7-9 digits). Null when the value "
            "cannot be validated."
        ),
        examples=["AA1234567", "19205792"],
    )


class IdFinData(BaseModel):
    fin: str | None = Field(
        None,
        description="Detected personal FIN, or null when not found.",
        examples=["7ABC123"],
    )
    confidence: float = Field(
        ...,
        description="Overall detection confidence from 0 to 1.",
        examples=[0.96],
    )
    mrz_details: MRZDetails | None = Field(
        None,
        description="Parsed MRZ details when MRZ lines were recovered.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Optional processing notes or failure hints.",
    )


class IdFinResponse(BaseModel):
    service: str = Field("id-fin", examples=["id-fin"])
    version: str = "v1"
    data: IdFinData | None = None
    error: ErrorDetail | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "service": "id-fin",
                    "version": "v1",
                    "data": {
                        "fin": "7ABC123",
                        "confidence": 0.96,
                        "mrz_details": {
                            "fin": "7ABC123",
                            "confidence": 0.96,
                            "line1": "I<AZEAA1234567<<<<<<<<<<<<<<<",
                            "line2": "9001011M3001011AZE<<<<<<<<<<<7",
                            "line3": "DOE<<JOHN<<<<<<<<<<<<<<<<<<<<<",
                            "checksum_valid": True,
                            "method": "td1_direct",
                            "card_type": "new_card",
                            "card_serial_number": "AA1234567",
                        },
                        "notes": [],
                    },
                    "error": None,
                }
            ]
        }
    }


class IdFinBatchItem(IdFinData):
    index: int = Field(..., description="Zero-based index in the batch.", examples=[0])
    file_name: str = Field(..., description="Original upload file name.", examples=["back.jpg"])


class IdFinBatchData(BaseModel):
    count: int = Field(..., description="Number of results in this batch.", examples=[2])
    results: list[IdFinBatchItem]


class IdFinBatchResponse(BaseModel):
    service: str = Field("id-fin", examples=["id-fin"])
    version: str = "v1"
    data: IdFinBatchData | None = None
    error: ErrorDetail | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "service": "id-fin",
                    "version": "v1",
                    "data": {
                        "count": 1,
                        "results": [
                            {
                                "index": 0,
                                "file_name": "back.jpg",
                                "fin": "7ABC123",
                                "confidence": 0.96,
                                "mrz_details": {
                                    "fin": "7ABC123",
                                    "confidence": 0.96,
                                    "line1": "I<AZEAA1234567<<<<<<<<<<<<<<<",
                                    "line2": "9001011M3001011AZE<<<<<<<<<<<7",
                                    "line3": "DOE<<JOHN<<<<<<<<<<<<<<<<<<<<<",
                                    "checksum_valid": True,
                                    "method": "td1_direct",
                                    "card_type": "new_card",
                                    "card_serial_number": "AA1234567",
                                },
                                "notes": [],
                            }
                        ],
                    },
                    "error": None,
                }
            ]
        }
    }
