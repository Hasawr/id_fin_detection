"""Fixed MRZ field positions for Azerbaijani ID cards.

These ICAO layouts do not change between cards of the same generation.
Indexes are 0-based Python slices on a normalized MRZ line.

Older card (TD2) — 2 lines × 36 characters
------------------------------------------
Line 2:
  [0:9]   document number / card serial (often 8 digits + ``<``)
  [9]     document-number check digit
  [10:13] nationality (``AZE``)
  [13:19] date of birth + check
  [20]    sex
  [21:28] expiry + check
  [28:35] optional data = personal FIN (7 chars)
  [35]    composite check digit

New card (TD1) — 3 lines × 30 characters
----------------------------------------
Line 1:
  [0:2]   document code (``I`` / ``IA``)
  [2:5]   issuing state (``AZE``)
  [5:14]  document number / card serial (``AA``/``AB`` + 7 digits)
  [14]    document-number check digit
  [15:29] optional data; FIN starts at [15:22] (7 chars)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSlice:
    """Inclusive-exclusive character window on one MRZ line."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def read(self, line_text: str) -> str:
        if self.end > len(line_text):
            return ""
        return line_text[self.start : self.end]


# --- Older TD2 card (front, 2-line MRZ) ---

TD2_LINE_LENGTH = 36
TD2_SERIAL = FieldSlice(0, 9)
TD2_SERIAL_CHECK = FieldSlice(9, 10)
TD2_NATIONALITY = FieldSlice(10, 13)
TD2_FIN = FieldSlice(28, 35)
TD2_COMPOSITE_CHECK = FieldSlice(35, 36)

# OCR may shift AZE by ±1; FIN stays 18 chars after nationality start.
TD2_FIN_AFTER_NATIONALITY = 18
TD2_NATIONALITY_ALLOWED = frozenset({9, 10, 11})

# --- New TD1 card (back, 3-line MRZ) ---

TD1_LINE_LENGTH = 30
TD1_ISSUING_STATE = FieldSlice(2, 5)
TD1_SERIAL = FieldSlice(5, 14)
TD1_SERIAL_CHECK = FieldSlice(14, 15)
TD1_FIN = FieldSlice(15, 22)
TD1_OPTIONAL_END = 29

# Serial search window when OCR drops/inserts a character near the header.
TD1_SERIAL_SEARCH_STARTS = (5, 6, 4)
