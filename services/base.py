from pathlib import Path
from typing import Protocol


class OCRService(Protocol):
    async def process(
        self,
        *,
        image_path: Path,
    ) -> dict[str, object]:
        """Process service inputs and return JSON-serializable data."""
        ...
