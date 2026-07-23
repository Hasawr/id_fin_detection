from pathlib import Path


class PassportService:
    async def process(self, *, image_path: Path) -> dict[str, object]:
        raise NotImplementedError("Passport OCR is not implemented yet.")
