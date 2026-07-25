import hashlib
import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from shared.config import Settings, get_settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _keys_match(supplied_key: str, expected_key: str) -> bool:
    supplied_digest = hashlib.sha256(
        supplied_key.encode("utf-8")
    ).digest()
    expected_digest = hashlib.sha256(
        expected_key.encode("utf-8")
    ).digest()
    return hmac.compare_digest(
        supplied_digest,
        expected_digest,
    )


def require_api_key(
    supplied_key: Annotated[str | None, Security(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    if not settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    is_valid_key = supplied_key is not None and any(
        _keys_match(supplied_key, expected_key)
        for expected_key in settings.api_keys
    )
    if not is_valid_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return supplied_key
