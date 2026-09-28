from datetime import timedelta
import jwt
import pytest
from app.core import security
from app.config import settings


def test_password_hashing_and_verification():
    password = "MySecurePassword123!"
    hashed = security.get_password_hash(password)
    assert hashed != password
    assert security.verify_password(password, hashed) is True
    assert security.verify_password("WrongPassword!", hashed) is False


def test_create_and_decode_access_token():
    user_id = "test-user-123"
    token = security.create_access_token(data={"sub": user_id})
    payload = security.decode_token(token, settings.SECRET_KEY)
    assert payload.get("sub") == user_id
    assert payload.get("type") == "access"


def test_create_and_decode_refresh_token():
    user_id = "test-user-456"
    token = security.create_refresh_token(data={"sub": user_id})
    payload = security.decode_token(token, settings.REFRESH_SECRET_KEY)
    assert payload.get("sub") == user_id
    assert payload.get("type") == "refresh"


def test_expired_token_raises_exception():
    token = security.create_access_token(
        data={"sub": "expired-user"},
        expires_delta=timedelta(seconds=-1),
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        security.decode_token(token, settings.SECRET_KEY)


def test_invalid_token_signature_raises_exception():
    token = security.create_access_token(data={"sub": "user-wrong-key"})
    with pytest.raises(jwt.InvalidSignatureError):
        security.decode_token(token, "wrong-secret-key-32-bytes-minimum-length!!")

