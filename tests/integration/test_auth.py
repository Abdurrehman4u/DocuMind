import uuid
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_auth_flow_integration():
    unique_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "TestPassword123!"

    # 1. Register User
    register_res = client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": password},
    )
    assert register_res.status_code == 201
    reg_data = register_res.json()
    assert "access_token" in reg_data
    assert "refresh_token" in reg_data
    access_token = reg_data["access_token"]
    refresh_token = reg_data["refresh_token"]

    # 2. Duplicate registration fails
    dup_res = client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": password},
    )
    assert dup_res.status_code == 400

    # 3. Access Protected Route /me with token
    me_res = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == unique_email

    # 4. Access Protected Route without token fails
    me_unauth_res = client.get("/api/v1/auth/me")
    assert me_unauth_res.status_code == 401

    # 5. Login User
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": password},
    )
    assert login_res.status_code == 200
    login_data = login_res.json()
    assert "access_token" in login_data

    # 6. Invalid password fails
    bad_login = client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": "WrongPassword"},
    )
    assert bad_login.status_code == 401

    # 7. Refresh Token
    refresh_res = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_res.status_code == 200
    refresh_data = refresh_res.json()
    assert "access_token" in refresh_data
    assert "refresh_token" in refresh_data
