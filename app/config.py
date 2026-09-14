from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENV: str = "development"
    SECRET_KEY: str = "dev-secret-key-change-in-prod-1234567890"
    REFRESH_SECRET_KEY: str = "dev-refresh-secret-key-change-in-prod-1234567890"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    DATABASE_URL: str = "sqlite:///./documind_dev.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    @property
    def CELERY_BROKER_URL(self) -> str:
        return self.REDIS_URL

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return self.REDIS_URL

    CHROMA_PERSIST_DIR: str = "./chroma_data"

    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    UPLOAD_DIR: str = "./uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("SECRET_KEY", "REFRESH_SECRET_KEY", mode="after")
    @classmethod
    def validate_production_secrets(cls, v: str, info) -> str:
        dev_defaults = {
            "dev-secret-key-change-in-prod-1234567890",
            "dev-refresh-secret-key-change-in-prod-1234567890",
            "change-this-to-a-super-secret-key-in-production",
            "change-this-to-another-super-secret-key",
            "",
        }
        env = info.data.get("ENV", "development")
        if env != "development" and v in dev_defaults:
            raise ValueError(
                f"Secret '{info.field_name}' must be explicitly set and cannot use a development default when ENV is '{env}'."
            )
        return v


settings = Settings()
