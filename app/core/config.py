# app/core/config.py
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App
    app_name: str = "Viet Emotion Sentiment API"

    # Model
    use_half: bool = True

    # CORS
    allow_origins: List[str] = ["*"]
    allow_methods: List[str] = ["*"]
    allow_headers: List[str] = ["*"]
    allow_credentials: bool = True

    # .env mapping (không phân biệt HOA/thường), bỏ qua biến lạ
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()
    )

    # Cho phép viết CSV trong .env như: ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:5500
    @field_validator("allow_origins", "allow_methods", "allow_headers", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

settings = Settings()
