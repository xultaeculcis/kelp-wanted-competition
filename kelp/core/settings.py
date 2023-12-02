from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from kelp import consts


class Settings(BaseSettings):
    """Represents Application Settings with nested configuration sections"""

    environment: str = "local"

    model_config = SettingsConfigDict(
        env_file=consts.directories.ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


current_settings = Settings()
