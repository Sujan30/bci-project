from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    port: int = Field(default=8000, validation_alias="PORT")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    logger: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    redis_url: str | None = Field(default=None, validation_alias="REDIS_URL")
    model_path: str = Field(default="models/lda_pipeline.joblib", validation_alias="MODEL_PATH")
    max_upload_size: int = Field(default=100 * 1024 * 1024, validation_alias="MAX_UPLOAD_SIZE")

settings = Settings()
