from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8000
    host: str = "0.0.0.0"
    logger: str = "INFO"
    redis_url: str | None = None
    model_path: str = "/Users/sujannandikolsunilkumar/bci project/bci-project/models/lda_pipeline.joblib"
    max_upload_size:  int = 100 * 1024 * 1024  # 100MB

    class Config: 
        env_file = ".env"

settings = Settings()
