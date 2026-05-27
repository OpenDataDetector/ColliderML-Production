"""Backend configuration (env-driven via pydantic-settings)."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All env vars the backend reads. Defaults are dev-friendly."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Database ---
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/colliderml",
        description="Postgres/Supabase DSN",
    )

    # --- Admin ---
    admin_token: str = Field(
        default="dev-admin-token",
        description="Shared secret for /admin/* routes. Set in prod.",
    )

    # --- SFAPI (NERSC) ---
    sfapi_client_id: str = Field(default="", description="NERSC IRIS client ID")
    sfapi_client_secret: str = Field(default="", description="NERSC IRIS PEM secret")
    nersc_project: str = Field(default="", description="NERSC project, e.g. m4958")
    nersc_user: str = Field(default="", description="NERSC service account username")

    # --- Container image (updated when OpenDataDetector/sw#2 lands) ---
    container_image: str = Field(
        default="ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0",
    )
    colliderml_branch: str = Field(default="main")

    # --- HuggingFace upload (service account) ---
    hf_token: str = Field(default="", description="HF service-account token for output uploads")
    hf_dataset_org: str = Field(
        default="CERN",
        description="HF org to create per-request datasets under (e.g. CERN/ColliderML-Service-...)",
    )

    # --- HF benchmark results dataset ---
    hf_results_dataset: str = Field(
        default="CERN/colliderml-benchmark-results",
        description="HF dataset repo where benchmark result JSONs are pushed after scoring",
    )

    # --- Email (SMTP) ---
    smtp_host: str = Field(default="")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_from: str = Field(default="colliderml@noreply.example")

    # --- Compute cost model ---
    # How many node-hours to reserve per event, per channel. See cap.py.
    # Users may override via admin dashboard later.

    # --- Misc ---
    poll_interval_seconds: int = Field(default=60)


@lru_cache
def get_settings() -> Settings:
    return Settings()
