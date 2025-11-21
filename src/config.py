"""
Configuration management for the student academic risk prediction system.
"""

from pathlib import Path
from typing import List
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Paths(BaseModel):
    """Project Paths Configurations."""

    ROOT: Path = Path(__file__).parent.parent
    DATA: Path = ROOT / "data"
    RAW_DATA: Path = DATA / "raw"
    PROCESSED_DATA: Path = DATA / "processed"
    MODELS: Path = ROOT / "models"
    REPORTS: Path = ROOT / "reports"


class ModelConfig(BaseModel):
    """Model Configurations."""

    model_name: str = "combined_xgb_final_model"
    model_path: Path = Paths().MODELS / "combined_xgb_final_model.joblib"
    scaler_path: Path = Paths().MODELS / "combined_scaler.joblib"
    feature_names_path: Path = Paths().MODELS / "combined_feature_names.json"

    categorical_features: List[str] = [
        "higher_yes",
        "schoolsup_yes",
        "romantic_yes",
        "paid_yes",
        "guardian_mother",
        "guardian_other",
        "Mjob_health",
        "Mjob_other",
        "Mjob_services",
        "Mjob_teacher",
        "reason_home",
        "reason_other",
        "reason_reputation",
        "internet_yes",
        "sex_M",
        "subject_por",
        "absences_bin_low",
        "absences_bin_medium",
        "absences_bin_high",
        "goout_binned_medium",
        "goout_binned_high",
        "alc_level_moderate",
        "alc_level_high",
    ]

    numerical_features: List[str] = [
        "absences",
        "studytime",
        "freetime",
        "health",
        "goout",
        "Dalc",
        "Walc",
        "alc_total",
        "study_absence_ratio",
        "alc_goout",
    ]

    at_risk_threshold: float = 10.0


class Settings(BaseSettings):
    """Application settings."""

    APP_NAME: str = "Student Risk Classifier API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Server config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    MODEL_CONFIG: ModelConfig = ModelConfig()
    PATHS: Paths = Paths()

    model_config = {
        "env_file": ".env",
        "case_sensitive": True
    }


settings = Settings()
