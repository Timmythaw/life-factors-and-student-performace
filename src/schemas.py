"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    absences: int = Field(..., description="Number of school absences")
    studytime: float = Field(..., description="Weekly study time (engineered)")
    freetime: float = Field(..., description="Free time after school (engineered)")
    health: float = Field(..., description="Current health status (engineered)")
    goout: float = Field(..., description="Going out with friends (engineered)")
    Dalc: float = Field(..., description="Workday alcohol consumption (engineered)")
    Walc: float = Field(..., description="Weekend alcohol consumption (engineered)")
    higher_yes: bool = Field(..., description="Wants to take higher education")
    schoolsup_yes: bool = Field(..., description="Extra educational support")
    romantic_yes: bool = Field(..., description="In a romantic relationship")
    paid_yes: bool = Field(..., description="Extra paid classes")
    guardian_mother: bool = Field(..., description="Guardian is mother")
    guardian_other: bool = Field(..., description="Guardian is other")
    Mjob_health: bool = Field(..., description="Mother's job is health")
    Mjob_other: bool = Field(..., description="Mother's job is other")
    Mjob_services: bool = Field(..., description="Mother's job is services")
    Mjob_teacher: bool = Field(..., description="Mother's job is teacher")
    reason_home: bool = Field(..., description="Reason for school is home")
    reason_other: bool = Field(..., description="Reason for school is other")
    reason_reputation: bool = Field(..., description="Reason for school is reputation")
    internet_yes: bool = Field(..., description="Internet access at home")
    sex_M: bool = Field(..., description="Sex is male")
    subject_por: bool = Field(..., description="Subject is Portuguese")
    alc_total: float = Field(..., description="Total alcohol consumption (engineered)")
    study_absence_ratio: float = Field(
        ..., description="Study/absence ratio (engineered)"
    )
    alc_goout: float = Field(..., description="Alcohol/goout interaction (engineered)")
    absences_bin_low: bool = Field(..., description="Absences binned: low")
    absences_bin_medium: bool = Field(..., description="Absences binned: medium")
    absences_bin_high: bool = Field(..., description="Absences binned: high")
    goout_binned_medium: bool = Field(..., description="Goout binned: medium")
    goout_binned_high: bool = Field(..., description="Goout binned: high")
    alc_level_moderate: bool = Field(..., description="Alcohol level: moderate")
    alc_level_high: bool = Field(..., description="Alcohol level: high")

    model_config = {
        "json_schema_extra": {
            "example": {
                "absences": 6,
                "studytime": 2.0,
                "freetime": 3.0,
                "health": 3.0,
                "goout": 4.0,
                "Dalc": 1.0,
                "Walc": 1.0,
                "higher_yes": True,
                "schoolsup_yes": False,
                "romantic_yes": False,
                "paid_yes": False,
                "guardian_mother": True,
                "guardian_other": False,
                "Mjob_health": False,
                "Mjob_other": False,
                "Mjob_services": True,
                "Mjob_teacher": False,
                "reason_home": False,
                "reason_other": False,
                "reason_reputation": True,
                "internet_yes": True,
                "sex_M": False,
                "subject_por": False,
                "alc_total": 2.0,
                "study_absence_ratio": 0.33,
                "alc_goout": 0.5,
                "absences_bin_low": False,
                "absences_bin_medium": True,
                "absences_bin_high": False,
                "goout_binned_medium": True,
                "goout_binned_high": False,
                "alc_level_moderate": True,
                "alc_level_high": False,
            }
        }
    }


class PredictionOutput(BaseModel):
    """Output schema for prediction results."""

    at_risk: bool = Field(..., description="Whether student is at risk")
    risk_score: float = Field(
        ..., ge=0, le=1, description="Probability of being at risk"
    )
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    predicted_grade_range: str = Field(..., description="Expected grade range")
    recommendations: list[str] = Field(..., description="Personalized recommendations")


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""

    students: list[StudentFeatures]


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""

    predictions: list[PredictionOutput]
    summary: dict = Field(..., description="Summary statistics")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
