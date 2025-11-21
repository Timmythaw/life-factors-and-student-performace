"""
Model loading, preprocessing, and prediction service.
FIXED version - handles boolean to int conversion.
"""

import joblib
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from xgboost import XGBClassifier
import logging

from .config import settings
from .schemas import StudentFeatures, PredictionOutput

logger = logging.getLogger(__name__)


class ModelService:
    """Service for handling model predictions."""

    def __init__(self):
        self.model: Optional[XGBClassifier] = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False

    def load_model(self) -> bool:
        """Load the trained model, scaler, and feature names."""
        try:
            model_path = settings.MODEL_CONFIG.model_path
            if not model_path.exists():
                logger.error(f"Model file not found at {model_path}")
                return False

            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")

            scaler_path = settings.MODEL_CONFIG.scaler_path
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")

            feature_names_path = settings.MODEL_CONFIG.feature_names_path
            if feature_names_path.exists():
                with open(feature_names_path, "r") as f:
                    self.feature_names = json.load(f)
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")

            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def preprocess_input(self, student: StudentFeatures) -> pd.DataFrame:
        """Convert student input to model-ready DataFrame."""
        data = student.model_dump()
        df = pd.DataFrame([data])
        
        # CRITICAL FIX: Convert boolean columns to int (0/1)
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)
        
        # Reorder columns to match training order
        if self.feature_names:
            df = df[self.feature_names]
        
        # Apply scaler (fitted on all 33 features)
        if self.scaler is not None:
            df = pd.DataFrame(self.scaler.transform(df), columns=self.feature_names)
        
        return df

    def predict(self, student: StudentFeatures) -> PredictionOutput:
        """Make prediction for a single student."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        
        X = self.preprocess_input(student)
        prediction = self.model.predict(X)[0] # type: ignore[reportOptionalMemberAccess]
        probabilities = self.model.predict_proba(X)[0] # type: ignore[reportOptionalMemberAccess]
        
        risk_score = float(probabilities[1])
        at_risk = bool(prediction == 1)
        confidence = self._get_confidence_level(risk_score)
        grade_range = self._predict_grade_range(risk_score)
        recommendations = self._generate_recommendations(student, at_risk, risk_score)
        
        return PredictionOutput(
            at_risk=at_risk,
            risk_score=risk_score,
            confidence=confidence,
            predicted_grade_range=grade_range,
            recommendations=recommendations,
        )

    def predict_batch(self, students: List[StudentFeatures]) -> List[PredictionOutput]:
        return [self.predict(s) for s in students]

    def _get_confidence_level(self, risk_score: float) -> str:
        if risk_score < 0.3 or risk_score > 0.7:
            return "high"
        elif 0.4 <= risk_score <= 0.6:
            return "low"
        return "medium"

    def _predict_grade_range(self, risk_score: float) -> str:
        if risk_score < 0.2:
            return "15-20 (Excellent)"
        elif risk_score < 0.4:
            return "12-14 (Good)"
        elif risk_score < 0.6:
            return "10-11 (Satisfactory)"
        elif risk_score < 0.8:
            return "8-9 (At Risk)"
        return "0-7 (High Risk)"

    def _generate_recommendations(
        self, student: StudentFeatures, at_risk: bool, risk_score: float
    ) -> List[str]:
        recommendations = []

        if at_risk:
            if student.studytime < 2:
                recommendations.append(
                    "📚 Increase weekly study time. Consider studying 2-3 hours daily."
                )
            if student.Dalc > 2 or student.Walc > 2 or student.alc_total > 3:
                recommendations.append(
                    "⚠️ Reduce alcohol consumption as it may negatively impact academic performance."
                )
            if student.absences > 10 or student.absences_bin_high:
                recommendations.append(
                    "🎯 Improve attendance. High absences are strongly correlated with poor performance."
                )
            if student.goout > 3 or student.goout_binned_high:
                recommendations.append(
                    "⏰ Balance social life with academics. Consider reducing going out frequency."
                )
            if not student.schoolsup_yes:
                recommendations.append(
                    "🤝 Seek educational support from school to improve performance."
                )
            if student.freetime > 4 and student.studytime < 3:
                recommendations.append(
                    "📖 Use free time more effectively for studying and homework."
                )
        else:
            recommendations.append(
                "✅ Keep up the good work! Continue your current study habits."
            )
            if student.studytime >= 3:
                recommendations.append("🌟 Your dedication to studying is paying off!")
            if student.absences < 5 or student.absences_bin_low:
                recommendations.append("👏 Excellent attendance record!")
        
        if student.health < 3:
            recommendations.append(
                "🏥 Consider addressing health concerns that may affect academic performance."
            )
        if not student.higher_yes and not at_risk:
            recommendations.append(
                "🎓 Consider higher education opportunities to maximize your potential."
            )
        
        return recommendations if recommendations else ["Continue monitoring progress."]


model_service = ModelService()