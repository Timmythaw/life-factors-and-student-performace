"""
Model loading, preprocessing, and prediction service.
"""

import joblib
import json
import pandas as pd
from typing import List
import logging

from .config import settings
from .schemas import StudentFeatures, PredictionOutput

logger = logging.getLogger(__name__)


class ModelService:
    """Service for handling model predictions."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False

    def load_model(self) -> bool:
        """Load the trained model, scaler, and feature names."""
        try:
            # Load model
            model_path = settings.MODEL_CONFIG.model_path
            if not model_path.exists():
                logger.error(f"Model file not found at {model_path}")
                return False

            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")

            # Load scaler
            scaler_path = settings.MODEL_CONFIG.scaler_path
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning("No scaler found, will proceed without scaling")

            # Load feature names
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
        """Convert engineered student input to model-ready DataFrame."""
        # Use model_dump for Pydantic v2
        data = student.model_dump()
        df = pd.DataFrame([data])
        # Ensure all expected features are present
        if self.feature_names:
            # Fill missing features with defaults
            missing = []
            for feature in self.feature_names:
                if feature not in df.columns:
                    missing.append(feature)
                    # Use False for booleans, 0.0 for numerics
                    if feature in settings.MODEL_CONFIG.categorical_features:
                        df[feature] = False
                    elif feature in settings.MODEL_CONFIG.numerical_features:
                        df[feature] = 0.0
                    else:
                        df[feature] = 0
            df = df[self.feature_names]
            if missing:
                logger.error(f"Loaded feature names: {self.feature_names}")
                logger.error(f"DataFrame columns: {list(df.columns)}")
                logger.error(f"Missing features filled: {missing}")
        # Scale numeric features if scaler exists
        if self.scaler:
            numeric_cols = [
                col
                for col in settings.MODEL_CONFIG.numerical_features
                if col in df.columns
            ]
            if numeric_cols:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        logger.error(f"Preprocessed DataFrame columns: {list(df.columns)}")
        return df

    def predict(self, student: StudentFeatures) -> PredictionOutput:
        """Make prediction for a single student."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        # Preprocess
        X = self.preprocess_input(student)
        # Predict
        if not hasattr(self.model, "predict") or not hasattr(
            self.model, "predict_proba"
        ):
            raise RuntimeError(
                "Loaded model does not support predict or predict_proba methods."
            )
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        # Extract risk probability (assuming binary: 0=not at risk, 1=at risk)
        risk_score = float(probabilities[1])
        at_risk = bool(prediction == 1)
        # Determine confidence level
        confidence = self._get_confidence_level(risk_score)
        # Predict grade range
        grade_range = self._predict_grade_range(risk_score)
        # Generate recommendations
        recommendations = self._generate_recommendations(student, at_risk, risk_score)
        return PredictionOutput(
            at_risk=at_risk,
            risk_score=risk_score,
            confidence=confidence,
            predicted_grade_range=grade_range,
            recommendations=recommendations,
        )

    def predict_batch(self, students: List[StudentFeatures]) -> List[PredictionOutput]:
        """Make predictions for multiple students."""
        return [self.predict(student) for student in students]

    def _get_confidence_level(self, risk_score: float) -> str:
        """Determine confidence level based on probability."""
        if risk_score < 0.3 or risk_score > 0.7:
            return "high"
        elif 0.4 <= risk_score <= 0.6:
            return "low"
        else:
            return "medium"

    def _predict_grade_range(self, risk_score: float) -> str:
        """Predict grade range based on risk score."""
        if risk_score < 0.2:
            return "15-20 (Excellent)"
        elif risk_score < 0.4:
            return "12-14 (Good)"
        elif risk_score < 0.6:
            return "10-11 (Satisfactory)"
        elif risk_score < 0.8:
            return "8-9 (At Risk)"
        else:
            return "0-7 (High Risk)"

    def _generate_recommendations(
        self, student: StudentFeatures, at_risk: bool, risk_score: float
    ) -> List[str]:
        """Generate personalized recommendations based on student profile."""
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


# Global model service instance
model_service = ModelService()
