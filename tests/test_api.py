"""
Tests for the Student Risk Classifier API.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

# Sample valid student data matching your schema
SAMPLE_STUDENT = {
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

# High-risk student profile
HIGH_RISK_STUDENT = {
    "absences": 25,
    "studytime": 1.0,
    "freetime": 5.0,
    "health": 2.0,
    "goout": 5.0,
    "Dalc": 4.0,
    "Walc": 5.0,
    "higher_yes": False,
    "schoolsup_yes": False,
    "romantic_yes": True,
    "paid_yes": False,
    "guardian_mother": False,
    "guardian_other": True,
    "Mjob_health": False,
    "Mjob_other": True,
    "Mjob_services": False,
    "Mjob_teacher": False,
    "reason_home": False,
    "reason_other": True,
    "reason_reputation": False,
    "internet_yes": False,
    "sex_M": True,
    "subject_por": False,
    "alc_total": 9.0,
    "study_absence_ratio": 0.04,
    "alc_goout": 4.5,
    "absences_bin_low": False,
    "absences_bin_medium": False,
    "absences_bin_high": True,
    "goout_binned_medium": False,
    "goout_binned_high": True,
    "alc_level_moderate": False,
    "alc_level_high": True,
}


class TestRootEndpoints:
    """Tests for root and utility endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns expected structure."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert data["status"] in ["healthy", "unhealthy"]


class TestPredictions:
    """Tests for prediction endpoints."""
    
    def test_single_prediction_success(self):
        """Test successful single student prediction."""
        response = client.post("/api/v1/predict", json=SAMPLE_STUDENT)
        assert response.status_code == 200
        
        data = response.json()
        assert "at_risk" in data
        assert "risk_score" in data
        assert "confidence" in data
        assert "predicted_grade_range" in data
        assert "recommendations" in data
        
        # Validate types
        assert isinstance(data["at_risk"], bool)
        assert isinstance(data["risk_score"], float)
        assert 0 <= data["risk_score"] <= 1
        assert data["confidence"] in ["low", "medium", "high"]
        assert isinstance(data["recommendations"], list)

    def test_high_risk_student_prediction(self):
        """Test prediction for high-risk student profile."""
        response = client.post("/api/v1/predict", json=HIGH_RISK_STUDENT)
        assert response.status_code == 200
        
        data = response.json()
        # High-risk student should have higher risk score
        assert data["risk_score"] > 0.3  # Expect elevated risk

    def test_batch_prediction_success(self):
        """Test successful batch prediction."""
        batch_data = {
            "students": [SAMPLE_STUDENT, HIGH_RISK_STUDENT]
        }
        
        response = client.post("/api/v1/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 2
        
        # Check summary statistics
        summary = data["summary"]
        assert "total_students" in summary
        assert "at_risk_count" in summary
        assert "average_risk_score" in summary
        assert summary["total_students"] == 2

    def test_prediction_with_missing_field(self):
        """Test prediction fails with missing required field."""
        incomplete_data = {k: v for k, v in SAMPLE_STUDENT.items() if k != "absences"}
        
        response = client.post("/api/v1/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error

    def test_prediction_with_invalid_type(self):
        """Test prediction fails with invalid data type."""
        invalid_data = SAMPLE_STUDENT.copy()
        invalid_data["absences"] = "not_a_number"
        
        response = client.post("/api/v1/predict", json=invalid_data)
        assert response.status_code == 422

    def test_prediction_recommendations_for_at_risk(self):
        """Test that at-risk students get appropriate recommendations."""
        response = client.post("/api/v1/predict", json=HIGH_RISK_STUDENT)
        assert response.status_code == 200
        
        data = response.json()
        recommendations = data["recommendations"]
        
        # Should have multiple recommendations for high-risk student
        assert len(recommendations) >= 1


class TestModelInfo:
    """Tests for model information endpoint."""
    
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")
        
        # May be 200 or 503 depending on model loading
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "model_type" in data
            assert "num_features" in data
            assert "has_scaler" in data


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_absences(self):
        """Test with zero absences."""
        student = SAMPLE_STUDENT.copy()
        student["absences"] = 0
        student["absences_bin_low"] = True
        student["absences_bin_medium"] = False
        student["study_absence_ratio"] = float('inf') if student["studytime"] > 0 else 0
        # Handle inf
        student["study_absence_ratio"] = 999.0  # Cap at high value
        
        response = client.post("/api/v1/predict", json=student)
        assert response.status_code == 200

    def test_extreme_values(self):
        """Test with extreme but valid values."""
        student = SAMPLE_STUDENT.copy()
        student["absences"] = 100
        student["goout"] = 5.0
        student["Dalc"] = 5.0
        student["Walc"] = 5.0
        student["alc_total"] = 10.0
        
        response = client.post("/api/v1/predict", json=student)
        assert response.status_code == 200

    def test_empty_batch(self):
        """Test batch prediction with empty list fails validation."""
        response = client.post("/api/v1/predict/batch", json={"students": []})
        # Should fail validation - empty list not allowed
        assert response.status_code in [422, 500]

    def test_large_batch(self):
        """Test batch prediction with multiple students."""
        students = [SAMPLE_STUDENT.copy() for _ in range(10)]
        
        response = client.post("/api/v1/predict/batch", json={"students": students})
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["predictions"]) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])