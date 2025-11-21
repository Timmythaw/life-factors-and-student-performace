"""
Tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app
import json
import os
from src.model import model_service
model_service.load_model()

client = TestClient(app)


with open(os.path.join(os.path.dirname(__file__), '../models/combined_feature_names.json'), 'r') as f:
    FEATURE_NAMES = json.load(f)

def build_student_data():
    # Set reasonable defaults for each feature
    defaults = {
        'absences': 6,
        'studytime': 2.0,
        'freetime': 3.0,
        'health': 3.0,
        'goout': 4.0,
        'Dalc': 1.0,
        'Walc': 1.0,
        'alc_total': 2.0,
        'study_absence_ratio': 0.33,
        'alc_goout': 0.5,
    }
    # All bool features default to False unless set above
    data = {}
    # Track groups for mutually exclusive features
    groups = {
        'Mjob_': [],
        'reason_': [],
        'guardian_': [],
        'absences_bin_': [],
        'goout_binned_': [],
        'alc_level_': [],
    }
    for name in FEATURE_NAMES:
        if name in defaults:
            data[name] = defaults[name]
        elif name.startswith(tuple(groups.keys())):
            for prefix in groups:
                if name.startswith(prefix):
                    groups[prefix].append(name)
            data[name] = False
        elif name.startswith(('higher', 'schoolsup', 'romantic', 'paid', 'internet', 'sex_', 'subject_')):
            data[name] = False
        else:
            data[name] = 0.0
    # For each group, set exactly one feature to True (the first in each group)
    for group in groups.values():
        if group:
            for i, name in enumerate(group):
                data[name] = (i == 0)
    # Set some bools to True for variety, but only if not in a mutually exclusive group
    for k in ['higher_yes']:
        if k in data:
            data[k] = True
    return data


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_single_prediction():
    """Test single student prediction."""
    student_data = build_student_data()
    print("Payload sent to /api/v1/predict:", student_data)
    response = client.post("/api/v1/predict", json=student_data)
    assert response.status_code == 200
    data = response.json()
    assert "at_risk" in data
    assert "risk_score" in data
    assert "confidence" in data
    assert "predicted_grade_range" in data
    assert "recommendations" in data
    assert isinstance(data["at_risk"], bool)
    assert 0 <= data["risk_score"] <= 1


def test_batch_prediction():
    """Test batch prediction."""
    student_data = build_student_data()
    
    batch_data = {
        "students": [student_data, student_data]
    }
    
    response = client.post("/api/v1/predict/batch", json=batch_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "summary" in data
    assert len(data["predictions"]) == 2
    assert "total_students" in data["summary"]


def test_invalid_input():
    """Test prediction with invalid input."""
    invalid_data = {
        "absences": "not_an_int",
        "studytime": None
    }
    
    response = client.post("/api/v1/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity


def test_model_info():
    """Test model info endpoint."""
    response = client.get("/api/v1/model/info")
    # May be 200 or 503 depending on if model is loaded
    assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])