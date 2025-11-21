"""
Debug script to verify model, scaler, and feature alignment.
Run this to diagnose the feature mismatch issue.
"""
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

# Paths - adjust if needed
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "combined_xgb_final_model.joblib"
SCALER_PATH = MODELS_DIR / "combined_scaler.joblib"
FEATURES_PATH = MODELS_DIR / "combined_feature_names.json"


def main():
    print("=" * 60)
    print("MODEL DEBUGGING SCRIPT")
    print("=" * 60)
    
    # 1. Load and inspect feature names
    print("\n📋 FEATURE NAMES FROM JSON:")
    print("-" * 40)
    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)
    print(f"Number of features: {len(feature_names)}")
    print(f"Features: {feature_names}")
    
    # 2. Load and inspect scaler
    print("\n⚖️ SCALER INSPECTION:")
    print("-" * 40)
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler type: {type(scaler).__name__}")
    print(f"Number of features scaler expects: {scaler.n_features_in_}")
    
    if hasattr(scaler, 'feature_names_in_'):
        print(f"Scaler feature names: {list(scaler.feature_names_in_)}")
    else:
        print("Scaler does NOT have feature_names_in_ attribute")
        print("(This means it was fitted on a numpy array, not a DataFrame with column names)")
    
    print(f"Scaler mean_ shape: {scaler.mean_.shape}")
    print(f"Scaler scale_ shape: {scaler.scale_.shape}")
    
    # 3. Load and inspect model
    print("\n🤖 MODEL INSPECTION:")
    print("-" * 40)
    model = joblib.load(MODEL_PATH)
    print(f"Model type: {type(model).__name__}")
    
    if hasattr(model, 'n_features_in_'):
        print(f"Number of features model expects: {model.n_features_in_}")
    
    if hasattr(model, 'feature_names_in_'):
        print(f"Model feature names: {list(model.feature_names_in_)}")
    else:
        print("Model does NOT have feature_names_in_ attribute")
    
    # 4. Check alignment
    print("\n🔍 ALIGNMENT CHECK:")
    print("-" * 40)
    
    model_n_features = getattr(model, 'n_features_in_', None)
    scaler_n_features = scaler.n_features_in_
    json_n_features = len(feature_names)
    
    print(f"JSON feature count: {json_n_features}")
    print(f"Scaler feature count: {scaler_n_features}")
    print(f"Model feature count: {model_n_features}")
    
    if model_n_features == scaler_n_features == json_n_features:
        print("✅ All feature counts MATCH!")
    else:
        print("❌ Feature count MISMATCH!")
        print("   This is likely the cause of your error.")
    
    # 5. Test with sample data
    print("\n🧪 TEST PREDICTION:")
    print("-" * 40)
    
    # Create sample data matching the JSON feature names
    sample_data = {
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
    
    # Convert booleans to int (0/1) for the model
    df = pd.DataFrame([sample_data])
    
    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    print(f"Sample DataFrame columns: {list(df.columns)}")
    print(f"Sample DataFrame shape: {df.shape}")
    
    # Reorder to match feature names from JSON
    df = df[feature_names]
    print(f"Reordered DataFrame columns: {list(df.columns)}")
    
    # Try scaling
    print("\n⚖️ Testing scaler transform...")
    try:
        if scaler_n_features == len(feature_names):
            # Scaler expects all features
            scaled_data = scaler.transform(df)
            print(f"✅ Scaler transform successful! Shape: {scaled_data.shape}")
        else:
            print(f"⚠️ Scaler expects {scaler_n_features} features but we have {len(feature_names)}")
    except Exception as e:
        print(f"❌ Scaler transform FAILED: {e}")
    
    # Try prediction
    print("\n🎯 Testing model predict...")
    try:
        # Try without scaling first
        pred = model.predict(df)
        proba = model.predict_proba(df)
        print(f"✅ Model prediction successful!")
        print(f"   Prediction: {pred[0]}")
        print(f"   Probabilities: {proba[0]}")
    except Exception as e:
        print(f"❌ Model prediction FAILED: {e}")
    
    # 6. Recommendations
    print("\n" + "=" * 60)
    print("📝 RECOMMENDATIONS:")
    print("=" * 60)
    
    if scaler_n_features != len(feature_names):
        print("""
The scaler was fitted on a DIFFERENT number of features than what's 
in your feature_names.json file.

SOLUTION: You need to re-save your scaler with the correct features.
In your training notebook, after fitting the scaler, save it like this:

    # Make sure to fit scaler on the same columns as your model
    scaler = StandardScaler()
    scaler.fit(X_train[feature_names])  # Use the exact feature list
    joblib.dump(scaler, 'combined_scaler.joblib')
        """)
    
    if not hasattr(model, 'feature_names_in_'):
        print("""
Your model doesn't have feature_names_in_ stored. This is OK, but means
the model was trained on a numpy array instead of a DataFrame.

The prediction should still work if the column ORDER matches exactly.
        """)
    
    print("\nDone!")


if __name__ == "__main__":
    main()