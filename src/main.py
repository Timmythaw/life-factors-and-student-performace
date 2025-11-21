"""
FastAPI application for student performance prediction.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from .config import settings
from .schemas import (
    StudentFeatures,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse,
)
from .model import model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting application...")
    success = model_service.load_model()
    if not success:
        logger.error("Failed to load model!")
    else:
        logger.info("Model loaded successfully!")

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API for predicting student academic performance based on lifestyle factors",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Student Performance Prediction API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "unhealthy",
        version=settings.APP_VERSION,
        model_loaded=model_service.is_loaded,
    )


@app.post(
    f"{settings.API_PREFIX}/predict",
    response_model=PredictionOutput,
    tags=["Predictions"],
    status_code=status.HTTP_200_OK,
)
async def predict_student_performance(student: StudentFeatures):
    """
    Predict whether a student is at risk of academic underperformance.

    Returns:
    - at_risk: Boolean indicating risk status
    - risk_score: Probability of being at risk (0-1)
    - confidence: Confidence level (low/medium/high)
    - predicted_grade_range: Expected grade range
    - recommendations: Personalized recommendations
    """
    try:
        if not model_service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later.",
            )

        prediction = model_service.predict(student)
        return prediction

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {repr(e)}",
        )


@app.post(
    f"{settings.API_PREFIX}/predict/batch",
    response_model=BatchPredictionOutput,
    tags=["Predictions"],
    status_code=status.HTTP_200_OK,
)
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Make predictions for multiple students at once.

    Maximum 100 students per request.
    """
    try:
        if not model_service.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later.",
            )

        predictions = model_service.predict_batch(batch_input.students)

        # Calculate summary statistics
        at_risk_count = sum(1 for p in predictions if p.at_risk)
        avg_risk_score = sum(p.risk_score for p in predictions) / len(predictions)

        summary = {
            "total_students": len(predictions),
            "at_risk_count": at_risk_count,
            "not_at_risk_count": len(predictions) - at_risk_count,
            "at_risk_percentage": round(at_risk_count / len(predictions) * 100, 2),
            "average_risk_score": round(avg_risk_score, 3),
        }

        return BatchPredictionOutput(predictions=predictions, summary=summary)

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get(f"{settings.API_PREFIX}/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    return {
        "model_name": settings.MODEL_CONFIG.model_name,
        "model_type": str(type(model_service.model).__name__),
        "num_features": (
            len(model_service.feature_names) if model_service.feature_names else None
        ),
        "has_scaler": model_service.scaler is not None,
        "at_risk_threshold": settings.MODEL_CONFIG.at_risk_threshold,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG
    )
