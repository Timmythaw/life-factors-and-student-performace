.PHONY: install train run test clean debug help

# Default target
help:
	@echo "Student Risk Classifier - Available Commands"
	@echo "============================================="
	@echo ""
	@echo "  make install    - Install dependencies with uv"
	@echo "  make train      - Train the model"
	@echo "  make run        - Start the FastAPI server"
	@echo "  make dev        - Start server in development mode"
	@echo "  make test       - Run tests"
	@echo "  make debug      - Run model debug script"
	@echo "  make clean      - Remove cached files"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code with black"
	@echo ""

# Install dependencies
install:
	uv sync

# Train the model
train:
	uv run python scripts/train_model.py

# Run the API server
run:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run in development mode with auto-reload
dev:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	uv run pytest tests/ -v

# Debug model loading
debug:
	uv run python src/debug_model.py

# Clean cached files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Lint code
lint:
	uv run ruff check src/ scripts/ tests/

# Format code
format:
	uv run black src/ scripts/ tests/

# Build Docker image
docker-build:
	docker build -t student-risk-api:latest .

# Run Docker container
docker-run:
	docker run -d --name student-api -p 8000:8000 \
		-v $(PWD)/models:/app/models:ro \
		student-risk-api:latest

# Stop Docker container
docker-stop:
	docker stop student-api && docker rm student-api