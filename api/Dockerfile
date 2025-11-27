# Dockerfile for Rossmann Sales Forecasting API
# Build from project root: docker build -f api/Dockerfile -t rossmann-api .

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY api/requirements.txt /app/api/requirements.txt
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/api/requirements.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# Optional: Install TensorFlow for LSTM support (uncomment if needed)
# RUN pip install --no-cache-dir tensorflow==2.15.0

# Copy application code
COPY api /app/api
COPY src /app/src
COPY models /app/models
COPY data /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

