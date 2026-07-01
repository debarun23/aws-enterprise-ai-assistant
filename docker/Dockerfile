FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose API port
EXPOSE 8000

# Run API
CMD ["python", "run_api.py"]