# ─────────────────────────────────────────────────────────────────────────────
# LLM Cost Autopilot — single image, entry point varies by service
#
# Services (set via CMD in docker-compose.yml):
#   api        → uvicorn src.api.main:app
#   worker     → python scripts/worker.py
#   dashboard  → streamlit run src/dashboard/app.py
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps — only what's truly needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create dirs that must exist at runtime
RUN mkdir -p data models logs

# Non-root user for security
RUN useradd -m -u 1000 autopilot && chown -R autopilot:autopilot /app
USER autopilot

# Default: API service. Overridden per-service in docker-compose.yml.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
