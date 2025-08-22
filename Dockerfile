FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS builder
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

FROM base AS runtime
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import requests; import sys; sys.exit(0 if requests.get('http://localhost:8000/health', timeout=3).ok else 1)"
CMD ["uvicorn", "nge_trader.entrypoints.api:app", "--host", "0.0.0.0", "--port", "8000"]


