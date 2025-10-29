# ---- Base image
FROM python:3.11-slim

# ---- Env de base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- Dépendances système (runtime pylibdmtx + Pillow) + curl pour le healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libdmtx0b libdmtx-dev libgl1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Code
COPY . .

# ---- Paramètres de perf/limites (tu peux ajuster via variables d'env à l'exécution)
ENV MAX_UPLOAD_MB=8 \
    MAX_SIDE_PX=1600 \
    TIME_BUDGET_MS=12000

# ---- Réseau
EXPOSE 8080

# ---- Healthcheck (nécessite /health sur FastAPI)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

# ---- Lancement (2 workers pour éviter qu'une requête longue bloque tout)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
