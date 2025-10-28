FROM python:3.11-slim

# Evite les fichiers .pyc et flush stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dépendances système pour Pillow + pylibdmtx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libdmtx-dev \
    build-essential \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libffi-dev \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les deps Python en cache minimal
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Code
COPY app.py .

# Defaults perf (surchageables via variables d'env Northflank)
ENV PORT=8080 \
    WORKERS=2 \
    THREADS=1 \
    MAX_SIDE=1400 \
    URL_TIMEOUT=10 \
    MAX_DOWNLOAD_MB=8

EXPOSE 8080

# Healthcheck (utile pour Northflank)
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s CMD curl -fsS http://localhost:8080/health || exit 1

# Gunicorn + UvicornWorker = +robuste, prêt pour la concurrence
CMD ["bash","-lc","exec gunicorn -k uvicorn.workers.UvicornWorker -w ${WORKERS:-2} --threads ${THREADS:-1} -b 0.0.0.0:${PORT:-8080} --access-logfile - --timeout 30 app:app"]
