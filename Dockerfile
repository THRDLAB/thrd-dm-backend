FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libdmtx0a libdmtx-dev build-essential libjpeg62-turbo-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080","--no-server-header"]
