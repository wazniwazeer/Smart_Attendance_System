FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System prep (optional but harmless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App + artifacts (for local dev; in prod you can mount a volume instead)
COPY app.py ./app.py
COPY artifacts ./artifacts

EXPOSE 5000
CMD ["python", "app.py"]
