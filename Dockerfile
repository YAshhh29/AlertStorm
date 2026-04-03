FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY alertstorm /app/alertstorm
COPY openenv.yaml /app/openenv.yaml
COPY inference.py /app/inference.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
	CMD curl -f http://localhost:8000/tasks || exit 1

WORKDIR /app/alertstorm
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
