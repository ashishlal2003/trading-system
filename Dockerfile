FROM python:3.12-slim

WORKDIR /app

# libgomp1 is required by numba at runtime
RUN apt-get update && apt-get install -y \
    gcc g++ make curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user
RUN useradd -m -u 1000 trader \
    && mkdir -p /app/db /app/logs \
    && chown -R trader:trader /app
USER trader

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TZ=Asia/Kolkata

CMD ["python", "main.py"]
