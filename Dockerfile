FROM python:3.11-slim

WORKDIR /app

# Use different apt sources and install minimal required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create swap file
RUN dd if=/dev/zero of=/swapfile bs=1M count=1024 && \
    chmod 600 /swapfile && \
    mkswap /swapfile && \
    swapon /swapfile

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 100 --retries 5

# Copy application code
COPY . .

# Environment variables
ENV CUDA_VISIBLE_DEVICES=-1
ENV ONNXRUNTIME_EXECUTION_PROVIDERS=CPUExecutionProvider
ENV INSIGHTFACE_MODEL=buffalo_s

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
