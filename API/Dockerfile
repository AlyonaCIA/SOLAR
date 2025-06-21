# Base image with Python
FROM python:3.11-slim

# Install system dependencies including OpenJPEG for JP2 support
RUN apt-get update && apt-get install -y \
    libopenjp2-7 \
    libopenjp2-7-dev \
    gcc \
    g++ \
    # Add these for FITS support
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Accept PORT as build argument with default value of 8080
ARG PORT=8080
ENV PORT=${PORT}

# Copy the whole project
COPY . .

# Install the solar package
RUN pip install -e .

# Install API requirements
RUN pip install --no-cache-dir -r API/requirements.txt

# Create necessary directories
RUN mkdir -p temp_data outputs

# Change to the API directory
WORKDIR /app/API

# Create a startup script to help debug module issues
RUN echo '#!/bin/bash\n\
echo "Current directory: $(pwd)"\n\
echo "Directory listing:"\n\
ls -la\n\
echo "app directory:"\n\
ls -la app\n\
echo "Starting server on port: $PORT"\n\
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}' > /app/API/start.sh && \
chmod +x /app/API/start.sh

# Expose the port
EXPOSE ${PORT}

# Use the startup script
CMD ["/app/API/start.sh"]