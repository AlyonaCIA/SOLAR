#!/bin/bash
# filepath: c:\Users\scalderonl\Documents\SOLAR\SOLAR\cloudbuild.sh

# Exit on any error
set -e

# Configuration
PROJECT_ID="mlopsprojectdev"
REGION="us-central1"
SERVICE_NAME="solar-api"
REPOSITORY="solar-api-repo"
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)
IMAGE_NAME="us-central1-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE_NAME:$IMAGE_TAG"

echo "====== SOLAR API DEPLOYMENT SCRIPT ======"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Image: $IMAGE_NAME"
echo "========================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud command not found. Please install Google Cloud SDK."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker command not found. Please install Docker."
    exit 1
fi

# Ensure we're in the right project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Ensure Docker is authenticated with Google Cloud
echo "Configuring Docker authentication..."
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME -f API/Dockerfile .

# Push the image
echo "Pushing image to Artifact Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 10

# Get the deployed URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format="value(status.url)")

echo "====== DEPLOYMENT COMPLETE ======"
echo "Service URL: $SERVICE_URL"
echo "Test the /analyze endpoint:"
echo "curl -X POST \"$SERVICE_URL/analyze\" -H \"Content-Type: multipart/form-data\" -F \"files=@sample_file.jp2\""
echo ""
echo "Test the /get-image-analyze endpoint:"
echo "curl -X POST \"$SERVICE_URL/get-image-analyze\" -H \"Content-Type: application/json\" -d '{\"timestamp\":\"2023-05-01T12:00:00Z\"}'"
echo "================================="