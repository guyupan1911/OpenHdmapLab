#!/bin/bash

# Change to project root directory (parent of devtools)
cd "$(dirname "$0")/.."

echo "🚀 Starting development container..."
echo ""

# Check current container status
if docker compose -f docker/docker-compose.yml ps | grep -q "Up"; then
    echo "📦 Container is already running"
fi

# Build and start container
# Build uses host network (configured in docker-compose.yml) to inherit proxy settings
echo "🔨 Building and starting container..."
docker compose -f docker/docker-compose.yml up -d --build

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container started successfully"
    echo ""
    echo "💡 To enter the container:"
    echo "   ./dev_into.sh"
    echo ""
    echo "💡 To check container status:"
    echo "   docker compose -f docker/docker-compose.yml ps"
    echo ""
    echo "💡 To stop the container:"
    echo "   docker compose -f docker/docker-compose.yml down"
else
    echo ""
    echo "❌ Failed to start container, please check error messages"
    exit 1
fi
