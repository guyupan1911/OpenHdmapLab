#!/bin/bash
set -e

# Change to project root directory (parent of devtools)
cd "$(dirname "$0")/.."

COMPOSE_FILE="docker/docker-compose.yml"

echo "🚀 Starting development container..."
echo ""

# Check current container status
if docker compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
    echo "📦 Container is already running"
    echo ""
    echo "💡 To enter the container:"
    echo "   ./dev_into.sh"
    exit 0
fi

echo "▶️  Starting container..."
echo "   Docker Compose will use the existing image or build when it needs to."

docker compose -f "$COMPOSE_FILE" up -d

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
