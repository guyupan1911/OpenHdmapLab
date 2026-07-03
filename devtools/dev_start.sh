#!/bin/bash
set -e

# Change to project root directory (parent of devtools)
cd "$(dirname "$0")/.."

COMPOSE_FILE="docker/docker-compose.yml"

BUILD_IMAGE=false
if [[ "${1:-}" == "--build" ]]; then
    BUILD_IMAGE=true
elif [[ -n "${1:-}" ]]; then
    echo "Usage: $0 [--build]"
    exit 1
fi

echo "🚀 Starting development container..."
echo ""

compose_args=(up -d)
if [[ "$BUILD_IMAGE" == "true" ]]; then
    compose_args+=(--build)
    echo "▶️  Building image and starting container..."
else
    echo "▶️  Starting container..."
    echo "   Docker Compose will recreate the container if docker-compose.yml changed."
    echo "   It will only build the image when the configured image is missing."
fi

docker compose -f "$COMPOSE_FILE" "${compose_args[@]}"

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
