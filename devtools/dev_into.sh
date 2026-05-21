#!/bin/bash

# Change to project root directory (parent of devtools)
cd "$(dirname "$0")/.."

# Check if container is running
if ! docker compose -f docker/docker-compose.yml ps | grep -q "Up"; then
    echo "❌ Container is not running"
    echo ""
    echo "💡 Please start the container first:"
    echo "   ./dev_start.sh"
    exit 1
fi

# Enter container
echo "🔗 Entering development container..."
docker compose -f docker/docker-compose.yml exec dev bash
