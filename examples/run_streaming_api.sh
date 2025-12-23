#!/bin/bash
# Script to run the streaming API server

cd /app/examples

# Find an available port
for port in 8001 8002 8003 8004 8005; do
    if ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo "Starting server on port $port..."
        python streaming_api.py --port $port
        break
    fi
done
