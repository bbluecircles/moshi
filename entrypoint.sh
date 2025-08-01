#!/bin/bash

# Check if NGROK_AUTH_TOKEN is provided
if [ -z "$NGROK_AUTH_TOKEN" ]; then
    echo "Error: NGROK_AUTH_TOKEN environment variable is not set."
    exit 1
fi

# Configure Ngrok with auth token
ngrok config add-authtoken $NGROK_AUTH_TOKEN

# Start Ngrok in the background to expose port 8000
ngrok http --url=myfantasyvoice.ngrok.dev 8000 &

# Wait briefly to ensure Ngrok starts
sleep 5

# Start FastAPI application
python3 moshi/moshi/main.py
