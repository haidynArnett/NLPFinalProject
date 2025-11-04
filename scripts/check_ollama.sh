#!/bin/bash

# Script to check if Ollama is installed and available over localhost

echo "Checking if Ollama is installed..."

# Check if ollama command exists
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed or not in PATH, This could make it hard to use Ollama via the command line"
fi

echo "Ollama command found"

# Determine the host to check
if [ -n "$OLLAMA_HOST" ]; then
    # Use OLLAMA_HOST environment variable if set
    OLLAMA_URL="$OLLAMA_HOST"
    # Add http:// prefix if not present
    if [[ ! "$OLLAMA_URL" =~ ^https?:// ]]; then
        OLLAMA_URL="http://$OLLAMA_URL"
    fi
    echo "Using custom Ollama host from OLLAMA_HOST: $OLLAMA_URL"
else
    # Default to localhost:11434
    OLLAMA_URL="http://localhost:11434"
    echo "Using default Ollama host: $OLLAMA_URL"
fi

# Check if Ollama service is running on the specified host
echo "Checking if Ollama service is running..."

if curl -s --max-time 5 "$OLLAMA_URL/api/version" &> /dev/null; then
    echo "Ollama service is available at $OLLAMA_URL"
else
    echo "Ollama service is not responding at $OLLAMA_URL"
    if [ -z "$OLLAMA_HOST" ]; then
        echo "   Make sure Ollama is running with 'ollama serve'"
    else
        echo "   Make sure OLLAMA_HOST is set correctly and the service is running"
    fi
    exit 1
fi

echo "Ollama is properly installed and available"
exit 0
