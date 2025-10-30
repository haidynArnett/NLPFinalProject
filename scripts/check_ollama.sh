#!/bin/bash

# Script to check if Ollama is installed and available over localhost

echo "Checking if Ollama is installed..."

# Check if ollama command exists
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed or not in PATH"
    exit 1
fi

echo "✅ Ollama command found"

# Check if Ollama service is running on localhost:11434
echo "Checking if Ollama service is running on localhost..."

if curl -s --max-time 5 http://localhost:11434/api/version &> /dev/null; then
    echo "✅ Ollama service is available on localhost:11434"
else
    echo "❌ Ollama service is not responding on localhost:11434"
    echo "   Make sure Ollama is running with 'ollama serve'"
    exit 1
fi

echo "🎉 Ollama is properly installed and available!"
exit 0
