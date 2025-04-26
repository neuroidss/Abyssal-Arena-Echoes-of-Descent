#!/bin/bash

echo "Attempting to start a simple Python HTTP server..."
echo "Make sure you are in the directory containing index.html and stch.js"
echo "Open your browser to http://localhost:8000 (or the address shown)"
echo "Press Ctrl+C to stop the server."

# Try Python 3 first
if command -v python3 &> /dev/null
then
    python3 -m http.server 8000
# Else try Python 2
elif command -v python &> /dev/null
then
    python -m SimpleHTTPServer 8000
# Else try Node's http-server if available
elif command -v npx &> /dev/null
then
    echo "Python not found, trying npx http-server..."
    npx http-server -p 8000 .
else
    echo "Error: Could not find Python 3, Python 2, or npx."
    echo "Please install one of them or use another method to serve the files locally."
    exit 1
fi
