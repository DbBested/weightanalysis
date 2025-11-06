#!/bin/bash
# Quick launcher for interactive viewer

cd "$(dirname "$0")"

echo "🎬 Starting Interactive Viewer..."
echo ""
echo "Controls:"
echo "  SPACE    - Play/Pause"
echo "  ← →      - Navigate frames"
echo "  + -      - Speed control"
echo "  T F B P S - Toggle overlays"
echo "  Q        - Quit"
echo ""
echo "Starting..."

python3 view_results.py
