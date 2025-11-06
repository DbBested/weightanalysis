# Quick Start Guide

## Installation (5 minutes)

### macOS
```bash
# Install dependencies
brew install eigen cmake

# Build the project
cd weightanalysis
./build.sh
```

### Linux
```bash
# Install dependencies
sudo apt-get install libeigen3-dev cmake build-essential python3-dev

# Build the project
cd weightanalysis
./build.sh
```

## Run Your First Analysis (2 minutes)

```bash
python3 src/python/main.py --video your_video.mp4 --lift-type squat
```

Replace:
- `your_video.mp4` with your video file path
- `squat` with `bench` or `deadlift` as needed

## What You'll Get

1. **Interactive Viewer** opens automatically
   - Press SPACE to play/pause
   - Use arrow keys to navigate
   - Press Q to quit

2. **Output Files** in `output/` directory:
   - `analysis_*.mp4` - Annotated video
   - `summary_*.json` - Analysis data

## Tips for Best Results

### Video Recording
✅ DO:
- Record from the side (perpendicular to movement)
- Use good lighting
- Keep lifter and bar fully visible
- Use tripod or stable surface

❌ DON'T:
- Record at angles
- Have cluttered background
- Zoom in too much (need full range of motion)
- Use shaky camera

### Video Quality
- Minimum: 720p @ 30fps
- Recommended: 1080p @ 30fps or 60fps
- Format: MP4, MOV, AVI

## Viewer Controls

| Key | Action |
|-----|--------|
| `SPACE` | Play/Pause |
| `←` `→` | Previous/Next frame |
| `+` `-` | Speed up/down |
| `T` | Toggle trajectory |
| `F` | Toggle force arrows |
| `B` | Toggle bar bending |
| `Q` | Quit |

## Example Output

### Video Overlay Shows:
- 🟢 **Green trail**: Bar path
- 🔵 **Cyan arrows**: Forces (with values in Newtons)
- 🟣 **Magenta curve**: Bar bending
- 🏷️ **Labels**: Current phase and time
- 📊 **Stats**: Peak force, current force

### JSON Summary Contains:
```json
{
  "peak_force_N": 2450,
  "peak_power_W": 1850,
  "phases": [...],
  "timing": {...}
}
```

## Troubleshooting

**Build fails?**
```bash
# Check dependencies
brew list eigen cmake  # macOS
dpkg -l | grep eigen   # Linux
```

**Can't find video?**
```bash
# Use absolute path
python3 src/python/main.py --video /full/path/to/video.mp4 --lift-type squat
```

**Poor tracking?**
- Try better lighting
- Ensure bar is visible
- Check video quality

## Next Steps

- See [USAGE.md](USAGE.md) for detailed documentation
- See [README.md](README.md) for technical details
- Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for implementation info

## Need Help?

- Issues: GitHub Issues
- Docs: README.md and USAGE.md
- Examples: See output/ after first run
