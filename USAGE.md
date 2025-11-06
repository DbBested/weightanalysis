# Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/weightanalysis.git
cd weightanalysis

# Run the build script
./build.sh
```

The build script will:
- Check for required dependencies
- Install Python packages
- Build the C++ backend
- Install the Python module

### 2. Basic Usage

Analyze a weightlifting video:

```bash
python3 src/python/main.py --video path/to/video.mp4 --lift-type squat
```

## Command Line Options

```
--video PATH          Path to input video file (required)
--lift-type TYPE      Type of lift: squat, bench, or deadlift (default: squat)
--output DIR          Output directory for results (default: output/)
```

## Examples

### Squat Analysis
```bash
python3 src/python/main.py \
    --video squat_video.mp4 \
    --lift-type squat \
    --output results/squat/
```

### Bench Press Analysis
```bash
python3 src/python/main.py \
    --video bench_video.mp4 \
    --lift-type bench \
    --output results/bench/
```

### Deadlift Analysis
```bash
python3 src/python/main.py \
    --video deadlift_video.mp4 \
    --lift-type deadlift \
    --output results/deadlift/
```

## Interactive Viewer Controls

Once the analysis completes, an interactive viewer window will open:

| Key | Action |
|-----|--------|
| `SPACE` | Play/Pause |
| `←` / `→` | Step backward/forward one frame |
| `+` / `=` | Increase playback speed |
| `-` | Decrease playback speed |
| `T` | Toggle trajectory display |
| `F` | Toggle force arrows |
| `B` | Toggle bar bending visualization |
| `P` | Toggle phase labels |
| `S` | Toggle summary overlay |
| `R` | Reset to first frame |
| `Q` / `ESC` | Quit viewer |

## Output Files

After analysis, the following files are created in the output directory:

1. **`analysis_*.mp4`** - Annotated video with all visualizations
2. **`summary_*.json`** - JSON file with analysis results:
   ```json
   {
     "video": "input_video.mp4",
     "lift_type": "squat",
     "duration": 5.2,
     "phases": [
       {
         "name": "descent",
         "start_time": 0.5,
         "end_time": 2.1,
         "duration": 1.6
       },
       ...
     ],
     "dynamics": {
       "peak_force_N": 2450.5,
       "peak_force_time_s": 2.8,
       "peak_power_W": 1850.2,
       "estimated_mass_kg": 102.5
     }
   }
   ```

## Understanding the Visualizations

### Force Arrows
- **Cyan arrows** show the instantaneous force on the barbell
- Arrow direction indicates force direction
- Arrow length is proportional to force magnitude
- **Numeric value** shows force in Newtons (N)

### Bar Bending
- **Magenta/Pink curve** shows the deflected bar shape
- Gray line shows the nominal (straight) bar position
- **Red highlighted regions** indicate critical bending zones
- Maximum deflection is displayed in millimeters

### Trajectory
- **Green trail** shows the bar's path through space
- Fading effect shows recent history
- Helps identify bar path inefficiencies

### Phase Labels
- **Color-coded label** at top shows current lift phase:
  - **Gray**: Setup
  - **Orange**: Descent (eccentric phase)
  - **Red**: Bottom (transition)
  - **Green**: Ascent (concentric phase)
  - **Yellow**: Completion
- Timestamp shows elapsed time

### Summary Overlay
- **Bottom-right panel** shows real-time stats:
  - Peak force (overall maximum)
  - Current force
  - Current phase

## Tips for Best Results

### Video Recording
1. **Camera Position**: Side view perpendicular to the bar's movement
2. **Lighting**: Good, even lighting without harsh shadows
3. **Background**: Uncluttered background for better detection
4. **Framing**: Ensure entire lifter and bar path are visible
5. **Stability**: Use a tripod or stable surface

### Lift Execution
1. Use colored tape or markers on the bar for better tracking
2. Wear form-fitting clothing for better pose detection
3. Ensure the bar is clearly visible throughout the lift
4. Avoid obstructions between camera and lifter

### Performance
- For faster processing on large videos, the system automatically downsamples to 30 fps
- C++ backend provides significant speedup for dynamics calculations
- On modern consumer hardware, expect ~1-2 minutes per minute of video

## Troubleshooting

### "C++ backend not compiled" Warning
Run the build script:
```bash
./build.sh
```

### Poor Pose Detection
- Improve lighting
- Ensure lifter is fully visible
- Remove background clutter
- Use higher resolution video

### Bar Tracking Lost
- Use brighter/metallic bar
- Add visual markers to bar
- Ensure bar is clearly distinguishable from background
- Check that bar remains in frame

### Installation Issues

**Eigen3 not found:**
```bash
# macOS
brew install eigen

# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# Fedora
sudo dnf install eigen3-devel
```

**pybind11 errors:**
```bash
pip install --upgrade pybind11
```

## Advanced Usage

### Python API

You can use the individual modules programmatically:

```python
from video_processor import VideoProcessor
from pose_detector import PoseDetector
from bar_tracker import BarTracker
import weightanalysis_cpp as wa_cpp

# Process video
with VideoProcessor('video.mp4') as video:
    frames = video.extract_frames()

# Detect poses
detector = PoseDetector()
poses = detector.detect_pose_sequence(frames)

# Track bar
tracker = BarTracker()
detections = tracker.track_sequence(frames)

# Run dynamics
model = wa_cpp.MultibodyModel()
model.initialize_squat_model()
solver = wa_cpp.InverseDynamicsSolver(model)
result = solver.solve(joint_states, bar_states)
```

### Custom Visualization

```python
from visualizer import LiftVisualizer

visualizer = LiftVisualizer(1920, 1080)

# Customize colors
visualizer.colors['force_arrow'] = (255, 0, 0)  # Red

# Create custom visualization
annotated = visualizer.create_complete_visualization(
    frame, bar_center, bar_endpoints,
    force_vector, deflection, critical_regions,
    trajectory, frame_idx, phase_name, time, summary
)
```

## Performance Optimization

For best performance:

1. **Use SSD storage** for video files
2. **Close other applications** during analysis
3. **Use lower resolution** if speed is critical
4. **Ensure adequate RAM** (8GB+ recommended)
5. The C++ backend is compiled with `-O3 -march=native` for maximum performance

## Citation

If you use this tool in research, please cite:

```
@software{weightlifting_analysis,
  title = {Weightlifting Analysis System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/weightanalysis}
}
```
