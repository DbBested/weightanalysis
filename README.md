# Weightlifting Analysis System

This is a tool for analyzing weightlifting videos using computer vision and physics simulation. It tracks the lifter and barbell, computes forces through inverse dynamics, and visualizes everything on top of the video.

## What it does

The system automatically detects your body position using MediaPipe and tracks the barbell through the lift. It then runs multibody inverse dynamics to calculate forces, torques, and power output. You get an annotated video showing force arrows, bar bending, trajectory paths, and timing for each phase of the lift.

The physics calculations run in C++ for performance (about 10-100x faster than pure Python), while the video processing and UI are in Python. On a typical laptop, processing takes about 1.5 minutes per minute of 1080p video.

## Features

**Computer Vision**
- Full body pose tracking with MediaPipe
- Automatic barbell detection and tracking using Hough line transforms
- Works through partial occlusions and motion blur

**Physics**
- Multibody inverse dynamics using recursive Newton-Euler algorithm
- Estimates bar mass and body segment parameters from the video
- Physics-informed Kalman filtering to smooth noisy tracking data
- Bar bending analysis using Euler-Bernoulli beam theory

**Visualization**
- Force vectors drawn on the bar with magnitude in Newtons
- Bar deflection overlay showing where it bends
- Trajectory trail following the bar path
- Automatic phase labels (setup, descent, bottom, ascent, completion)
- Summary stats for peak force, power, and timing

**Performance**
- C++ backend handles all physics computation
- Runs on regular laptops without needing a GPU
- Processes about 30+ fps on modern CPUs
- Multi-threaded frame analysis

## Installation

You need Python 3.8+, CMake 3.15+, a C++17 compiler, and Eigen3.

**macOS:**
```bash
brew install eigen cmake
git clone https://github.com/DbBested/weightanalysis.git
cd weightanalysis
./build.sh
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libeigen3-dev cmake build-essential python3-dev
git clone https://github.com/DbBested/weightanalysis.git
cd weightanalysis
./build.sh
```

**Windows:**
```bash
# Install Eigen3 and CMake via vcpkg or manually, then:
git clone https://github.com/DbBested/weightanalysis.git
cd weightanalysis
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Usage

```bash
python3 src/python/main.py --video squat.mp4 --lift-type squat
```

The program will process the video and open an interactive viewer when done. You can step through frames, toggle overlays, and adjust playback speed. It also saves an annotated video and JSON file with all the analysis data.

See [USAGE.md](USAGE.md) for more details.

## Output

The system generates two files:

An annotated video with force arrows, bar bending visualization, trajectory trail, and phase labels overlaid on each frame.

A JSON file with the raw data:
```json
{
  "lift_type": "squat",
  "duration": 5.2,
  "dynamics": {
    "peak_force_N": 2450.5,
    "peak_force_time_s": 2.8,
    "peak_power_W": 1850.2,
    "estimated_mass_kg": 102.5
  },
  "phases": [...]
}
```

## Supported Lifts

Currently works with squats, bench press, and deadlifts. Overhead press is partially implemented. Olympic lifts (snatch, clean & jerk) might be added later but need more complex models.

## How it works

**Computer Vision**
- Uses MediaPipe Pose (BlazePose model) for body tracking
- Tracks the bar with Hough line transforms plus optical flow
- Color-based segmentation and edge detection for bar isolation

**Biomechanics**
- Multi-segment rigid body model with 6-12 degrees of freedom
- Recursive Newton-Euler Algorithm for inverse dynamics
- Central difference for velocities and accelerations with Savitzky-Golay smoothing
- Constrained least squares and Kalman filtering for parameter estimation

**Bar Physics**
- Euler-Bernoulli beam theory for deflection
- Models a standard Olympic bar (steel, E = 200 GPa)
- Superposition of point loads at hand positions
- Highlights critical regions where bending moment exceeds 70% of peak

## Benchmarks

Tested on Apple M1 Pro:
- Video Processing: ~60 fps
- Pose Detection: ~35 fps
- Bar Tracking: ~120 fps
- Inverse Dynamics: ~2000 fps (C++ backend)
- Overall: ~1.5 minutes per minute of 1080p video

## Project Structure

```
weightanalysis/
├── src/
│   ├── cpp/              # C++ backend
│   │   ├── dynamics/     # Inverse dynamics, multibody model
│   │   ├── physics/      # Bar bending
│   │   └── bindings.cpp  # Python bindings
│   └── python/           # Python frontend
│       ├── video_processor.py
│       ├── pose_detector.py
│       ├── bar_tracker.py
│       ├── phase_detector.py
│       ├── visualizer.py
│       ├── viewer.py
│       └── main.py
├── CMakeLists.txt
├── requirements.txt
├── build.sh
└── README.md
```

## Building from Source

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j
ctest  # if you have tests
make install
```
