#!/bin/bash
# Build script for weightlifting analysis system

set -e  # Exit on error

echo "========================================"
echo "Weightlifting Analysis - Build Script"
echo "========================================"

# Check for required tools
echo -e "\n[1/5] Checking dependencies..."

command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required but not installed."; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "Error: cmake is required but not installed."; exit 1; }

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "  ✓ Python ${PYTHON_VERSION}"

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
echo "  ✓ CMake ${CMAKE_VERSION}"

# Check for Eigen3
echo -e "\n[2/5] Checking for Eigen3..."
if [ -d "/usr/local/include/eigen3" ] || [ -d "/opt/homebrew/include/eigen3" ] || pkg-config --exists eigen3 2>/dev/null; then
    echo "  ✓ Eigen3 found"
else
    echo "  ⚠ Eigen3 not found. Installing via brew (macOS) or please install manually..."
    if command -v brew >/dev/null 2>&1; then
        brew install eigen
    else
        echo "  Error: Please install Eigen3 manually"
        echo "    macOS: brew install eigen"
        echo "    Linux: sudo apt-get install libeigen3-dev"
        exit 1
    fi
fi

# Install Python dependencies
echo -e "\n[3/5] Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "  ✓ Python packages installed"

# Build C++ backend
echo -e "\n[4/5] Building C++ backend..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "  Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "  Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

# Install Python module
echo "  Installing Python module..."
make install

cd ..

echo -e "\n[5/5] Verifying installation..."

# Copy module to src/python
cd build
cp weightanalysis_cpp*.so ../src/python/ 2>/dev/null || cp weightanalysis_cpp*.dylib ../src/python/ 2>/dev/null || true
cd ..

# Test import
python3 -c "import sys; sys.path.insert(0, 'src/python'); import weightanalysis_cpp; print('  ✓ C++ backend imported successfully')" || {
    echo "  ✗ C++ backend import failed"
    exit 1
}

echo -e "\n========================================"
echo "Build complete! 🎉"
echo "========================================"
echo ""
echo "To run the analysis:"
echo "  python3 src/python/main.py --video your_video.mp4 --lift-type squat"
echo ""
echo "For help:"
echo "  python3 src/python/main.py --help"
echo ""
