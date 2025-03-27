#!/bin/bash

echo "===== PYTHON VERSION ====="
python --version

echo "===== PIP VERSION ====="
pip --version

echo "===== INSTALLED PACKAGES ====="
pip list

echo "===== SYSTEM LIBRARIES ====="
ldconfig -p | grep -E 'libgl|libsm|libxext|libxrender'

echo "===== DISK SPACE ====="
df -h

echo "===== MEMORY USAGE ====="
free -h