#!/bin/bash

# Quick fix for Jupyter kernel issues

echo "🚀 Quick Jupyter kernel fix..."

# Activate environment
source wakeword_env/bin/activate

# Install ipykernel
pip install ipykernel

# Remove any existing kernel (if it exists)
jupyter kernelspec uninstall wakeword_env -y 2>/dev/null || true

# Create new kernel
python -m ipykernel install --user --name wakeword_env --display-name "Wakeword (GPU)"

echo "✅ Kernel fixed!"
echo "📋 Available kernels:"
jupyter kernelspec list

echo ""
echo "🔄 Please restart JupyterLab:"
echo "1. Stop JupyterLab (Ctrl+C)"
echo "2. Start again: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo "3. Select 'Wakeword (GPU)' kernel in notebook"