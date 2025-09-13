#!/bin/bash

# Fix Jupyter kernel configuration for wakeword environment

echo "🔧 Fixing Jupyter kernel configuration..."

# Activate the virtual environment
source wakeword_env/bin/activate

# Install ipykernel if not already installed
echo "📦 Installing ipykernel..."
pip install ipykernel

# Create a new kernel specification for the wakeword environment
echo "🎯 Creating Jupyter kernel for wakeword environment..."
python -m ipykernel install --user --name wakeword_env --display-name "Wakeword (GPU)"

echo "✅ Kernel created successfully!"
echo ""
echo "🔍 Checking available kernels:"
jupyter kernelspec list

echo ""
echo "🚀 Next steps:"
echo "1. Restart JupyterLab completely:"
echo "   - Close all browser tabs"
echo "   - Press Ctrl+C in the terminal"
echo "   - Start JupyterLab again:"
echo "     source wakeword_env/bin/activate"
echo "     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo ""
echo "2. In JupyterLab, select 'Wakeword (GPU)' kernel for your notebook"
echo "   - Click on the kernel name in the top right of the notebook"
echo "   - Select 'Wakeword (GPU)' from the dropdown"
echo ""
echo "3. If issues persist, try:"
echo "   jupyter kernelspec uninstall wakeword_env -y"
echo "   Then run this script again"

echo ""
echo "🎯 Troubleshooting commands:"
echo "   jupyter kernelspec list          # Show all available kernels"
echo "   jupyter kernelspec remove wakeword_env  # Remove if needed"