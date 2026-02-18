#!/bin/bash
# Prepare project for Colab upload
# This script creates a ZIP file with only the essential files needed for training

echo "🚀 Preparing project for Google Colab..."
echo "========================================="

PROJECT_DIR="/Users/xe/Documents/GITHUB CAPSTONE /Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures"
OUTPUT_ZIP="$HOME/Desktop/colab_project.zip"

cd "$PROJECT_DIR"

echo "📦 Creating ZIP file (excluding unnecessary files)..."
echo ""

# Create ZIP excluding large/unnecessary files
zip -r "$OUTPUT_ZIP" \
  src/ \
  data/processed/ \
  notebooks/03_ml_model_training.ipynb \
  configs/ \
  models/checkpoints/.gitkeep \
  reports/figures/.gitkeep \
  requirements.txt \
  -x "*.git*" \
  -x "*__pycache__*" \
  -x "*.venv*" \
  -x "*.pyc" \
  -x ".DS_Store" \
  -x "data/raw/*" \
  -x "logs/*" \
  -x "*.log"

echo ""
echo "✅ ZIP file created!"
echo "📍 Location: $OUTPUT_ZIP"
echo "📊 Size: $(du -h "$OUTPUT_ZIP" | cut -f1)"
echo ""
echo "📤 Next steps:"
echo "1. Open your Colab notebook in VS Code"
echo "2. Click the 📁 folder icon in Colab file browser"
echo "3. Click upload button and select: colab_project.zip"
echo "4. In a notebook cell, run:"
echo "   !unzip -q /content/colab_project.zip -d /content/"
echo "   !mv /content/Agentic-Early-Warning-Intelligence-System-for-Silent-System-Failures /content/"
echo ""
echo "🚀 Ready to train on Colab GPU!"
