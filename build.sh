#!/bin/bash
# Render build script

# Install dependencies
pip install -r requirements.txt

# Download NLTK data if needed
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo "Build completed successfully!"
