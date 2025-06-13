#!/bin/bash

# Development setup script for EnrollPredict.ai

echo "🚀 Setting up EnrollPredict.ai development environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file..."
    cat > .env << EOF
GROQ_API_KEY=your_groq_api_key_here
DEBUG=True
EOF
    echo "📝 Please edit .env file with your actual API keys"
fi

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "🌐 The app will be available at http://localhost:8000"
