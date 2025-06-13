"""
Main entry point for the EnrollPredict.ai application.
"""
import uvicorn
from pathlib import Path
import sys

# Add the project root to the Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from api.main import app

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        reload_dirs=[str(BASE_DIR)]
    )
