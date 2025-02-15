import uvicorn
from pathlib import Path
from src.app import app

# Create necessary directories if they don't exist
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
