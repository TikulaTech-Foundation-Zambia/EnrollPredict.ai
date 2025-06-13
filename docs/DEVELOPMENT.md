# Development Guide - EnrollPredict.ai

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Test Structure**
   ```bash
   python test_structure.py
   ```

3. **Run Application**
   ```bash
   python main.py
   ```

## 📁 Project Organization

### `/api/` - Web API Layer
- `main.py` - FastAPI application setup
- `routes/` - API endpoint handlers
- `dependencies/` - Dependency injection (config, services)
- `schemas/` - Pydantic models for request/response

### `/src/` - Core Application Logic
- `models/` - AI agents and ML model interfaces
- `data/` - Data processing utilities
- `training/` - Model training scripts
- `utils/` - Shared utilities

### `/models/` - ML Artifacts
- `trained/` - Saved model files (.keras, .pkl)
- `scalers/` - Data preprocessing scalers
- `configs/` - Model configuration files

### `/web/` - Frontend Assets
- `static/` - CSS, JavaScript, images
- `templates/` - Jinja2 HTML templates

### `/data/` - Data Management
- `datasets/` - Training and test datasets
- `processed/` - Cleaned/preprocessed data
- `raw/` - Original raw data files

## 🔧 Development Workflow

### Adding New Models
1. Train model in `/notebooks/` or `/src/training/`
2. Save model to `/models/trained/`
3. Update `/models/configs/model_config.json`
4. Add model interface in `/src/models/`

### Adding New API Endpoints
1. Define schema in `/api/schemas/`
2. Create route handler in `/api/routes/`
3. Add dependencies if needed in `/api/dependencies/`
4. Update main app in `/api/main.py`

### Adding New Web Pages
1. Create HTML template in `/web/templates/`
2. Add static assets to `/web/static/`
3. Create route handler in `/api/routes/web.py`

## 🧪 Testing

- Run structure tests: `python test_structure.py`
- Run unit tests: `pytest tests/`
- Manual API testing: Visit http://localhost:8000/api/docs

## 📦 Dependencies

- **Core**: FastAPI, Uvicorn, Jinja2
- **AI/ML**: LangChain, LangGraph, Groq, TensorFlow, scikit-learn
- **Data**: Pandas, NumPy
- **Dev**: pytest, black, flake8

## 🚀 Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Traditional Hosting
```bash
pip install -r requirements.txt
gunicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🔍 Common Tasks

### Update Models
```bash
cd src/training
python train.py
```

### Add New Dependencies
```bash
pip install package_name
pip freeze > requirements.txt
```

### Database Migration (if added)
```bash
# Future: Add database migration scripts
```

## 🐛 Troubleshooting

### Import Errors
- Ensure virtual environment is activated
- Check that `__init__.py` files exist in directories
- Verify PYTHONPATH includes project root

### Model Loading Issues
- Check model file paths in `models/configs/model_config.json`
- Ensure model files exist in `models/trained/`
- Verify TensorFlow/scikit-learn versions

### API Issues
- Check `.env` file exists with required variables
- Verify port 8000 is available
- Check logs for detailed error messages

## 📖 Code Style

- Follow PEP 8 Python style guide
- Use type hints where possible
- Add docstrings to functions and classes
- Use meaningful variable names
- Keep functions small and focused

## 🔐 Security Notes

- Never commit API keys to git
- Use environment variables for sensitive config
- Validate all user inputs
- Implement rate limiting for production
- Use HTTPS in production
