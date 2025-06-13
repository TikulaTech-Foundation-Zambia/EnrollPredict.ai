# EnrollPredict.ai

An AI-powered enrollment prediction application that uses machine learning models to predict admission chances based on academic credentials.

## 🚀 Features

- **AI Chat Interface**: Interactive chat powered by Groq's LLaMA model
- **ML Predictions**: Multiple machine learning models (CNN, LSTM, Regression) for enrollment prediction
- **Web Interface**: Modern, responsive web application
- **RESTful API**: Clean API endpoints for integration
- **Model Management**: Organized model storage and versioning

## 📁 Project Structure

```
EnrollPredict.ai/
├── api/                    # FastAPI application
│   ├── main.py            # Application entry point
│   ├── dependencies/      # Dependency injection
│   ├── routes/           # API route handlers
│   └── schemas/          # Pydantic models
├── src/                   # Source code
│   ├── data/             # Data processing utilities
│   ├── models/           # ML models and AI agents
│   ├── training/         # Training scripts
│   └── utils/            # Utility functions
├── web/                   # Web assets
│   ├── static/           # CSS, JS, images
│   └── templates/        # Jinja2 templates
├── models/               # Trained models and artifacts
│   ├── trained/          # Model files (.keras, .pkl)
│   ├── scalers/          # Data preprocessing scalers
│   └── configs/          # Model configurations
├── data/                 # Datasets
│   ├── raw/              # Raw data files
│   ├── processed/        # Cleaned data
│   └── datasets/         # Final datasets
├── notebooks/            # Jupyter notebooks
├── scripts/              # Utility scripts
├── tests/                # Test files
├── docs/                 # Documentation
├── main.py              # Application launcher
└── requirements.txt     # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EnrollPredict.ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 🚀 Running the Application

### Development Server
```bash
python main.py
```

### Production Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The application will be available at:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/api/docs
- Alternative Docs: http://localhost:8000/api/redoc

## 📊 Model Training

To retrain the models with new data:

```bash
cd src/training
python train.py
```

## 🔧 API Endpoints

### Web Routes
- `GET /` - Main page
- `GET /chat` - Chat interface
- `GET /about` - About page
- `GET /developers` - Developers page

### API Routes
- `POST /api/chat` - Send chat message
- `GET /api/chat/history/{session_id}` - Get chat history
- `DELETE /api/chat/history/{session_id}` - Clear chat history

## 🔍 Model Information

The application includes multiple models:
- **CNN Model**: Convolutional Neural Network for feature extraction
- **LSTM Model**: Long Short-Term Memory for sequence prediction
- **Regression Model**: Traditional ML regression approach

## 📝 Environment Variables

Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key_here
```

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Support

For questions or support, please open an issue in the repository.
