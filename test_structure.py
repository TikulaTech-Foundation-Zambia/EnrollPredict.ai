"""
Test script to verify the organized project structure works correctly.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        from src.models.tools import predict_enrollment, get_admission_statistics
        print("✅ Tools import successful")
    except ImportError as e:
        print(f"❌ Tools import failed: {e}")
    
    try:
        from src.utils.model_manager import ModelManager
        print("✅ Model manager import successful")
    except ImportError as e:
        print(f"❌ Model manager import failed: {e}")
    
    try:
        from src.data.processor import DataProcessor
        print("✅ Data processor import successful")
    except ImportError as e:
        print(f"❌ Data processor import failed: {e}")
    
    try:
        from api.main import app
        print("✅ FastAPI app import successful")
    except ImportError as e:
        print(f"❌ FastAPI app import failed: {e}")

def test_data_loading():
    """Test data loading functionality."""
    print("\n📊 Testing data loading...")
    
    try:
        from src.data.processor import DataProcessor
        processor = DataProcessor()
        df = processor.load_enrollment_data()
        if df is not None:
            print(f"✅ Data loaded successfully: {len(df)} records")
        else:
            print("⚠️ Data loading returned None (file might not exist)")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")

def test_model_manager():
    """Test model manager functionality."""
    print("\n🤖 Testing model manager...")
    
    try:
        from src.utils.model_manager import ModelManager
        manager = ModelManager()
        models = manager.list_available_models()
        print(f"✅ Model manager initialized. Available models: {models}")
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")

def test_prediction_tool():
    """Test the prediction tool."""
    print("\n🎯 Testing prediction tool...")
    
    try:
        from src.models.tools import predict_enrollment
        
        # Test with sample data
        result = predict_enrollment(
            gre_score=320,
            toefl_score=110,
            sop=4.5,
            lor=4.0,
            cgpa=8.5
        )
        print("✅ Prediction tool working")
        print(f"Sample prediction result: {result[:100]}...")
    except Exception as e:
        print(f"❌ Prediction tool test failed: {e}")

def test_directory_structure():
    """Test that all required directories exist."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "api", "src", "models", "data", "web", 
        "notebooks", "scripts", "tests", "docs"
    ]
    
    project_root = Path(__file__).resolve().parent
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")

if __name__ == "__main__":
    print("🚀 EnrollPredict.ai Project Structure Test")
    print("=" * 50)
    
    test_directory_structure()
    test_imports()
    test_data_loading()
    test_model_manager()
    test_prediction_tool()
    
    print("\n🎉 Test completed!")
    print("\n📖 To start the application:")
    print("   python main.py")
    print("\n🌐 Then visit: http://localhost:8000")
