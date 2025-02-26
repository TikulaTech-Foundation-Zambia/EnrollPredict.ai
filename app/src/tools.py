import os
import pandas as pd
import joblib
from langchain.tools import tool
from pathlib import Path

# Define the model paths with proper path resolution
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CNN_MODEL_PATH = MODEL_DIR / "admission_cnn_model.keras"
SCALER_PATH = MODEL_DIR / "admission_scaler.pkl"
REGRESSOR_PATH = MODEL_DIR / "admission_prediction_model_regressor.pkl"

@tool
def predict_with_cnn(gre_score: float, toefl_score: float, sop: float, lor: float, cgpa: float) -> str:
    """
    Makes chance of admission predictions using a Convolutional Neural Network (CNN) model.
    
    Parameters:
        gre_score (float): GRE score of the applicant (130-170)
        toefl_score (float): TOEFL score of the applicant (0-120)
        sop (float): Statement of Purpose score (1-5)
        lor (float): Letter of Recommendation score (1-5)
        cgpa (float): CGPA of the applicant (0-10)
        
    Returns:
        str: Predicted chance of admission as a percentage
    """
    # Create input data
    input_data = pd.DataFrame({
        'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score],
        'SOP': [sop],
        'LOR ': [lor],
        'CGPA': [cgpa]
    })
    
    # Check if CNN model is available
    try:
        import tensorflow as tf
        if os.path.exists(CNN_MODEL_PATH) and os.path.exists(SCALER_PATH):
            # Load model and scaler
            loaded_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            loaded_scaler = joblib.load(SCALER_PATH)
            
            # Scale the input
            scaled_input = loaded_scaler.transform(input_data)
            
            # Reshape for CNN (samples, features, 1)
            reshaped_input = scaled_input.reshape(scaled_input.shape[0], scaled_input.shape[1], 1)
            
            # Make prediction
            prediction = loaded_model.predict(reshaped_input, verbose=0)
            chance = float(prediction[0][0] * 100)
            return f"Predicted chance of admission: {chance:.2f}%"
        else:
            # Fall back to regression model if CNN not found
            return "CNN model not found, using regression model."
    except ImportError:
        # TensorFlow not installed, use regression model
        return "TensorFlow not installed, using regression model."

def create_dataframe(index: int, gre_score: float, toefl_score: float, sop: float, lor: float, cgpa: float) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame with the required column names for prediction.
    Note: The model expects the 'LOR ' column (with a trailing space).
    """
    data = {
        "GRE Score": [gre_score],
        "TOEFL Score": [toefl_score],
        "SOP": [sop],
        "LOR ": [lor],  # Trailing space in column name
        "CGPA": [cgpa]
    }
    return pd.DataFrame(data, index=[index])

