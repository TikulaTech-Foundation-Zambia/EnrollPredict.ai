import tensorflow as tf
import pandas as pd
import joblib
from langchain.tools import tool

# Load the saved CNN model and scaler
loaded_model = tf.keras.models.load_model('app/src/model/admission_cnn_model.keras')
loaded_scaler = joblib.load('app/src/model/admission_scaler.pkl')

# Function to predict with the loaded CNN model
@tool
def predict_with_cnn(gre_score, toefl_score, sop, lor, cgpa):
    """
    Makes chance of admission predictions using a Convolutional Neural Network (CNN) model.
    
    Parameters:
        gre_score (float): GRE score of the applicant
        toefl_score (float): TOEFL score of the applicant
        sop (float): Statement of Purpose score (1-5)
        lor (float): Letter of Recommendation score (1-5)
        cgpa (float): CGPA of the applicant (0-10)
        
    Returns:
        float: Predicted chance of admission (0-100%)
    """
    # Create input data
    input_data = pd.DataFrame({
        'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score],
        'SOP': [sop],
        'LOR ': [lor],
        'CGPA': [cgpa]
    })
    
    # Scale the input
    scaled_input = loaded_scaler.transform(input_data)
    
    # Reshape for CNN (samples, features, 1)
    reshaped_input = scaled_input.reshape(scaled_input.shape[0], scaled_input.shape[1], 1)
    
    # Make prediction
    prediction = loaded_model.predict(reshaped_input, verbose=0)
    
    # Convert to percentage
    return float(prediction[0][0] * 100)