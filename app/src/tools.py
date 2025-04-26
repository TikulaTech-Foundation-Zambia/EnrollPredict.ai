import os
import pandas as pd
import numpy as np
import joblib
from langchain.tools import tool
from pathlib import Path

# Define the model paths with proper path resolution
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CNN_MODEL_PATH = MODEL_DIR / "admission_cnn_model.keras"
LSTM_MODEL_PATH = MODEL_DIR / "admission_lstm_model.keras"
SCALER_PATH = MODEL_DIR / "admission_scaler.pkl"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.pkl"
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

@tool
def predict_enrollment_with_lstm(
    program: str, 
    year: int, 
    quarter: int, 
    previous_enrollment: float,
    applications_received: float,
    acceptance_rate: float,
    conversion_rate: float,
    mining_industry_growth: float = 0.05,
    marketing_budget: int = 80000,
    scholarship_funds: int = 150000,
    competitor_programs: int = 0,
    average_test_score: float = 70.0,
    female_applicant_percentage: int = 40,
    urban_applicant_percentage: int = 70,
    international_applicant_percentage: int = 15,
    faculty_student_ratio: float = 0.075,
    program_ranking: int = 3,
    program_trend_sentiment: float = 0.80,
    economic_sentiment: float = 0.75,
    general_interest: float = 0.80
) -> str:
    """
    Makes enrollment predictions using a time series LSTM model.
    
    Parameters:
        program (str): Program name (Mining Engineering, Computer Science, Business Administration, Electrical Engineering, or Environmental Science)
        year (int): Year for prediction (e.g., 2025)
        quarter (int): Quarter for prediction (1-4)
        previous_enrollment (float): Previous year's enrollment number
        applications_received (float): Number of applications received
        acceptance_rate (float): Acceptance rate (0.0-1.0)
        conversion_rate (float): Conversion rate (0.0-1.0)
        mining_industry_growth (float, optional): Mining industry growth rate (0.0-1.0)
        marketing_budget (int, optional): Marketing budget allocation
        scholarship_funds (int, optional): Scholarship funds available
        competitor_programs (int, optional): Number of new competitor programs
        average_test_score (float, optional): Average entry test score
        female_applicant_percentage (int, optional): Percentage of female applicants
        urban_applicant_percentage (int, optional): Percentage of urban applicants
        international_applicant_percentage (int, optional): Percentage of international applicants
        faculty_student_ratio (float, optional): Faculty to student ratio
        program_ranking (int, optional): Program ranking (lower is better)
        program_trend_sentiment (float, optional): Program trend sentiment (0.0-1.0)
        economic_sentiment (float, optional): Economic factor sentiment (0.0-1.0)
        general_interest (float, optional): General interest sentiment (0.0-1.0)
        
    Returns:
        str: Predicted enrollment and percentage change
    """
    # Validate input data
    valid_programs = ["Mining Engineering", "Computer Science", "Business Administration", 
                     "Electrical Engineering", "Environmental Science"]
    
    if program not in valid_programs:
        return f"Error: Program must be one of {', '.join(valid_programs)}"
    
    # Create input DataFrame
    input_data = {
        "Year": year,
        "Quarter": quarter,
        "Previous_Year_Enrollment": previous_enrollment,
        "Applications_Received": applications_received,
        "Acceptance_Rate": acceptance_rate,
        "Conversion_Rate": conversion_rate,
        "Mining_Industry_Growth": mining_industry_growth,
        "Secondary_School_Graduates": 3470,  # Default value based on recent years
        "Marketing_Budget_Allocation": marketing_budget,
        "Scholarship_Funds_Available": scholarship_funds,
        "Competitor_New_Programs": competitor_programs,
        "Average_Entry_Test_Score": average_test_score,
        "Female_Applicant_Percentage": female_applicant_percentage,
        "Urban_Applicant_Percentage": urban_applicant_percentage,
        "International_Applicant_Percentage": international_applicant_percentage,
        "Faculty_Student_Ratio": faculty_student_ratio,
        "Program_Ranking": program_ranking,
        "Program_Trend_Sentiment": program_trend_sentiment,
        "Economic_Factor_Sentiment": economic_sentiment,
        "General_Interest_Sentiment": general_interest,
        "Program": program
    }
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    try:
        import tensorflow as tf
        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(TARGET_SCALER_PATH):
            # Load model and scalers
            loaded_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
            feature_scaler = joblib.load(SCALER_PATH)
            target_scaler = joblib.load(TARGET_SCALER_PATH)
            
            # One-hot encode the program
            df_encoded = pd.get_dummies(df, columns=['Program'], drop_first=False)
            
            # Ensure we have the expected feature columns
            # Get the feature columns from your model (these should match what was used in training)
            expected_columns = feature_scaler.feature_names_in_
            
            # Create a DataFrame with the proper columns
            input_features = pd.DataFrame(columns=expected_columns)
            
            # Fill in with values from df_encoded where possible
            for col in expected_columns:
                if col in df_encoded.columns:
                    input_features[col] = df_encoded[col]
                else:
                    # If column is missing (e.g., one-hot encoded column), fill with zeros
                    input_features[col] = 0
            
            # Scale the input features
            scaled_input = feature_scaler.transform(input_features)
            
            # Reshape for LSTM (samples, timesteps, features)
            reshaped_input = scaled_input.reshape(scaled_input.shape[0], 1, scaled_input.shape[1])
            
            # Make prediction
            scaled_prediction = loaded_model.predict(reshaped_input, verbose=0)
            
            # Inverse transform to get the original scale
            prediction = target_scaler.inverse_transform(scaled_prediction)
            
            # Calculate percentage change
            predicted_enrollment = float(prediction[0][0])
            percentage_change = ((predicted_enrollment - previous_enrollment) / previous_enrollment) * 100
            
            return (f"Program: {program}\n"
                   f"Predicted Enrollment: {predicted_enrollment:.0f}\n"
                   f"Previous Enrollment: {previous_enrollment:.0f}\n"
                   f"Change: {percentage_change:.2f}%")
        else:
            missing_files = []
            if not os.path.exists(LSTM_MODEL_PATH):
                missing_files.append("LSTM model file")
            if not os.path.exists(SCALER_PATH):
                missing_files.append("feature scaler file")
            if not os.path.exists(TARGET_SCALER_PATH):
                missing_files.append("target scaler file")
            
            return f"Error: Missing required files: {', '.join(missing_files)}"
    except ImportError:
        return "Error: TensorFlow not installed, cannot make predictions with LSTM model."
    except Exception as e:
        return f"Error making prediction: {str(e)}"

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


import os
import joblib
import numpy as np
import tensorflow as tf
from pydantic.v1 import BaseModel, Field # Use pydantic v1 if required by Langchain/Langgraph version
from langchain_core.tools import tool

# --- Constants ---
# Ensure these paths point to where the files are located in your LangGraph environment
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "predict_model"
MODEL_PATH = MODEL_DIR / "enrollment_predictor_final.keras"
SCALER_X_PATH = MODEL_DIR / "scaler_x_final.joblib"
SCALER_Y_PATH = MODEL_DIR / "scaler_y_final.joblib"

# Define the exact feature names in the order the model expects them
# This MUST match the order used during training the final model
EXPECTED_FEATURES = [
    'Year', 'Previous_Year_Enrollment', 'Applications_Received', 'Acceptance_Rate', 'Conversion_Rate',
    'Secondary_School_Graduates', 'Scholarship_Funds_Available', 'Average_Entry_Test_Score',
    'Female_Applicant_Percentage', 'Urban_Applicant_Percentage', 'International_Applicant_Percentage',
    'Faculty_Student_Ratio'
]

# --- Input Schema ---
class EnrollmentPredictionInput(BaseModel):
    """Input schema for the enrollment prediction tool."""
    Year: int = Field(..., description="The target year for the prediction.")
    Previous_Year_Enrollment: int = Field(..., description="Actual enrollment figure from the previous year.")
    Applications_Received: int = Field(..., description="Total number of applications received.")
    Acceptance_Rate: float = Field(..., description="The proportion of applicants offered admission (e.g., 0.3 for 30%).")
    Conversion_Rate: float = Field(..., description="The proportion of accepted students who enroll (e.g., 0.4 for 40%). Should typically be <= Acceptance_Rate.")
    Secondary_School_Graduates: int = Field(..., description="Number of secondary school graduates in the relevant catchment area.")
    Scholarship_Funds_Available: float = Field(..., description="Total amount of scholarship funds available.")
    Average_Entry_Test_Score: float = Field(..., description="Average score on standardized entry tests for the incoming cohort.")
    Female_Applicant_Percentage: float = Field(..., description="Percentage of applicants who are female (0-100).")
    Urban_Applicant_Percentage: float = Field(..., description="Percentage of applicants from urban areas (0-100).")
    International_Applicant_Percentage: float = Field(..., description="Percentage of applicants who are international students (0-100).")
    Faculty_Student_Ratio: float = Field(..., description="Ratio of faculty members to students (e.g., 1 faculty per 15 students = 1/15 = 0.067).")

# --- Tool Definition ---
@tool("predict_university_enrollment", args_schema=EnrollmentPredictionInput)
def predict_enrollment(
    Year: int,
    Previous_Year_Enrollment: int,
    Applications_Received: int,
    Acceptance_Rate: float,
    Conversion_Rate: float,
    Secondary_School_Graduates: int,
    Scholarship_Funds_Available: float,
    Average_Entry_Test_Score: float,
    Female_Applicant_Percentage: float,
    Urban_Applicant_Percentage: float,
    International_Applicant_Percentage: float,
    Faculty_Student_Ratio: float
) -> dict:
    """
    Predicts the university enrollment for a given year based on various input factors.
    Requires the pre-trained model and scaler files to be available.
    """
    

    # Check if necessary files exist
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH]):
        error_msg = f"Error: Model or scaler file(s) not found. Ensure '{MODEL_PATH}', '{SCALER_X_PATH}', and '{SCALER_Y_PATH}' are accessible."
        print(error_msg)
        return {"error": error_msg}

    try:
        # Load model and scalers
        print("Loading model and scalers...")
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler_x = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        print("Model and scalers loaded successfully.")

        # Prepare input data from function arguments, ensuring correct order
        input_data_dict = locals() # Gets all local variables (function arguments) as a dict
        # Ensure all expected features are present in the input dict
        if not all(feat in input_data_dict for feat in EXPECTED_FEATURES):
             missing = [f for f in EXPECTED_FEATURES if f not in input_data_dict]
             error_msg = f"Error: Missing required input features: {missing}"
             print(error_msg)
             return {"error": error_msg}

        # Create numpy array in the correct feature order
        input_array = np.array([[input_data_dict[feat] for feat in EXPECTED_FEATURES]], dtype=np.float32)

        # Scale features
        print("Scaling input data...")
        input_scaled = scaler_x.transform(input_array)

        # Make prediction
        print("Making prediction...")
        pred_scaled = model.predict(input_scaled, verbose=0)

        # Inverse transform prediction
        print("Inverse transforming prediction...")
        prediction_original_scale = scaler_y.inverse_transform(pred_scaled)
        predicted_value = prediction_original_scale[0][0]

        print(f"Prediction successful: {predicted_value:.0f}")
        # Return result in a dictionary format
        return {"predicted_enrollment": round(predicted_value)}

    except Exception as e:
        error_msg = f"An error occurred during prediction: {e}"
        print(error_msg)
        # Return error in a dictionary format
        return {"error": error_msg}

