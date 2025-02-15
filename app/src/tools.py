import joblib
import os
import numpy as np
from typing import Dict, Any, List
from langchain.tools import BaseTool

class SimpleFallbackModel:
    """A simple fallback model using basic linear calculation"""
    def predict(self, features: List[List[float]]) -> np.ndarray:
        # Simple weighted average: GRE (30%), TOEFL (30%), CGPA (40%)
        predictions = []
        for feature in features:
            gre_norm = min(feature[0] / 340.0, 1.0) * 0.3
            toefl_norm = min(feature[1] / 120.0, 1.0) * 0.3
            cgpa_norm = min(feature[2] / 10.0, 1.0) * 0.4
            pred = gre_norm + toefl_norm + cgpa_norm
            predictions.append(pred)
        return np.array(predictions)

class AdmissionPredictionTool(BaseTool):
    name: str = "admission_predictor"
    description: str = "Predicts chance of admission based on student data. Required input: GRE_Score, TOEFL_Score, CGPA"
    
    def __init__(self, model_path=None):
        super().__init__()
        self.model = None
        
        try:
            if model_path is None:
                model_path = './src/model/admission_prediction_model_regressor.pkl'
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                print("Model file not found, using fallback model")
                self.model = SimpleFallbackModel()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Using fallback model instead")
            self.model = SimpleFallbackModel()

    def _run(self, input_data: Dict[str, Any]) -> str:
        try:
            # Extract required features
            features = [[
                float(input_data.get('GRE_Score', 0)),
                float(input_data.get('TOEFL_Score', 0)),
                float(input_data.get('CGPA', 0))
            ]]
            
            prediction = self.model.predict(features)
            chance = prediction[0] * 100  # Convert to percentage
            
            return f"Predicted chance of admission: {chance:.2f}%"
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")
