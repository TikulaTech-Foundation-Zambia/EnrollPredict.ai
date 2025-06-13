"""
Model management utilities for the EnrollPredict.ai application.
"""
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Class for managing ML models and their artifacts."""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.models_dir = self.base_dir / "models"
        self.config_path = self.models_dir / "configs" / "model_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Model config not found at {self.config_path}")
            return {}
    
    def load_scaler(self, scaler_type: str = "input") -> Optional[Any]:
        """
        Load a preprocessing scaler.
        
        Args:
            scaler_type: Type of scaler ("input" or "output")
            
        Returns:
            Loaded scaler or None if not found
        """
        try:
            if scaler_type == "input":
                scaler_path = self.base_dir / self.config["scalers"]["input_scaler"]
            else:
                scaler_path = self.base_dir / self.config["scalers"]["output_scaler"]
            
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded {scaler_type} scaler from {scaler_path}")
            return scaler
        except Exception as e:
            logger.error(f"Error loading {scaler_type} scaler: {e}")
            return None
    
    def load_keras_model(self, model_name: str) -> Optional[Any]:
        """
        Load a Keras model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model or None if not found
        """
        try:
            import tensorflow as tf
            
            if model_name not in self.config["model_config"]:
                logger.error(f"Model {model_name} not found in config")
                return None
            
            model_path = self.base_dir / self.config["model_config"][model_name]["path"]
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded {model_name} from {model_path}")
            return model
        except ImportError:
            logger.error("TensorFlow not available")
            return None
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            return None
    
    def preprocess_input(self, input_data: np.ndarray, model_type: str = "default") -> Optional[np.ndarray]:
        """
        Preprocess input data for a specific model type.
        
        Args:
            input_data: Raw input data
            model_type: Type of model ("cnn", "lstm", "default")
            
        Returns:
            Preprocessed data or None if error
        """
        try:
            scaler = self.load_scaler("input")
            if scaler is None:
                return None
            
            # Scale the data
            scaled_data = scaler.transform(input_data)
            
            # Reshape based on model type
            if model_type == "cnn":
                # CNN expects (samples, features, 1)
                return scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
            elif model_type == "lstm":
                # LSTM expects (samples, timesteps, features)
                return scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])
            else:
                # Default flat structure
                return scaled_data
                
        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            return None
    
    def predict_with_model(self, input_data: np.ndarray, model_name: str) -> Optional[float]:
        """
        Make prediction with a specific model.
        
        Args:
            input_data: Input features
            model_name: Name of the model to use
            
        Returns:
            Prediction value or None if error
        """
        try:
            model = self.load_keras_model(model_name)
            if model is None:
                return None
            
            # Determine model type for preprocessing
            model_type = "default"
            if "cnn" in model_name.lower():
                model_type = "cnn"
            elif "lstm" in model_name.lower():
                model_type = "lstm"
            
            # Preprocess input
            processed_input = self.preprocess_input(input_data, model_type)
            if processed_input is None:
                return None
            
            # Make prediction
            prediction = model.predict(processed_input, verbose=0)
            
            # Return the prediction (assuming single output)
            return float(prediction[0][0])
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {e}")
            return None
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name in self.config["model_config"]:
            return self.config["model_config"][model_name]
        return {}
    
    def list_available_models(self) -> list:
        """Get list of available models."""
        return list(self.config.get("model_config", {}).keys())
