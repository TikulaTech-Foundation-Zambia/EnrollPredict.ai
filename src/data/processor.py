"""
Data processing utilities for the EnrollPredict.ai application.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

class DataProcessor:
    """Class for handling data processing operations."""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.base_dir / "data"
        
    def load_enrollment_data(self, filename: str = "dataset.json") -> Optional[pd.DataFrame]:
        """
        Load enrollment data from JSON file.
        
        Args:
            filename: Name of the JSON file to load
            
        Returns:
            DataFrame with enrollment data or None if error
        """
        try:
            file_path = self.data_dir / "datasets" / filename
            with open(file_path, 'r') as file:
                data = json.load(file)
                df = pd.DataFrame(data['enrollment_data'])
                print(f"Successfully loaded data from {filename}")
                return df
        except FileNotFoundError:
            print(f"Error: {filename} not found in data/datasets/")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {filename}")
            return None
        except KeyError:
            print(f"Error: 'enrollment_data' key not found in {filename}")
            return None
    
    def preprocess_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features for model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, targets)
        """
        feature_columns = ['gre_score', 'toefl_score', 'sop', 'lor', 'cgpa']
        target_column = 'chance_of_admit'
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        return X, y
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> bool:
        """
        Save processed data to the processed directory.
        
        Args:
            df: DataFrame to save
            filename: Name of the output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = self.data_dir / "processed" / filename
            df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving processed data: {e}")
            return False
    
    def validate_input_data(self, data: Dict[str, float]) -> bool:
        """
        Validate input data for prediction.
        
        Args:
            data: Dictionary with input features
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['gre_score', 'toefl_score', 'sop', 'lor', 'cgpa']
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False
        
        # Validate ranges
        validations = {
            'gre_score': (130, 170),
            'toefl_score': (0, 120),
            'sop': (1, 5),
            'lor': (1, 5),
            'cgpa': (0, 10)
        }
        
        for field, (min_val, max_val) in validations.items():
            value = data[field]
            if not (min_val <= value <= max_val):
                print(f"Invalid value for {field}: {value}. Must be between {min_val} and {max_val}")
                return False
        
        return True
