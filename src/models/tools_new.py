"""
Tools for enrollment prediction using machine learning models.
"""
import numpy as np
import pandas as pd
from langchain.tools import tool
from typing import Dict, Any
import logging

from ..utils.model_manager import ModelManager
from ..data.processor import DataProcessor

# Initialize utilities
model_manager = ModelManager()
data_processor = DataProcessor()
logger = logging.getLogger(__name__)

@tool
def predict_enrollment(
    gre_score: float, 
    toefl_score: float, 
    sop: float, 
    lor: float, 
    cgpa: float
) -> str:
    """
    Predict chance of admission based on academic credentials using the best available model.
    
    Parameters:
        gre_score (float): GRE score of the applicant (130-170)
        toefl_score (float): TOEFL score of the applicant (0-120)
        sop (float): Statement of Purpose score (1-5)
        lor (float): Letter of Recommendation score (1-5)
        cgpa (float): CGPA of the applicant (0-10)
        
    Returns:
        str: Predicted chance of admission as a percentage with explanation
    """
    try:
        # Validate input data
        input_data = {
            'gre_score': gre_score,
            'toefl_score': toefl_score,
            'sop': sop,
            'lor': lor,
            'cgpa': cgpa
        }
        
        if not data_processor.validate_input_data(input_data):
            return "Invalid input data. Please check the ranges for each field."
        
        # Prepare input array
        input_array = np.array([[gre_score, toefl_score, sop, lor, cgpa]])
        
        # Try to use the enrollment model first
        prediction = model_manager.predict_with_model(input_array, "enrollment_model")
        
        if prediction is not None:
            chance_percentage = prediction * 100
            
            # Provide interpretation
            if chance_percentage >= 80:
                interpretation = "Excellent chances! Your profile is very strong."
            elif chance_percentage >= 60:
                interpretation = "Good chances. Consider strengthening weaker areas."
            elif chance_percentage >= 40:
                interpretation = "Moderate chances. Focus on improving test scores or CGPA."
            else:
                interpretation = "Lower chances. Significant improvement needed in multiple areas."
            
            return (f"🎓 Admission Prediction Results:\n\n"
                   f"Predicted Chance of Admission: {chance_percentage:.1f}%\n\n"
                   f"📊 Analysis:\n{interpretation}\n\n"
                   f"📈 Your Profile:\n"
                   f"• GRE Score: {gre_score}/170\n"
                   f"• TOEFL Score: {toefl_score}/120\n"
                   f"• Statement of Purpose: {sop}/5\n"
                   f"• Letter of Recommendation: {lor}/5\n"
                   f"• CGPA: {cgpa}/10\n\n"
                   f"💡 Tips for improvement:\n"
                   f"• GRE scores above 160 significantly improve chances\n"
                   f"• TOEFL scores above 100 are preferred\n"
                   f"• Strong SOP and LOR (4+) make a big difference\n"
                   f"• CGPA above 8.5 is considered excellent")
        
        # Fallback to CNN model
        prediction = model_manager.predict_with_model(input_array, "cnn_model")
        if prediction is not None:
            chance_percentage = prediction * 100
            return f"Predicted chance of admission (CNN model): {chance_percentage:.1f}%"
        
        # Final fallback - simple heuristic
        return _simple_prediction_heuristic(input_data)
        
    except Exception as e:
        logger.error(f"Error in predict_enrollment: {e}")
        return f"An error occurred during prediction: {str(e)}"

def _simple_prediction_heuristic(data: Dict[str, float]) -> str:
    """
    Simple rule-based prediction as a fallback when models are unavailable.
    """
    try:
        # Normalize scores to 0-1 scale
        gre_norm = (data['gre_score'] - 130) / 40  # 130-170 range
        toefl_norm = data['toefl_score'] / 120     # 0-120 range
        sop_norm = (data['sop'] - 1) / 4           # 1-5 range
        lor_norm = (data['lor'] - 1) / 4           # 1-5 range
        cgpa_norm = data['cgpa'] / 10              # 0-10 range
        
        # Weighted average (GRE and CGPA have higher weights)
        weights = [0.25, 0.15, 0.15, 0.15, 0.30]  # GRE, TOEFL, SOP, LOR, CGPA
        scores = [gre_norm, toefl_norm, sop_norm, lor_norm, cgpa_norm]
        
        weighted_score = sum(w * s for w, s in zip(weights, scores))
        chance_percentage = weighted_score * 100
        
        return (f"🎓 Admission Prediction (Heuristic Model):\n\n"
               f"Predicted Chance of Admission: {chance_percentage:.1f}%\n\n"
               f"Note: This is a simplified prediction. For more accurate results, "
               f"ensure the ML models are properly loaded.")
    
    except Exception as e:
        return f"Unable to make prediction: {str(e)}"

@tool
def get_admission_statistics() -> str:
    """
    Provide general statistics and insights about university admissions.
    
    Returns:
        str: Admission statistics and tips
    """
    return """📊 University Admission Statistics & Insights:

🏆 Typical Admission Ranges:
• GRE: 155-165 (competitive programs)
• TOEFL: 90-110 (international students)
• CGPA: 7.5-9.0 (on 10-point scale)
• SOP/LOR: 3.5-4.5 (strong applications)

📈 Factors that Improve Chances:
1. Strong academic performance (CGPA > 8.0)
2. High standardized test scores (GRE > 160, TOEFL > 100)
3. Well-written statement of purpose
4. Strong letters of recommendation
5. Relevant research/work experience
6. Extracurricular activities and leadership

💡 Pro Tips:
• Apply to a mix of reach, target, and safety schools
• Start preparing applications 6-12 months in advance
• Consider retaking standardized tests if scores are below target ranges
• Get feedback on your SOP from professors or advisors
• Choose recommenders who know your work well

🎯 Admission rates vary significantly by:
• Program competitiveness
• University ranking
• Application pool strength
• Available funding/seats"""

@tool
def compare_profiles(
    profile1_gre: float, profile1_toefl: float, profile1_cgpa: float,
    profile2_gre: float, profile2_toefl: float, profile2_cgpa: float
) -> str:
    """
    Compare two applicant profiles and provide insights.
    
    Parameters:
        profile1_gre, profile1_toefl, profile1_cgpa: First profile scores
        profile2_gre, profile2_toefl, profile2_cgpa: Second profile scores
        
    Returns:
        str: Comparison analysis
    """
    try:
        # Simple comparison logic
        profile1_score = (profile1_gre/170*0.4 + profile1_toefl/120*0.3 + profile1_cgpa/10*0.3) * 100
        profile2_score = (profile2_gre/170*0.4 + profile2_toefl/120*0.3 + profile2_cgpa/10*0.3) * 100
        
        stronger = "Profile 1" if profile1_score > profile2_score else "Profile 2"
        difference = abs(profile1_score - profile2_score)
        
        return f"""🔄 Profile Comparison Analysis:

Profile 1: GRE {profile1_gre}, TOEFL {profile1_toefl}, CGPA {profile1_cgpa}
Composite Score: {profile1_score:.1f}%

Profile 2: GRE {profile2_gre}, TOEFL {profile2_toefl}, CGPA {profile2_cgpa}
Composite Score: {profile2_score:.1f}%

📊 Result: {stronger} is stronger by {difference:.1f} points

🎯 Recommendations:
• Focus on improving the weakest components
• Consider the context of your target programs
• Remember that admission is holistic - test scores aren't everything!"""
        
    except Exception as e:
        return f"Error comparing profiles: {str(e)}"
