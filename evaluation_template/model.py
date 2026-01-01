"""
Model Template for Football Match Prediction

This module contains a template Model class that can be extended
to implement actual prediction logic.
"""

import pandas as pd
from typing import List, Union


class Model:
    """
    Template Model class for football match prediction.
    
    This class should be extended with actual model loading and prediction logic.
    """
    
    def __init__(self):
        """
        Initialize the model.
        Override this method to load your trained model.
        """
        self.model = None
        print("Model initialized (template - no actual model loaded)")
    
    def predict(self, X: pd.DataFrame) -> List[str]:
        """
        Predict match results for given input data.
        
        Args:
            X (pd.DataFrame): Input features with columns:
                - Team Home
                - Team Away
                - Saison
                - Spieltag
                - Wochentag
                (Note: Ergebnis column should be removed before calling this)
        
        Returns:
            List[str]: List of predicted results in format "X:Y" (e.g., ["2:1", "0:0", "1:3"])
        """
        # TODO: Implement actual prediction logic
        # This is a template that returns dummy predictions
        # Replace this with your actual model prediction code
        
        n_samples = len(X)
        # Return dummy predictions (all draws as placeholder)
        return ["0:0"] * n_samples
    
    def load_model(self, model_path: str):
        """
        Load a trained model from file.
        
        Args:
            model_path (str): Path to the saved model file
        """
        # TODO: Implement model loading logic
        # Example:
        # import pickle
        # self.model = pickle.load(open(model_path, "rb"))
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model on training data.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training labels (Ergebnis column)
        """
        # TODO: Implement model training logic
        pass

