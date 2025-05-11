"""
Simplified data provider module for the dashboard.
This version doesn't depend on llama_index or the full data_analysis_agent.
"""

import os
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from io import StringIO  # Import StringIO for proper JSON handling

# Import the simplified helper
from simplified_pandas_helper import SimplifiedPandasHelper

# Add this utility function to the file (outside of any class)
def safe_read_json(json_str):
    """
    Safely read JSON string into pandas DataFrame, avoiding FutureWarning.
    
    Args:
        json_str: JSON string to read
        
    Returns:
        pandas DataFrame from the JSON
    """
    # Use StringIO to avoid FutureWarning
    return pd.read_json(StringIO(json_str))

class SimplifiedDataProvider:
    """
    Simplified data provider class for the dashboard.
    """
    def __init__(self, data_path: str = None):
        """
        Initialize the data provider.
        
        Args:
            data_path: Path to the housing data CSV file
        """
        self.data_path = data_path or "data/Housing Data.csv"
        self.df = None
        self.reports_dir = "reports"
        
        # Load data when initialized
        self.load_data()
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the housing data from CSV file.
        
        Returns:
            DataFrame with housing data
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded data with shape {self.df.shape}")
            # Initialize the simplified pandas helper
            self.pandas_helper = SimplifiedPandasHelper(self.df)
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return empty DataFrame if loading fails
            self.df = pd.DataFrame()
            self.pandas_helper = SimplifiedPandasHelper(self.df)
            return self.df
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded DataFrame.
        
        Returns:
            DataFrame with housing data
        """
        if self.df is None:
            self.load_data()
        return self.df
    
    def get_filtered_data(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the DataFrame and return the filtered result.
        
        Args:
            filters: Dictionary of column names and filter values
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            self.load_data()
            
        if not filters:
            return self.df
            
        filtered_df = self.df.copy()
        
        for column, filter_value in filters.items():
            if column not in filtered_df.columns:
                print(f"Warning: Column '{column}' not found in the dataframe")
                continue
                
            if isinstance(filter_value, list):
                # Handle list of values (OR condition)
                if column == 'Bldg_Type':
                    # Special handling for building types which might not match exactly
                    from dashboard.config import BUILDING_TYPE_LABELS
                    
                    # Create a dictionary mapping display labels back to actual values
                    reverse_mapping = {}
                    for db_value, display_label in BUILDING_TYPE_LABELS.items():
                        reverse_mapping[display_label] = db_value
                        # Also add the original value for direct matches
                        reverse_mapping[db_value] = db_value
                    
                    # Convert the filter values to database values
                    actual_filter_values = []
                    for val in filter_value:
                        # If the value is in our reverse mapping, use the database value
                        if val in reverse_mapping:
                            actual_filter_values.append(reverse_mapping[val])
                        else:
                            # Otherwise keep the original value
                            actual_filter_values.append(val)
                    
                    # Apply the filter using the actual database values
                    filtered_df = filtered_df[filtered_df[column].isin(actual_filter_values)]
                else:
                    # For other columns, apply filter normally
                    filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
            elif isinstance(filter_value, dict) and "range" in filter_value:
                # Handle range filters
                min_val, max_val = filter_value["range"]
                filtered_df = filtered_df[(filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)]
            else:
                # Handle exact match
                filtered_df = filtered_df[filtered_df[column] == filter_value]
                
        return filtered_df
    
    def get_column_options(self) -> Dict[str, List[Any]]:
        """
        Get column options for dropdowns and filters.
        
        Returns:
            Dictionary mapping column names to their unique values
        """
        if self.df is None:
            self.load_data()
            
        if self.df.empty:
            return {}
        
        options = {}
        
        # Get categorical column options
        categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns
        for col in categorical_cols:
            unique_values = self.df[col].dropna().unique().tolist()
            if len(unique_values) <= 50:  # Only include columns with a reasonable number of options
                options[col] = unique_values
        
        # Get binned numeric column options
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Include min, max for numeric columns
            if len(self.df[col].dropna()) > 0:
                options[col] = {
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max())
                }
        
        return options
        
    def get_regression_models(self):
        """
        Returns simplified dummy regression models for the dashboard.
        This is a simplified version that returns static placeholder data.
        """
        # Return a simplified dummy regression model data
        return {
            "models": [
                {
                    "name": "Price vs. Area",
                    "x": "Lot_Area",
                    "y": "Sale_Price",
                    "r2": 0.45,
                    "description": "Simplified model showing relationship between lot area and sale price"
                },
                {
                    "name": "Price vs. Year Built",
                    "x": "Year_Built",
                    "y": "Sale_Price",
                    "r2": 0.38,
                    "description": "Simplified model showing relationship between year built and sale price"
                }
            ]
        } 