"""
Data provider module for the dashboard.
This module interfaces with the existing data analysis components to retrieve data for visualization.
"""

import os
import sys
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path to import from data_analysis_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_analysis_agent.pandas_helper import PandasHelper


class DashboardDataProvider:
    """
    Data provider class for the dashboard that interfaces with the existing data analysis components.
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
        self.analysis_report = None
        self.statistical_report = None
        self.regression_report = None
        
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
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return empty DataFrame if loading fails
            self.df = pd.DataFrame()
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
                    
                    print(f"Converted building type filter from UI: {filter_value}")
                    print(f"To actual database values: {actual_filter_values}")
                    
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
    
    def load_report(self, report_name: str) -> Dict[str, Any]:
        """
        Load a report from the reports directory.
        
        Args:
            report_name: Name of the report file (without path or extension)
            
        Returns:
            Dictionary containing the report data
        """
        report_path = os.path.join(self.reports_dir, f"{report_name}.json")
        
        try:
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            return report_data
        except Exception as e:
            print(f"Error loading report {report_name}: {e}")
            return {}
    
    def get_dataset_analysis(self) -> Dict[str, Any]:
        """
        Get the dataset analysis report.
        
        Returns:
            Dictionary containing dataset analysis data
        """
        if self.analysis_report is None:
            self.analysis_report = self.load_report("dataset_analysis")
        return self.analysis_report
    
    def get_statistical_analysis(self) -> Dict[str, Any]:
        """
        Get the statistical analysis report.
        
        Returns:
            Dictionary containing statistical analysis data
        """
        if self.statistical_report is None:
            self.statistical_report = self.load_report("statistical_analysis_report")
        return self.statistical_report
    
    def get_regression_models(self) -> Dict[str, Any]:
        """
        Get the regression models report.
        
        Returns:
            Dictionary containing regression models data
        """
        if self.regression_report is None:
            self.regression_report = self.load_report("regression_models")
        return self.regression_report
    
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