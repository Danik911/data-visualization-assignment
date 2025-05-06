"""
Configuration module for the data analysis agent.
This module provides a unified configuration interface to make the analysis code dataset-agnostic.
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigManager:
    """Configuration manager for the data analysis agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to a configuration file (JSON, YAML), optional
        """
        # Default configuration
        self.config = {
            # Thresholds for categorical data
            "categorical_threshold": 20,  # Columns with <= N unique values are considered categorical
            
            # Domain detection settings
            "domain_indicators": {
                "housing": ["price", "bedroom", "bathroom", "sqft", "garage", "house", "home", "property", 
                           "zip", "lot", "real estate", "tax", "built", "year", "style", "subdivision"],
                "healthcare": ["patient", "diagnosis", "symptom", "doctor", "hospital", "treatment", 
                              "medication", "dosage", "health", "medical", "clinical", "disease", "blood"],
                "finance": ["income", "expense", "revenue", "cost", "profit", "loss", "budget", "account", 
                           "transaction", "price", "sale", "total", "amount", "tax", "credit", "debit", "bank"],
                "transportation": ["distance", "duration", "travel", "time", "commute", "trip", "route", 
                                  "vehicle", "transport", "mileage", "fare", "cost", "speed", "gas", "fuel"],
                "manufacturing": ["product", "quantity", "factory", "assembly", "material", "inventory", 
                                 "cost", "quality", "defect", "output", "production", "efficiency", "machine"],
                "retail": ["sale", "product", "customer", "price", "discount", "quantity", "store", "order", 
                          "inventory", "retail", "item", "merchandise", "vendor", "stock", "revenue"]
            },
            
            # Target variable detection settings
            "target_indicators": {
                "housing": ["price", "value", "cost", "appraisal", "saleprice", "sold"],
                "healthcare": ["outcome", "mortality", "readmission", "survival", "recovery", "length of stay"],
                "finance": ["profit", "return", "yield", "revenue", "income", "loss", "growth", "value", "amount"],
                "transportation": ["time", "duration", "delay", "distance", "fare", "cost", "satisfaction"],
                "manufacturing": ["defect", "output", "productivity", "efficiency", "cost", "quality", "yield"],
                "retail": ["sales", "revenue", "profit", "units", "quantity", "transactions"]
            },
            
            # Visualization settings
            "visualization": {
                "color_palette": "Set2",
                "default_figsize": (10, 6),
                "save_path": "plots/",
                "dpi": 300,
                "fontsize": 12,
                "format": "png"
            },
            
            # Date detection patterns
            "date_detection_patterns": [
                "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", 
                "%d-%m-%Y", "%m-%d-%Y", "%b %d, %Y", "%d %b %Y"
            ],
            
            # Column type detection
            "geo_column_keywords": ["address", "street", "city", "state", "zip", "country", "latitude", "longitude", 
                                   "lat", "lon", "location", "postal", "geo"]
        }
        
        # Load custom configuration if provided
        if config_path:
            self._load_config(config_path)
        
        # Check for environment configuration
        env_config_path = os.getenv('DATA_ANALYSIS_CONFIG')
        if env_config_path:
            self._load_config(env_config_path)
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to a configuration file
        """
        try:
            path = Path(config_path)
            
            if not path.exists():
                print(f"[CONFIG] Configuration file not found: {config_path}")
                return
                
            if path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    custom_config = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(path, 'r') as f:
                        custom_config = yaml.safe_load(f)
                except ImportError:
                    print("[CONFIG] YAML support requires 'pyyaml' package. Using default configuration.")
                    return
            else:
                print(f"[CONFIG] Unsupported configuration format: {path.suffix}")
                return
            
            # Update configuration recursively
            self._update_dict_recursive(self.config, custom_config)
            print(f"[CONFIG] Loaded configuration from {config_path}")
            
        except Exception as e:
            print(f"[CONFIG] Error loading configuration: {str(e)}")
    
    def _update_dict_recursive(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Recursively update a dictionary.
        
        Args:
            base_dict: Dictionary to update
            update_dict: Dictionary with updated values
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._update_dict_recursive(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (can be nested using dots, e.g. 'visualization.color_palette')
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (can be nested using dots)
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value


# Global configuration instance
_config_manager = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager.
    
    Args:
        config_path: Optional path to a configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    elif config_path is not None:
        # If a config path is explicitly provided, reload config
        _config_manager = ConfigManager(config_path)
        
    return _config_manager


def update_config(key: str, value: Any) -> None:
    """
    Update a configuration value.
    
    Args:
        key: Configuration key
        value: New value
    """
    config = get_config()
    config.set(key, value)


def load_project_config(project_config_path: str) -> None:
    """
    Load a project-specific configuration.
    
    Args:
        project_config_path: Path to project configuration file
    """
    global _config_manager
    _config_manager = ConfigManager(project_config_path)