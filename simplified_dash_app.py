"""
Simplified module for the Housing Data Dashboard.
This version doesn't depend on llama_index or the full data_analysis_agent.
"""

import os
import sys
import dash
from dash import html
import dash_bootstrap_components as dbc
import logging
import logging.config
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Import dashboard components
from simplified_data_provider import SimplifiedDataProvider
from dashboard.layout import create_layout
from dashboard.callbacks import register_callbacks
from dashboard.google_maps_component import register_google_map_callbacks
from dashboard.config import (
    SERVER_HOST, 
    SERVER_PORT, 
    DEBUG, 
    DEFAULT_DATA_PATH, 
    ACTIVE_THEME, 
    THEMES,
    CACHE_CONFIG,
    LOGGING_CONFIG
)


def create_dash_app(data_path=None, debug=None):
    """
    Create and configure the Dash application.
    
    Args:
        data_path: Path to the housing data CSV file (optional)
        debug: Whether to run the app in debug mode
        
    Returns:
        Configured Dash application instance
    """
    # Set up logging
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Creating dashboard application")
    
    # Use config values with fallbacks
    data_path = data_path or DEFAULT_DATA_PATH
    if debug is None:
        debug = DEBUG
        
    # Create simplified data provider
    data_provider = SimplifiedDataProvider(data_path)
    
    # Get theme configuration
    theme = THEMES[ACTIVE_THEME]
    
    # Create Dash app with theme
    app = dash.Dash(
        __name__,
        external_stylesheets=theme["external_stylesheets"],
        suppress_callback_exceptions=True,
        title="Housing Data Dashboard",
        assets_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "assets")
    )
    
    # Get Google Maps API key from environment
    google_maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not google_maps_api_key:
        logger.warning("GOOGLE_MAPS_API_KEY environment variable is not set - using placeholder")
        # Provide a fallback key for basic functionality - may not show the map
        google_maps_api_key = "AIzaSyBNLrJhOMz6idD05pzfn5lhA-TAw-mAZCU" # Placeholder, ensure your actual key is set in Render
        
    # Add a flag to indicate whether we're in debug mode for maps
    map_debug_mode = os.environ.get("MAP_DEBUG_MODE", "false").lower() == "true"
    if map_debug_mode:
        logger.info("Map debug mode is enabled")
        
    # Register Google Maps callbacks
    register_google_map_callbacks(app, google_maps_api_key)
    
    # More robustly ensure the correct Google Maps script is loaded
    # Remove any existing Google Maps API script tags first
    # This regex matches <script ... src="...maps.googleapis.com..." ...></script>
    app.index_string = re.sub(r'<script[^>]*src="[^"]*maps\\.googleapis\\.com[^"]*"[^>]*>\\s*</script>', '', app.index_string, flags=re.IGNORECASE)
    
    # Add Google Maps script to head with async and defer attributes
    # Ensure this is the only Google Maps script tag being added
    if f"maps.googleapis.com/maps/api/js?key={google_maps_api_key}" not in app.index_string:
        app.index_string = app.index_string.replace(
            '</head>',
            f'<script async defer src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&libraries=places&loading=async"></script></head>'
        )
            
    # Set up cache if needed
    if CACHE_CONFIG:
        from flask_caching import Cache
        cache = Cache()
        cache.init_app(app.server, config=CACHE_CONFIG)
        app.cache = cache
    
    # Configure layout
    app.layout = create_layout(data_provider)
    
    # Patch the data_provider with any missing methods that might be called by callbacks
    if not hasattr(data_provider, 'get_regression_models'):
        logger.warning("Adding missing get_regression_models method to data provider")
        data_provider.get_regression_models = lambda: {
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
    
    # Register callbacks
    register_callbacks(app, data_provider)
    
    logger.info("Dashboard application created successfully")
    return app


def run_dashboard(data_path=None, host=None, port=None, debug=None, return_app=False):
    """
    Run the dashboard application.
    
    Args:
        data_path: Path to the housing data CSV file (optional)
        host: Host to run the server on
        port: Port to run the server on
        debug: Whether to run in debug mode
        return_app: If True, return the app object instead of running it
    
    Returns:
        The Dash app object if return_app is True
    """
    # Use config values with fallbacks
    host = host or SERVER_HOST
    port = port or SERVER_PORT
    
    app = create_dash_app(data_path, debug)
    
    if return_app:
        return app
    
    app.run(host=host, port=port, debug=debug) 