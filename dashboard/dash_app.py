"""
Main module for the Housing Data Dashboard.
This module creates and configures the Dash application.
"""

import os
import sys
import dash
from dash import html
import dash_bootstrap_components as dbc
import logging
import logging.config

# Import dashboard components
from dashboard.data_provider import DashboardDataProvider
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
        
    # Create data provider
    data_provider = DashboardDataProvider(data_path)
    
    # Get theme configuration
    theme = THEMES[ACTIVE_THEME]
    
    # Create Dash app with theme
    app = dash.Dash(
        __name__,
        external_stylesheets=theme["external_stylesheets"],
        suppress_callback_exceptions=True,
        title="Housing Data Dashboard",
        assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    )
    
    # Get Google Maps API key from environment or use a default for testing
    google_maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "AIzaSyDC0PSGje7Bi0kwtaTIrIwksyjgGc1cRgU")
    
    # Register Google Maps callbacks
    register_google_map_callbacks(app, google_maps_api_key)
    
    # Set up cache if needed
    if CACHE_CONFIG:
        from flask_caching import Cache
        cache = Cache()
        cache.init_app(app.server, config=CACHE_CONFIG)
        app.cache = cache
    
    # Configure layout
    app.layout = create_layout(data_provider)
    
    # Register callbacks
    register_callbacks(app, data_provider)
    
    logger.info("Dashboard application created successfully")
    return app


def run_dashboard(data_path=None, host=None, port=None, debug=None):
    """
    Run the dashboard application.
    
    Args:
        data_path: Path to the housing data CSV file (optional)
        host: Host to run the server on
        port: Port to run the server on
        debug: Whether to run in debug mode
    """
    # Use config values with fallbacks
    host = host or SERVER_HOST
    port = port or SERVER_PORT
    
    app = create_dash_app(data_path, debug)
    app.run(host=host, port=port, debug=debug)  # Updated from app.run_server() to app.run()


if __name__ == "__main__":
    # Check if data path is provided as argument
    data_path = None
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Run the dashboard
    run_dashboard(data_path)