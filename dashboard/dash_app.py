"""
Main module for the Housing Data Dashboard.
This module creates and configures the Dash application.
"""

import os
import sys
import dash
from dash import html
import dash_bootstrap_components as dbc

# Import dashboard components
from dashboard.data_provider import DashboardDataProvider
from dashboard.layout import create_layout
from dashboard.callbacks import register_callbacks


def create_dash_app(data_path=None, debug=True):
    """
    Create and configure the Dash application.
    
    Args:
        data_path: Path to the housing data CSV file (optional)
        debug: Whether to run the app in debug mode
        
    Returns:
        Configured Dash application instance
    """
    # Create data provider
    data_provider = DashboardDataProvider(data_path)
    
    # Create Dash app with Bootstrap theme
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="Housing Data Dashboard"
    )
    
    # Configure layout
    app.layout = create_layout(data_provider)
    
    # Register callbacks
    register_callbacks(app, data_provider)
    
    return app


def run_dashboard(data_path=None, host="0.0.0.0", port=8050, debug=True):
    """
    Run the dashboard application.
    
    Args:
        data_path: Path to the housing data CSV file (optional)
        host: Host to run the server on
        port: Port to run the server on
        debug: Whether to run in debug mode
    """
    app = create_dash_app(data_path, debug)
    app.run(host=host, port=port, debug=debug)  # Updated from app.run_server() to app.run()


if __name__ == "__main__":
    # Check if data path is provided as argument
    data_path = None
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Run the dashboard
    run_dashboard(data_path)