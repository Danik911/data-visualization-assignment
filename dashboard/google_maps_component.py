"""
Google Maps component for the dashboard.
This module provides functions to create and manage Google Maps visualizations.
"""

import dash
from dash import html, dcc
import json
import os


def create_google_map(id="google-price-map", style=None):
    """
    Create a Google Maps component for Dash
    
    Args:
        id: ID for the component
        style: Additional CSS style
        
    Returns:
        HTML Div containing the map
    """
    default_style = {"width": "100%", "height": "600px", "border-radius": "8px"}
    if style:
        default_style.update(style)
        
    return html.Div(
        [
            # Map container where Google Maps will render
            html.Div(id=f"{id}-container", style=default_style),
            
            # Hidden div to store map data JSON
            html.Div(id=f"{id}-data", style={"display": "none"}),
            
            # Hidden div to store click data
            html.Div(id=f"{id}-click-data", style={"display": "none"}),
            
            # Debug div to help with troubleshooting
            html.Div(id=f"{id}-debug", style={"display": "none", "padding": "10px", "backgroundColor": "#f8f9fa", "margin": "10px 0"}),
            
            # Add a small loading indicator
            html.Div(
                dcc.Loading(
                    html.Div(id=f"{id}-loading-placeholder", style={"height": "10px"}),
                    type="circle",
                ),
                style={"position": "absolute", "top": "50%", "left": "50%", "transform": "translate(-50%, -50%)"}
            )
        ],
        id=id,
        style={"position": "relative", "min-height": "600px"}
    )


def register_google_map_callbacks(app, api_key):
    """
    Register clientside callbacks for Google Maps
    
    Args:
        app: Dash application
        api_key: Google Maps API key
    """
    # Add Google Maps script to head with async and defer attributes
    app.index_string = app.index_string.replace(
        '</head>',
        f'<script async defer src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places"></script></head>'
    )