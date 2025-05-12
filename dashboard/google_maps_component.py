"""
Google Maps component for the dashboard.
This module provides functions to create and manage Google Maps visualizations.
"""

import dash
from dash import html, dcc, ClientsideFunction
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
        
    map_div_content = html.Div(
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

    # Add a div for the map status message
    status_message_div = html.Div(id=f"{id}-status-message", style={"textAlign": "center", "padding": "10px", "color": "green"})

    return html.Div([
        map_div_content, # The original map content
        status_message_div # The new status message div
    ])


def register_google_map_callbacks(app, api_key):
    """
    Register clientside callbacks for Google Maps
    
    Args:
        app: Dash application
        api_key: Google Maps API key
    """
    # Add Google Maps script to head with async and defer attributes
    # app.index_string = app.index_string.replace(
    #     '</head>',
    #     f'<script async defer src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places&loading=async"></script></head>'
    # )

    # Clientside callback to initialize/update the map when data changes
    app.clientside_callback(
        ClientsideFunction(
            namespace='googleMaps',
            function_name='initMap'
        ),
        dash.Output(f"{id_prefix}-status-message", "children"), # Changed from google-price-map-loading-placeholder
        [dash.Input(f"{id_prefix}-data", "children")],
        dash.State(f"{id_prefix}-container", "id")
    )

    # New clientside callback to clear the status message
    app.clientside_callback(
        ClientsideFunction(
            namespace='googleMaps',
            function_name='clearMessage'
        ),
        dash.Output(f"{id_prefix}-status-message", "children", allow_duplicate=True), # Keep existing output to avoid error
        [dash.Input(f"{id_prefix}-status-message", "children")],
        prevent_initial_call=True # Important: prevent_initial_call=True
    )

# Helper to get the id prefix, assuming your map id is like "google-price-map"
def get_id_prefix(map_component_id):
    if map_component_id.endswith("-container"):
        return map_component_id[:-len("-container")]
    return map_component_id

id_prefix = get_id_prefix("google-price-map") # Example usage, adjust as needed based on actual id