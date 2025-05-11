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
    # app.index_string = app.index_string.replace(
    #     '</head>',
    #     f'<script async defer src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places&loading=async"></script></head>'
    # )

    # Clientside callback to initialize/update the map when data changes
    app.clientside_callback(
        """
        function(data) {
            if (window.initGoogleMap) {
                // Assuming 'google-price-map' is the main ID for create_google_map
                const mapContainerId = 'google-price-map-container';
                console.log('RENDER_DEBUG: Clientside callback triggered. Data type:', typeof data, 'Container ID:', mapContainerId);
                if (data) { // Check if data is not null or undefined
                    let parsedData = data;
                    if (typeof data === 'string') {
                        if (data.length === 0) {
                             console.log('RENDER_DEBUG: Clientside callback - Empty string data received.');
                             return ''; // Do nothing if empty string
                        }
                        try {
                            parsedData = JSON.parse(data);
                        } catch (e) {
                            console.error('RENDER_DEBUG: Clientside callback - Error parsing string data:', e, 'Raw data:', data.substring(0,100));
                            return ''; // Return empty string for the dummy output
                        }
                    }
                    // Ensure parsedData has the expected structure (e.g., a .data property if your JS expects it)
                    // The initGoogleMap in google_maps_init.js expects an object, often with a 'data' key and a 'center' key
                    window.initGoogleMap(parsedData, mapContainerId);
                } else {
                    console.log('RENDER_DEBUG: Clientside callback - Null or undefined data received.');
                }
            } else {
                console.error('RENDER_DEBUG: window.initGoogleMap is not defined. Check google_maps_init.js.');
            }
            return ''; // Dummy return for the output
        }
        """,
        dash.Output("google-price-map-loading-placeholder", "children"),  # Corrected to dash.Output
        [dash.Input("google-price-map-data", "children")]  # Corrected to dash.Input
    )