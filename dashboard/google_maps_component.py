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
    # Add Google Maps script to head
    app.index_string = app.index_string.replace(
        '</head>',
        f'<script src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places"></script></head>'
    )
    
    # Simple callback to store the data in the hidden div
    app.clientside_callback(
        """
        function(map_data) {
            console.log('Map data received by clientside callback');
            return map_data;
        }
        """,
        dash.dependencies.Output('google-price-map-data', 'children'),
        dash.dependencies.Input('google-price-map-data', 'children')
    )
    
    # Remove loading indicator once map data is available
    app.clientside_callback(
        """
        function(map_data) {
            return map_data ? "" : null;
        }
        """,
        dash.dependencies.Output('google-price-map-loading-placeholder', 'children'),
        dash.dependencies.Input('google-price-map-data', 'children')
    )
    
    # Load Google Maps JavaScript API and initialize map
    app.clientside_callback(
        """
        function(map_data) {
            if (!map_data) {
                console.log('No map data provided');
                return window.dash_clientside.no_update;
            }
            
            try {
                const data = JSON.parse(map_data);
                if (!data.data || !data.center) {
                    console.log('Invalid map data structure:', data);
                    return window.dash_clientside.no_update;
                }
                
                // Initialize map if it doesn't exist yet
                if (!window.googleMap) {
                    console.log('Initializing Google Map');
                    const mapContainer = document.getElementById('google-price-map-container');
                    if (!mapContainer) {
                        console.error('Map container not found: google-price-map-container');
                        return window.dash_clientside.no_update;
                    }
                    
                    window.googleMap = new google.maps.Map(mapContainer, {
                        center: data.center,
                        zoom: data.zoom || 12,
                        mapTypeId: google.maps.MapTypeId.ROADMAP,
                        mapTypeControl: true,
                        streetViewControl: true,
                        fullscreenControl: true
                    });
                    
                    // Create info window for markers
                    window.infoWindow = new google.maps.InfoWindow();
                    
                    // Store markers
                    window.mapMarkers = [];
                }
                
                // Clear existing markers
                if (window.mapMarkers) {
                    for (let marker of window.mapMarkers) {
                        marker.setMap(null);
                    }
                    window.mapMarkers = [];
                }
                
                // Price range for color calculation
                const prices = data.data.map(item => item.Sale_Price);
                const minPrice = Math.min(...prices);
                const maxPrice = Math.max(...prices);
                const priceRange = maxPrice - minPrice;
                
                console.log(`Adding ${data.data.length} markers to map`);
                
                // Add markers for each property
                for (let property of data.data) {
                    const normalizedPrice = (property.Sale_Price - minPrice) / priceRange;
                    
                    // Color gradient from green (low) to red (high)
                    const hue = (1 - normalizedPrice) * 120;
                    const color = `hsl(${hue}, 100%, 50%)`;
                    
                    const marker = new google.maps.Marker({
                        position: {
                            lat: property.Latitude,
                            lng: property.Longitude
                        },
                        map: window.googleMap,
                        title: `$${property.Sale_Price.toLocaleString()}`,
                        icon: {
                            path: google.maps.SymbolPath.CIRCLE,
                            fillColor: color,
                            fillOpacity: 0.7,
                            strokeColor: 'white',
                            strokeWeight: 1,
                            scale: 10 + (normalizedPrice * 10) // Size 10-20 based on price
                        }
                    });
                    
                    // Create info window content
                    const contentString = `
                        <div style="padding: 8px;">
                            <h5 style="margin-top: 0;">$${property.Sale_Price.toLocaleString()}</h5>
                            ${property.Bldg_Type ? `<p><b>Type:</b> ${property.Bldg_Type}</p>` : ''}
                            ${property.Year_Built ? `<p><b>Year Built:</b> ${property.Year_Built}</p>` : ''}
                            <p><b>Location:</b> ${property.Latitude.toFixed(6)}, ${property.Longitude.toFixed(6)}</p>
                        </div>
                    `;
                    
                    // Add click event listener
                    marker.addListener('click', () => {
                        window.infoWindow.setContent(contentString);
                        window.infoWindow.open(window.googleMap, marker);
                        
                        // Store click data
                        const clickData = {
                            id: property.id || 0,
                            lat: property.Latitude,
                            lng: property.Longitude,
                            price: property.Sale_Price
                        };
                        if (document.getElementById('google-price-map-click-data')) {
                            document.getElementById('google-price-map-click-data').textContent = JSON.stringify(clickData);
                        }
                    });
                    
                    window.mapMarkers.push(marker);
                }
                
                // Create legend for price colors
                if (!window.mapLegend) {
                    const legend = document.createElement('div');
                    legend.style.backgroundColor = 'white';
                    legend.style.padding = '10px';
                    legend.style.margin = '10px';
                    legend.style.borderRadius = '5px';
                    legend.style.boxShadow = '0 2px 6px rgba(0,0,0,.3)';
                    
                    const title = document.createElement('h4');
                    title.textContent = 'Price Legend';
                    title.style.margin = '0 0 8px';
                    legend.appendChild(title);
                    
                    // Add legend items
                    const steps = 5;
                    for (let i = 0; i < steps; i++) {
                        const item = document.createElement('div');
                        item.style.display = 'flex';
                        item.style.alignItems = 'center';
                        item.style.marginBottom = '5px';
                        
                        const normalizedValue = i / (steps - 1);
                        const hue = (1 - normalizedValue) * 120;
                        const color = `hsl(${hue}, 100%, 50%)`;
                        
                        const colorBox = document.createElement('div');
                        colorBox.style.width = '20px';
                        colorBox.style.height = '20px';
                        colorBox.style.backgroundColor = color;
                        colorBox.style.marginRight = '8px';
                        colorBox.style.border = '1px solid #ccc';
                        item.appendChild(colorBox);
                        
                        const price = minPrice + (priceRange * normalizedValue);
                        const label = document.createElement('span');
                        label.textContent = `$${Math.round(price).toLocaleString()}`;
                        item.appendChild(label);
                        
                        legend.appendChild(item);
                    }
                    
                    window.mapLegend = legend;
                    window.googleMap.controls[google.maps.ControlPosition.RIGHT_BOTTOM].push(legend);
                }
                
                // Center map
                window.googleMap.setCenter(data.center);
                console.log('Map updated successfully');
                
                return window.dash_clientside.no_update;
            } catch (error) {
                console.error('Error rendering Google Maps:', error);
                return window.dash_clientside.no_update;
            }
        }
        """,
        dash.dependencies.Output('google-price-map-container', 'children'),
        dash.dependencies.Input('google-price-map-data', 'children')
    )