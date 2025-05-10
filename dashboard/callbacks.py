"""
Callbacks module for the dashboard.
This module handles all interactive elements and user interactions.
"""

from dash import Input, Output, State, callback, dash_table, html, ctx, dcc
import pandas as pd
import json
from typing import Dict, List, Any, Optional
import logging
import time

# Import visualization functions for generating plots
from dashboard.visualizations import (
    generate_price_map,
    generate_price_distribution,
    generate_correlation_heatmap,
    generate_scatter_plot,
    generate_box_plot,
    generate_time_series,
    generate_feature_importance,
    generate_parallel_coordinates,
    generate_summary_cards,
    generate_property_comparisons,  # New function for property comparisons
    generate_year_trend_analysis,   # New function for year trend analysis
    generate_google_price_map       # New function for Google Maps integration
)

logger = logging.getLogger(__name__)

def register_callbacks(app, data_provider):
    """
    Register all dashboard callbacks.
    
    Args:
        app: Dash app instance
        data_provider: DashboardDataProvider instance
    """
    # Setup dropdown options for Property Analysis tab
    @callback(
        Output("scatter-x-axis", "options"),
        Output("scatter-y-axis", "options"),
        Output("scatter-color", "options"),
        Output("box-plot-numeric", "options"),
        Output("box-plot-category", "options"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_dropdown_options(active_tab):
        # Only update when Property Analysis tab is active
        if active_tab != "tab-property":
            return [], [], [], [], []
        
        # Get data from provider
        df = data_provider.get_data()
        
        # Create options for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_options = [{"label": col.replace('_', ' '), "value": col} for col in numeric_cols]
        
        # Create options for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        categorical_options = [{"label": col.replace('_', ' '), "value": col} for col in categorical_cols]
        
        # All columns for color dropdown
        all_options = [{"label": col.replace('_', ' '), "value": col} for col in df.columns]
        
        return numeric_options, numeric_options, all_options, numeric_options, categorical_options
    
    # Setup default values for Property Analysis dropdowns
    @callback(
        Output("scatter-x-axis", "value"),
        Output("scatter-y-axis", "value"),
        Output("scatter-color", "value"),
        Output("box-plot-numeric", "value"),
        Output("box-plot-category", "value"),
        Input("scatter-x-axis", "options"),
        Input("scatter-y-axis", "options"),
        Input("box-plot-numeric", "options"),
        Input("box-plot-category", "options")
    )
    def set_default_dropdown_values(scatter_x_options, scatter_y_options, numeric_options, categorical_options):
        scatter_x = "Lot_Area" if scatter_x_options and any(opt["value"] == "Lot_Area" for opt in scatter_x_options) else None
        scatter_y = "Sale_Price" if scatter_y_options and any(opt["value"] == "Sale_Price" for opt in scatter_y_options) else None
        color = "Bldg_Type" if scatter_y_options and any(opt["value"] == "Bldg_Type" for opt in scatter_y_options) else None
        
        box_numeric = "Sale_Price" if numeric_options and any(opt["value"] == "Sale_Price" for opt in numeric_options) else None
        box_category = "Bldg_Type" if categorical_options and any(opt["value"] == "Bldg_Type" for opt in categorical_options) else None
        
        return scatter_x, scatter_y, color, box_numeric, box_category
    
    # Apply filters and store filtered dataframe
    @callback(
        Output("filtered-data-store", "data"),
        Input("building-type-filter", "value"),
        Input("price-range-filter", "value"),
        Input("area-range-filter", "value"),
        Input("reset-filters-button", "n_clicks"),
        prevent_initial_call=True  # Added to prevent initial duplicate callbacks
    )
    def filter_data(building_types, price_range, area_range, reset_clicks):
        # Create filters dictionary
        filters = {}
        
        # Debug filter values received from UI
        print("==== FILTER DEBUG ====")
        print(f"Building types from dropdown: {building_types}")
        if building_types:
            print(f"Building type value type: {type(building_types)}")
            if isinstance(building_types, list):
                for bt in building_types:
                    print(f"  - '{bt}' (type: {type(bt)})")
            
        # Only add filters if they have values
        if building_types:
            filters["Bldg_Type"] = building_types
            print(f"Applied Building Type filter: {building_types}")
            
        if price_range:
            filters["Sale_Price"] = {"range": price_range}
            print(f"Applied Price Range filter: {price_range}")
            
        if area_range:
            filters["Lot_Area"] = {"range": area_range}
            print(f"Applied Lot Area filter: {area_range}")
        
        # Reset filters on button click (context triggered)
        if ctx.triggered and "reset-filters-button" in ctx.triggered[0]["prop_id"]:
            filters = {}
            print("Reset all filters")
        
        # Get filtered data from provider
        filtered_df = data_provider.get_filtered_data(filters)
        
        # Debug dataset after filtering
        if building_types:
            print("=== FILTERED DATASET DEBUG ===")
            print(f"Original Bldg_Type values in data:")
            bldg_counts = data_provider.get_data()['Bldg_Type'].value_counts().to_dict()
            for bt, count in bldg_counts.items():
                print(f"  - '{bt}': {count} records")
            
            print(f"Filtered dataset Bldg_Type values:")
            if 'Bldg_Type' in filtered_df.columns:
                filtered_counts = filtered_df['Bldg_Type'].value_counts().to_dict()
                for bt, count in filtered_counts.items():
                    print(f"  - '{bt}': {count} records")
            
            # Check if specific building type exists
            if isinstance(building_types, list) and len(building_types) > 0:
                for bt in building_types:
                    matching = filtered_df[filtered_df['Bldg_Type'] == bt]
                    print(f"Records matching exactly '{bt}': {len(matching)}")
        
        # Log filter results
        active_filter_count = len(filters)
        print(f"Applied {active_filter_count} filters, resulting in {len(filtered_df)} records")
        
        # Convert to JSON for store
        filtered_json = filtered_df.to_json(date_format='iso', orient='split')
        print(f"Generated filtered data JSON (length: {len(filtered_json)})")
        
        # Return only the filtered data 
        return filtered_json
    
    # Update summary cards based on filtered data
    @callback(
        Output("summary-cards-row", "children"),
        Input("filtered-data-store", "data")
    )
    def update_summary_cards(filtered_data_json):
        # Check if filtered_data_json is None
        if filtered_data_json is None:
            # Return empty summary cards when there's no data
            from dashboard.layout import create_summary_cards
            empty_summary = {
                "total_properties": {"value": 0, "description": "Total Properties"},
                "avg_price": {"value": "$0", "description": "Average Price"},
                "median_price": {"value": "$0", "description": "Median Price"},
                "price_range": {"value": "$0 - $0", "description": "Price Range"},
                "avg_area": {"value": "0 sq.ft", "description": "Average Lot Area"},
                "common_type": {"value": "None (0.0%)", "description": "Most Common Building Type"}
            }
            return create_summary_cards(empty_summary)
        
        # Continue with normal processing if we have data
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')

        # ADD THIS CHECK FOR EMPTY DATAFRAME
        if filtered_df.empty:
            from dashboard.layout import create_summary_cards
            empty_summary = {
                "total_properties": {"value": 0, "description": "Total Properties"},
                "avg_price": {"value": "N/A", "description": "Average Price"},
                "median_price": {"value": "N/A", "description": "Median Price"},
                "price_range": {"value": "N/A", "description": "Price Range"},
                "avg_area": {"value": "N/A", "description": "Average Lot Area"},
                "common_type": {"value": "N/A", "description": "Most Common Building Type"}
            }
            return create_summary_cards(empty_summary)
        
        # Generate summary statistics
        summary_data = generate_summary_cards(filtered_df)
        
        # Import layout module here to avoid circular import
        from dashboard.layout import create_summary_cards
        return create_summary_cards(summary_data)
    
    # Update Overview tab visualizations
    @callback(
        Output("google-price-map-data", "children", allow_duplicate=True),
        Output("price-distribution", "figure"),
        Output("feature-importance", "figure"),
        Output("building-type-distribution", "figure"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab"),
        prevent_initial_call='initial_duplicate'
    )
    def update_overview_visualizations(filtered_data_json, active_tab):
        # ADD IMPORT TIME HERE
        import time

        # Check if this is a triggered update from another callback
        if ctx.triggered and "filtered-data-store" in ctx.triggered[0]["prop_id"]:
            print("Map update triggered by filter change")
            
        # Default empty figure for when tab is not active or data is missing
        empty_fig = {"data": [], "layout": {}}
        no_data_fig = {"data": [], "layout": {"title": "No property data available for the current filter criteria"}}

        # Only update when Overview tab is active
        if active_tab != "tab-overview":
            # WHEN TAB IS NOT ACTIVE, send an empty map structure but indicate no tab activation
            empty_map_data_inactive_tab = {
                'data': [], 'center': {'lat': 0, 'lng': 0}, 'zoom': 2, 
                'filter_change': False, # Or True, depending on desired behavior for data persistency
                'timestamp': time.time(),
                'tab_activated_timestamp': None # Explicitly None
            }
            return json.dumps(empty_map_data_inactive_tab), empty_fig, empty_fig, empty_fig
            
        # Check if filtered_data_json is None or represents an empty DataFrame
        if filtered_data_json is None:
            return None, no_data_fig, no_data_fig, no_data_fig
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        print(f"Processing map data with {len(filtered_df)} records")

        # Handle empty DataFrame for chart generation
        if filtered_df.empty:
            # For the map data, we still need to send a valid structure even if empty
            empty_map_data = {
                'data': [],
                'center': {'lat': 0, 'lng': 0}, # Default center
                'zoom': 2, # Default zoom
                'filter_change': True,
                'timestamp': time.time(),
                'tab_activated_timestamp': time.time(),
                'error': 'No data for selected filters.'
            }
            return json.dumps(empty_map_data), no_data_fig, no_data_fig, no_data_fig
        
        # Generate visualizations
        try:
            # Check if dataframe has required columns
            required_columns = ['Latitude', 'Longitude', 'Sale_Price']
            missing_columns = [col for col in required_columns if col not in filtered_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns for map: {', '.join(missing_columns)}")
                
            # Check if dataframe has non-null values
            if filtered_df[required_columns].isnull().any().any():
                # Clean the dataframe by dropping rows with null values in required columns
                filtered_df = filtered_df.dropna(subset=required_columns)
                print(f"Dropped rows with null values. Remaining rows: {len(filtered_df)}")
                
                if len(filtered_df) == 0:
                    raise ValueError("No valid data points after cleaning null values")
            
            # Generate Google Maps data
            print("Generating map data...")
            price_map_data = generate_google_price_map(filtered_df)
            
            # Add filter change flag to ensure complete map regeneration
            price_map_data["filter_change"] = True
            price_map_data["tab_activated_timestamp"] = time.time()
            
            # Ensure price_map_data has the required structure
            if not isinstance(price_map_data, dict) or 'data' not in price_map_data or 'center' not in price_map_data:
                raise ValueError(f"Invalid map data structure: {price_map_data.keys() if isinstance(price_map_data, dict) else 'not a dict'}")

            # ==> ADD DETAILED LOGGING HERE <==
            logger.debug(f"MAP_DEBUG: Pre-serialization: price_map_data type: {type(price_map_data)}")
            if isinstance(price_map_data.get('data'), list):
                logger.debug(f"MAP_DEBUG: Pre-serialization: price_map_data['data'] length: {len(price_map_data['data'])}")
                logger.debug(f"MAP_DEBUG: Pre-serialization: price_map_data['data'] sample (first 2): {price_map_data['data'][:2]}")
            else:
                logger.debug(f"MAP_DEBUG: Pre-serialization: price_map_data['data'] is not a list or not found. Value: {price_map_data.get('data')}")
            logger.debug(f"MAP_DEBUG: Pre-serialization: price_map_data center: {price_map_data.get('center')}")
            logger.debug(f"MAP_DEBUG: Pre-serialization: price_map_data full (first 500 chars of string): {str(price_map_data)[:500]}")
            
            # Convert to JSON for passing to the frontend
            price_map_json = json.dumps(price_map_data)
            # ==> ADD LOGGING FOR SERIALIZED JSON <==
            logger.debug(f"MAP_DEBUG: Serialized JSON length: {len(price_map_json)}")
            logger.debug(f"MAP_DEBUG: Serialized JSON (first 300 chars): {price_map_json[:300]}")
            
            print(f"Generated map data with {len(price_map_data['data'])} properties and timestamp {price_map_data.get('timestamp', 'none')}")
            print(f"Map center: {price_map_data['center']}")
            
        except Exception as e:
            import traceback
            print(f"Error generating map data: {str(e)}")
            print(traceback.format_exc())
            
            # Provide minimal data structure to avoid JavaScript errors
            # Include a timestamp to force map refresh even with error data
            import time
            fallback_data = {
                "data": [],
                "center": {"lat": 41.6, "lng": -93.6},  # Default center (Des Moines, IA)
                "zoom": 10,
                "error": str(e),
                "timestamp": int(time.time() * 1000),  # Current time in milliseconds
                "filter_change": True,  # Force complete map regeneration
                'tab_activated_timestamp': time.time()
            }
            price_map_json = json.dumps(fallback_data)
        
        price_distribution = generate_price_distribution(filtered_df)
        
        # Get regression report data for feature importance
        regression_data = data_provider.get_regression_models()
        feature_importance = generate_feature_importance(filtered_df, "Sale_Price", regression_data)
        
        # Building type distribution
        if "Bldg_Type" in filtered_df.columns and "Sale_Price" in filtered_df.columns:
            building_type_dist = generate_box_plot(filtered_df, "Sale_Price", "Bldg_Type")
        else:
            building_type_dist = {"data": [], "layout": {"title": "Building Type data not available"}}
        
        return price_map_json, price_distribution, feature_importance, building_type_dist
    
    # Add fallback for Google Maps if it doesn't load
    @callback(
        Output("google-price-map-fallback", "children"),
        Input("google-price-map-data", "children"),
        Input("filtered-data-store", "data")
    )
    def update_map_fallback(google_map_data, filtered_data_json):
        # Create a static fallback map using Plotly if needed
        if not google_map_data:
            # Check if filtered_data_json is None
            if filtered_data_json is None:
                # Return empty div with message if no data is available
                return html.Div("No data available for map visualization", 
                               style={"textAlign": "center", "marginTop": "50px", "color": "#888"})
                
            # Convert JSON to DataFrame
            filtered_df = pd.read_json(filtered_data_json, orient='split')
            
            # Generate regular price map as fallback
            price_map_fig = generate_price_map(filtered_df)
            
            return dcc.Graph(
                id="price-map-fallback-graph",
                figure=price_map_fig,
                config={"displayModeBar": True}
            )
        
        # Return empty div if Google Maps loaded successfully
        return html.Div()
    
    # Update Property Analysis tab visualizations
    @callback(
        Output("property-scatter-plot", "figure"),
        Input("scatter-x-axis", "value"),
        Input("scatter-y-axis", "value"),
        Input("scatter-color", "value"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_property_scatter_plot(x_col, y_col, color_col, filtered_data_json, active_tab):
        # Only update when Property Analysis tab is active
        if active_tab != "tab-property" or not x_col or not y_col:
            return {"data": [], "layout": {"title": "Select x and y variables"}}
        
        # Check if filtered_data_json is None
        if filtered_data_json is None:
            return {"data": [], "layout": {"title": "No property data available for the current filter criteria"}}
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Generate scatter plot
        return generate_scatter_plot(filtered_df, x_col, y_col, color_col, trendline=True)
    
    @callback(
        Output("property-box-plot", "figure"),
        Input("box-plot-numeric", "value"),
        Input("box-plot-category", "value"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_property_box_plot(numeric_col, category_col, filtered_data_json, active_tab):
        # Only update when Property Analysis tab is active
        if active_tab != "tab-property" or not numeric_col or not category_col:
            return {"data": [], "layout": {"title": "Select numeric and category variables"}}
        
        # Check if filtered_data_json is None
        if filtered_data_json is None:
            return {"data": [], "layout": {"title": "No property data available for the current filter criteria"}}
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Generate box plot
        return generate_box_plot(filtered_df, numeric_col, category_col)
    
    @callback(
        Output("correlation-heatmap", "figure"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_correlation_heatmap(filtered_data_json, active_tab):
        # Only update when Property Analysis tab is active
        if active_tab != "tab-property":
            return {"data": [], "layout": {}}
        
        # Check if filtered_data_json is None
        if filtered_data_json is None:
            return {"data": [], "layout": {"title": "No property data available for the current filter criteria"}}
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Generate correlation heatmap
        return generate_correlation_heatmap(filtered_df)
    
    @callback(
        Output("parallel-coordinates", "figure"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_parallel_coordinates(filtered_data_json, active_tab):
        # Only update when Property Analysis tab is active
        if active_tab != "tab-property":
            return {"data": [], "layout": {}}
        
        # Check if filtered_data_json is None
        if filtered_data_json is None:
            return {"data": [], "layout": {"title": "No property data available for the current filter criteria"}}
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Select important columns for parallel coordinates
        important_cols = ["Sale_Price", "Lot_Area", "Lot_Frontage"]
        color_col = "Bldg_Type"
        
        # Add any other numeric columns if they exist
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            if col not in important_cols and len(important_cols) < 7:
                important_cols.append(col)
        
        # Check if all columns exist
        valid_cols = [col for col in important_cols if col in filtered_df.columns]
        
        # Generate parallel coordinates plot
        return generate_parallel_coordinates(filtered_df, valid_cols, color_col)
    
    # Update Market Trends tab visualizations
    @callback(
        Output("price-per-sqft-analysis", "figure"),
        Output("building-type-comparison", "figure"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_market_trends_visualizations(filtered_data_json, active_tab):
        # Only update when Market Trends tab is active
        if active_tab != "tab-market":
            empty_fig = {"data": [], "layout": {}}
            return empty_fig, empty_fig
        
        # Check if filtered_data_json is None
        if filtered_data_json is None:
            empty_fig = {"data": [], "layout": {"title": "No property data available for the current filter criteria"}}
            return empty_fig, empty_fig
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Calculate price per square foot if columns exist
        if "Sale_Price" in filtered_df.columns and "Lot_Area" in filtered_df.columns:
            filtered_df["Price_Per_SqFt"] = filtered_df["Sale_Price"] / filtered_df["Lot_Area"]
            
            # Generate scatter plot for price per sq ft
            price_per_sqft_fig = generate_scatter_plot(
                filtered_df, 
                "Lot_Area", 
                "Price_Per_SqFt", 
                "Bldg_Type" if "Bldg_Type" in filtered_df.columns else None
            )
        else:
            price_per_sqft_fig = {"data": [], "layout": {"title": "Required columns not available"}}
        
        # Generate building type comparison chart
        if "Bldg_Type" in filtered_df.columns and "Sale_Price" in filtered_df.columns:
            # Aggregate average price by building type
            building_type_avg = filtered_df.groupby("Bldg_Type")["Sale_Price"].mean().reset_index()
            
            # Create comparison bar chart
            import plotly.express as px
            building_type_fig = px.bar(
                building_type_avg, 
                x="Bldg_Type", 
                y="Sale_Price",
                title="Average Price by Building Type",
                labels={"Sale_Price": "Average Sale Price", "Bldg_Type": "Building Type"}
            )
        else:
            building_type_fig = {"data": [], "layout": {"title": "Building Type data not available"}}
        
        return price_per_sqft_fig, building_type_fig
    
    # Update Market Trends tab with temporal analysis
    @callback(
        Output("price-by-year", "figure"),
        Output("age-price-correlation", "figure"),
        Output("decade-bldg-heatmap", "figure"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_year_trend_visualizations(filtered_data_json, active_tab):
        # Only update when Market Trends tab is active
        if active_tab != "tab-market":
            # Return empty figures when tab is not active
            empty_fig = {"data": [], "layout": {}}
            return empty_fig, empty_fig, empty_fig
        
        # Check if filtered_data_json is None
        if filtered_data_json is None:
            empty_fig = {"data": [], "layout": {"title": "No property data available for the current filter criteria"}}
            return empty_fig, empty_fig, empty_fig
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Generate year trend visualizations
        year_trends = generate_year_trend_analysis(filtered_df)
        
        # Return the generated figures, or empty figures if not available
        price_by_year = year_trends.get('price_by_year', {"data": [], "layout": {"title": "Year data not available"}})
        age_price = year_trends.get('age_price_correlation', {"data": [], "layout": {"title": "Age data not available"}})
        decade_heatmap = year_trends.get('decade_bldg_heatmap', {"data": [], "layout": {"title": "Decade data not available"}})
        
        return price_by_year, age_price, decade_heatmap
    
    # Handle property comparisons tab
    @callback(
        Output("comparison-price-box", "figure"),
        Output("comparison-price-bar", "figure"),
        Output("comparison-scatter", "figure"),
        Output("comparison-radar", "figure"),
        Input("generate-comparison-button", "n_clicks"),
        Input("comparison-column", "value"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_property_comparisons(n_clicks, compare_col, filtered_data_json, active_tab):
        # Only update when Property Comparison tab is active
        if active_tab != "tab-comparison":
            # Return empty figures when tab is not active
            empty_fig = {"data": [], "layout": {}}
            return empty_fig, empty_fig, empty_fig, empty_fig
        
        # Check if filtered_data_json is None
        if filtered_data_json is None:
            # Return empty figures with a message when no data is available
            empty_fig = {"data": [], "layout": {"title": "No property data available for the current filter criteria"}}
            return empty_fig, empty_fig, empty_fig, empty_fig
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Check if comparison column exists
        if compare_col not in filtered_df.columns:
            empty_fig = {"data": [], "layout": {"title": f"Column '{compare_col}' not found in data"}}
            return empty_fig, empty_fig, empty_fig, empty_fig
        
        # Generate property comparisons
        comparison_figs = generate_property_comparisons(filtered_df, compare_col)
        
        # Extract the figures we need
        price_box = comparison_figs.get('Sale_Price_box', {"data": [], "layout": {"title": "Price data not available"}})
        price_bar = comparison_figs.get('Sale_Price_bar', {"data": [], "layout": {"title": "Price data not available"}})
        scatter = comparison_figs.get('price_vs_area', {"data": [], "layout": {"title": "Area data not available"}})
        radar = comparison_figs.get('radar_comparison', {"data": [], "layout": {"title": "Comparison data not available"}})
        
        return price_box, price_bar, scatter, radar
    
    # Update Data Table tab
    @callback(
        Output("data-table-container", "children"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_data_table(filtered_data_json, active_tab):
        # Only update when Data Table tab is active
        if active_tab != "tab-data":
            return []
        
        # Check if filtered_data_json is None
        if filtered_data_json is None:
            return [html.Div("No property data available for the current filter criteria", 
                           style={"textAlign": "center", "marginTop": "50px", "color": "#888"})]
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Limit to 1000 rows for performance
        if len(filtered_df) > 1000:
            filtered_df = filtered_df.head(1000)
            note = html.Div("Note: Showing first 1,000 records", className="text-muted mb-2")
        else:
            note = html.Div(f"Showing all {len(filtered_df)} records", className="text-muted mb-2")
        
        # Create data table
        table = dash_table.DataTable(
            id="housing-data-table",
            columns=[{"name": col.replace('_', ' '), "id": col} for col in filtered_df.columns],
            data=filtered_df.to_dict('records'),
            page_size=20,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            css=[{"selector": ".show-hide", "rule": "display: none"}]
        )
        
        return [note, table]