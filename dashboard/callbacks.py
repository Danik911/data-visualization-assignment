"""
Callbacks module for the dashboard.
This module handles all interactive elements and user interactions.
"""

from dash import Input, Output, State, callback, dash_table, html, ctx
import pandas as pd
import json
from typing import Dict, List, Any, Optional

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
    generate_year_trend_analysis    # New function for year trend analysis
)


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
        Input("reset-filters-button", "n_clicks")
    )
    def filter_data(building_types, price_range, area_range, reset_clicks):
        # Create filters dictionary
        filters = {}
        
        # Only add filters if they have values
        if building_types:
            filters["Bldg_Type"] = building_types
            
        if price_range:
            filters["Sale_Price"] = {"range": price_range}
            
        if area_range:
            filters["Lot_Area"] = {"range": area_range}
        
        # Reset filters on button click (context triggered)
        if ctx.triggered and "reset-filters-button" in ctx.triggered[0]["prop_id"]:
            filters = {}
        
        # Get filtered data from provider
        filtered_df = data_provider.get_filtered_data(filters)
        
        # Convert to JSON for store
        return filtered_df.to_json(date_format='iso', orient='split')
    
    # Update summary cards based on filtered data
    @callback(
        Output("summary-cards-row", "children"),
        Input("filtered-data-store", "data")
    )
    def update_summary_cards(filtered_data_json):
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Generate summary statistics
        summary_data = generate_summary_cards(filtered_df)
        
        # Import layout module here to avoid circular import
        from dashboard.layout import create_summary_cards
        return create_summary_cards(summary_data)
    
    # Update Overview tab visualizations
    @callback(
        Output("price-map", "figure"),
        Output("price-distribution", "figure"),
        Output("feature-importance", "figure"),
        Output("building-type-distribution", "figure"),
        Input("filtered-data-store", "data"),
        Input("dashboard-tabs", "active_tab")
    )
    def update_overview_visualizations(filtered_data_json, active_tab):
        # Only update when Overview tab is active
        if active_tab != "tab-overview":
            # Return empty figures when tab is not active
            empty_fig = {"data": [], "layout": {}}
            return empty_fig, empty_fig, empty_fig, empty_fig
        
        # Convert JSON to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        
        # Generate visualizations
        price_map = generate_price_map(filtered_df)
        price_distribution = generate_price_distribution(filtered_df)
        
        # Get regression report data for feature importance
        regression_data = data_provider.get_regression_models()
        feature_importance = generate_feature_importance(filtered_df, "Sale_Price", regression_data)
        
        # Building type distribution
        if "Bldg_Type" in filtered_df.columns and "Sale_Price" in filtered_df.columns:
            building_type_dist = generate_box_plot(filtered_df, "Sale_Price", "Bldg_Type")
        else:
            building_type_dist = {"data": [], "layout": {"title": "Building Type data not available"}}
        
        return price_map, price_distribution, feature_importance, building_type_dist
    
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