"""
Visualizations module for the dashboard.
This module provides functions to generate Plotly figures for the dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
from dashboard.config import get_colorscale, get_chart_defaults, get_color_palette, get_column_display_label
import logging  # Added for logging


# ---- Helper functions ----

def _validate_columns(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, str]:
    """
    Validate if the DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of column names that must be present
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {', '.join(missing_cols)}"
        return False, error_msg
    return True, ""


def _create_empty_figure(message: str) -> go.Figure:
    """
    Create an empty figure with an error message.
    
    Args:
        message: Error message to display
        
    Returns:
        Empty Plotly figure with error message
    """
    fig = go.Figure()
    fig.update_layout(
        title=message,
        annotations=[
            {
                "text": message,
                "showarrow": False,
                "font": {"size": 14}
            }
        ]
    )
    return fig


def _format_hover_data(df: pd.DataFrame, 
                      base_cols: List[str], 
                      optional_cols: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Format hover data for Plotly figures.
    
    Args:
        df: DataFrame containing data
        base_cols: List of column names to always include
        optional_cols: List of column names to include if they exist in the DataFrame
        
    Returns:
        Dictionary suitable for hover_data parameter in Plotly Express
    """
    hover_data = {col: True for col in base_cols if col in df.columns}
    
    if optional_cols:
        for col in optional_cols:
            hover_data[col] = True if col in df.columns else False
            
    return hover_data


def _apply_standard_layout(fig: go.Figure, 
                          title: str, 
                          chart_type: Optional[str] = None) -> go.Figure:
    """
    Apply standard layout settings to a figure.
    
    Args:
        fig: Plotly figure to modify
        title: Title for the figure
        chart_type: Type of chart for specific settings
        
    Returns:
        Modified Plotly figure
    """
    # Get layout defaults from config
    layout_defaults = get_chart_defaults("layout")
    
    # Get chart-specific defaults if available
    if chart_type and chart_type in get_chart_defaults():
        chart_defaults = get_chart_defaults(chart_type)
    else:
        chart_defaults = {}
    
    # Apply layout settings
    fig.update_layout(
        title=title,
        margin=layout_defaults.get("margin", {"r": 10, "t": 40, "l": 10, "b": 10})
    )
    
    # Add more layout options based on chart type if needed
    if chart_type == "maps":
        fig.update_layout(
            coloraxis_colorbar=dict(title="Sale Price ($)")
        )
    
    return fig


# ---- Visualization functions ----

def generate_price_map(df: pd.DataFrame, use_clustering: bool = None) -> go.Figure:
    """
    Generate a map visualization of housing prices by location.
    
    Args:
        df: DataFrame containing housing data with Latitude and Longitude columns
        use_clustering: Whether to use clustering for better visualization of many points
        
    Returns:
        Plotly figure object with the map visualization
    """
    from dashboard.visualizations_helpers import PriceMap
    
    # Use the PriceMap class to generate the visualization
    price_map = PriceMap(df, use_clustering)
    return price_map.generate()


def generate_google_price_map(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a Google Maps visualization of housing prices by location.
    
    Args:
        df: DataFrame containing housing data with Latitude and Longitude columns
        
    Returns:
        Dictionary with Google Maps configuration and data
    """
    from dashboard.visualizations_helpers import GooglePriceMap
    
    # Use the GooglePriceMap class to generate the visualization data
    price_map = GooglePriceMap(df)
    return price_map.generate()


def generate_price_distribution(df: pd.DataFrame, bin_size: int = 50000) -> go.Figure:
    """
    Generate a histogram showing the distribution of housing prices.
    
    Args:
        df: DataFrame containing housing data with Sale_Price column
        bin_size: Size of histogram bins
        
    Returns:
        Plotly figure object with the price distribution
    """
    if df.empty or 'Sale_Price' not in df.columns or df['Sale_Price'].isna().all():
        return _create_empty_figure("No valid price data available for distribution plot.")
    
    # Ensure Sale_Price is numeric and drop NaNs for calculation if any exist
    # (though primary check is above, this is a safeguard for calculations)
    sale_prices = pd.to_numeric(df['Sale_Price'], errors='coerce').dropna()
    if sale_prices.empty:
        return _create_empty_figure("No valid numeric price data after cleaning for distribution plot.")

    from dashboard.visualizations_helpers import PriceDistribution
    
    # Use the PriceDistribution class to generate the visualization
    # We pass the cleaned sale_prices by creating a temporary DataFrame for the helper
    price_dist = PriceDistribution(pd.DataFrame(sale_prices, columns=['Sale_Price']), bin_size)
    return price_dist.generate()


def generate_correlation_heatmap(df: pd.DataFrame, columns: List[str] = None) -> go.Figure:
    """
    Generate a correlation heatmap for numeric columns.
    
    Args:
        df: DataFrame containing housing data
        columns: List of columns to include in the heatmap (optional)
        
    Returns:
        Plotly figure object with the correlation heatmap
    """
    from dashboard.visualizations_helpers import CorrelationHeatmap
    
    # Use the CorrelationHeatmap class to generate the visualization
    corr_heatmap = CorrelationHeatmap(df, columns)
    return corr_heatmap.generate()


def generate_scatter_plot(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str,
    color_col: Optional[str] = None,
    trendline: bool = True
) -> go.Figure:
    """
    Generate a scatter plot comparing two numeric variables.
    
    Args:
        df: DataFrame containing housing data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Column name for color coding points (optional)
        trendline: Whether to add a trendline
        
    Returns:
        Plotly figure object with the scatter plot
    """
    from dashboard.visualizations_helpers import ScatterComparison
    
    # Use the ScatterComparison class to generate the visualization
    scatter_comp = ScatterComparison(df, x_col, y_col, color_col, trendline)
    return scatter_comp.generate()


def generate_box_plot(
    df: pd.DataFrame, 
    numeric_col: str, 
    category_col: str
) -> go.Figure:
    """
    Generate a box plot showing distribution of a numeric variable across categories.
    
    Args:
        df: DataFrame containing housing data
        numeric_col: Column name for the numeric variable
        category_col: Column name for the categorical variable
        
    Returns:
        Plotly figure object with the box plot
    """
    from dashboard.visualizations_helpers import BoxPlotVisualization
    
    # Use the BoxPlotVisualization class to generate the visualization
    box_plot = BoxPlotVisualization(df, numeric_col, category_col)
    return box_plot.generate()


def generate_time_series(
    df: pd.DataFrame, 
    date_col: str, 
    value_col: str,
    group_col: Optional[str] = None
) -> go.Figure:
    """
    Generate a time series plot.
    
    Args:
        df: DataFrame containing housing data
        date_col: Column name containing date information
        value_col: Column name containing the values to plot
        group_col: Column name to group by (optional)
        
    Returns:
        Plotly figure object with the time series plot
    """
    from dashboard.visualizations_helpers import TimeSeries
    
    # Use the TimeSeries class to generate the visualization
    time_series = TimeSeries(df, date_col, value_col, group_col)
    return time_series.generate()


def generate_feature_importance(df: pd.DataFrame, target_col: str, report_data: Dict[str, Any] = None) -> go.Figure:
    """
    Generate a feature importance plot based on regression analysis results.
    
    Args:
        df: DataFrame containing housing data
        target_col: Target column name
        report_data: Dictionary containing regression analysis results (optional)
        
    Returns:
        Plotly figure object with the feature importance plot
    """
    # If report data is provided, use it
    if report_data and 'feature_importance' in report_data:
        # Extract feature importance from report
        features = list(report_data['feature_importance'].keys())
        importance = list(report_data['feature_importance'].values())
        
        # Create bar plot of feature importance
        fig = px.bar(
            x=features,
            y=importance,
            title=f"Feature Importance for {target_col}",
            labels={'x': 'Feature', 'y': 'Importance'},
            color=importance,
            color_continuous_scale='Viridis'
        )
        
    else:
        # If no report data, run a basic correlation analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if target_col not in numeric_cols:
            return _create_empty_figure(f"Feature importance requires numeric target column: {target_col}")
        
        # Calculate correlation with target
        correlations = df[numeric_cols].corr()[target_col].abs().drop(target_col)
        correlations = correlations.sort_values(ascending=False)
        
        # Use top 15 features
        top_features = correlations.head(15).index.tolist()
        top_correlations = correlations.head(15).values.tolist()
        
        # Create bar plot of absolute correlations
        fig = px.bar(
            x=top_features,
            y=top_correlations,
            title=f"Feature Correlation with {target_col}",
            labels={'x': 'Feature', 'y': f'Absolute Correlation with {target_col}'},
            color=top_correlations,
            color_continuous_scale='Viridis'
        )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="",
        yaxis_title="Importance",
        coloraxis_showscale=False
    )
    
    return fig


def generate_parallel_coordinates(
    df: pd.DataFrame, 
    columns: List[str], 
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Generate a parallel coordinates plot for multi-dimensional data exploration.
    
    Args:
        df: DataFrame containing housing data
        columns: List of columns to include
        color_col: Column to use for color coding (optional)
        
    Returns:
        Plotly figure object with parallel coordinates plot
    """
    # Check if all columns exist
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        return _create_empty_figure(f"Missing columns for parallel coordinates: {', '.join(missing_columns)}")
    
    # Create a copy of the dataframe with only the needed columns
    plot_df = df[columns].copy()
    
    # If color column is specified and exists, add it
    if color_col and color_col in df.columns:
        # Check if the color column is categorical
        if df[color_col].dtype == 'object' or df[color_col].dtype.name == 'category':
            # Create a numerical mapping for categorical values
            unique_categories = df[color_col].unique()
            category_map = {cat: i for i, cat in enumerate(unique_categories)}
            
            # Create a numerical color array
            color_array = df[color_col].map(category_map).values
            
            # Create dimensions for parallel coordinates
            dimensions = []
            for col in columns:
                dimensions.append(
                    dict(
                        range=[plot_df[col].min(), plot_df[col].max()],
                        label=col.replace('_', ' '),
                        values=plot_df[col].values
                    )
                )
            
            # Create the parallel coordinates plot with go.Parcoords
            fig = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=color_array,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title=color_col.replace('_', ' '),
                            tickvals=list(range(len(unique_categories))),
                            ticktext=list(unique_categories),
                        )
                    ),
                    dimensions=dimensions
                )
            )
            
            # Add a legend with category colors
            fig.update_layout(
                title="Parallel Coordinates Plot",
                font=dict(size=10),
            )
            
        else:
            # For numeric color columns, use px.parallel_coordinates as before
            fig = px.parallel_coordinates(
                plot_df,
                color=color_col,
                labels={col: col.replace('_', ' ') for col in plot_df.columns},
                title="Parallel Coordinates Plot",
                color_continuous_scale=px.colors.sequential.Viridis
            )
    else:
        # Without color column, use px.parallel_coordinates as before
        fig = px.parallel_coordinates(
            plot_df,
            labels={col: col.replace('_', ' ') for col in plot_df.columns},
            title="Parallel Coordinates Plot"
        )
    
    fig.update_layout(
        font=dict(size=10)
    )
    
    return fig


def generate_summary_cards(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Generate summary statistics for the dashboard cards.
    
    Args:
        df: DataFrame containing housing data
        
    Returns:
        Dictionary containing summary statistics
    """
    # Default values for empty or problematic data
    default_summary = {
        "total_properties": {"value": 0, "description": "Total Properties"},
        "avg_price": {"value": "N/A", "description": "Average Price"},
        "median_price": {"value": "N/A", "description": "Median Price"},
        "price_range": {"value": "N/A", "description": "Price Range"},
        "avg_area": {"value": "N/A", "description": "Average Lot Area"},
        "common_type": {"value": "N/A", "description": "Most Common Building Type"}
    }

    if df.empty or not all(col in df.columns for col in ['Sale_Price', 'Lot_Area', 'Bldg_Type']):
        # If essential columns are missing, also return default
        if df.empty:
            return default_summary
        else: # df is not empty, but columns are missing, fill what we can
            summary = default_summary.copy()
            summary["total_properties"] = {"value": f"{len(df):,}", "description": "Total Properties"}
            if 'Sale_Price' in df.columns and not df['Sale_Price'].empty and df['Sale_Price'].notna().any():
                summary["avg_price"] = {"value": f"${df['Sale_Price'].mean():,.0f}", "description": "Average Price"}
                summary["median_price"] = {"value": f"${df['Sale_Price'].median():,.0f}", "description": "Median Price"}
                summary["price_range"] = {"value": f"${df['Sale_Price'].min():,.0f} - ${df['Sale_Price'].max():,.0f}", "description": "Price Range"}
            if 'Lot_Area' in df.columns and not df['Lot_Area'].empty and df['Lot_Area'].notna().any():
                summary["avg_area"] = {"value": f"{df['Lot_Area'].mean():,.0f} sq.ft", "description": "Average Lot Area"}
            if 'Bldg_Type' in df.columns and not df['Bldg_Type'].empty and df['Bldg_Type'].notna().any():
                most_common = df['Bldg_Type'].mode()
                if not most_common.empty:
                    most_common_type = most_common[0]
                    type_percentage = (df['Bldg_Type'].value_counts(normalize=True).max() * 100)
                    summary["common_type"] = {"value": f"{most_common_type} ({type_percentage:.1f}%)", "description": "Most Common Building Type"}
            return summary

    try:
        # Calculate summary statistics
        total_properties = len(df)
        
        # Average price if Sale_Price column exists
        if 'Sale_Price' in df.columns:
            avg_price = df['Sale_Price'].mean()
            median_price = df['Sale_Price'].median()
            price_range = f"${df['Sale_Price'].min():,.0f} - ${df['Sale_Price'].max():,.0f}"
        
        # Average area if Lot_Area column exists
        if 'Lot_Area' in df.columns:
            avg_area = df['Lot_Area'].mean()
        
        # Building type distribution if Bldg_Type column exists
        if 'Bldg_Type' in df.columns:
            most_common_type = df['Bldg_Type'].value_counts().idxmax()
            type_percentage = (df['Bldg_Type'].value_counts().max() / len(df)) * 100
        
        # Use the user-friendly building type label
        from dashboard.config import get_building_type_label
        friendly_type_name = get_building_type_label(most_common_type)
        
        # Create summary dictionary
        summary = {
            "total_properties": {"value": total_properties, "description": "Total Properties"},
            "avg_price": {"value": f"${avg_price:.0f}" if 'Sale_Price' in df.columns else "N/A", "description": "Average Price"},
            "median_price": {"value": f"${median_price:.0f}" if 'Sale_Price' in df.columns else "N/A", "description": "Median Price"},
            "price_range": {"value": price_range if 'Sale_Price' in df.columns else "N/A", "description": "Price Range"},
            "avg_area": {"value": f"{avg_area:.0f} sq.ft" if 'Lot_Area' in df.columns else "N/A", "description": "Average Lot Area"},
            "common_type": {"value": f"{friendly_type_name} ({type_percentage:.1f}%)" if 'Bldg_Type' in df.columns else "N/A", "description": "Most Common Building Type"}
        }
        
        return summary
    except Exception as e:
        return _create_empty_figure(f"Error generating summary cards: {str(e)}")


def generate_property_comparisons(df: pd.DataFrame, compare_col: str = "Bldg_Type", metrics: List[str] = None) -> Dict[str, go.Figure]:
    """
    Generate side-by-side comparisons of different property types across key metrics.
    
    Args:
        df: DataFrame containing housing data
        compare_col: Column to use for grouping/comparison (e.g., Building Type)
        metrics: List of metrics to compare (optional, defaults to key metrics if available)
        
    Returns:
        Dictionary of plotly figures with property comparisons
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from dashboard.visualizations_helpers import get_column_display_label, create_empty_figure
    import logging

    logger = logging.getLogger(__name__)

    # Initialize results with empty figures for all expected keys using layout IDs
    results = {
        "comparison-price-box": create_empty_figure("Initializing..."),
        "comparison-price-bar": create_empty_figure("Initializing..."),
        "comparison-scatter": create_empty_figure("Initializing..."),
        "comparison-radar": create_empty_figure("Initializing...")
    }

    try:
        # Check for missing or empty data
        if compare_col not in df.columns or df[compare_col].dropna().empty:
            msg = f"Comparison column '{compare_col}' not found or contains no data."
            logger.warning(msg)
            # Update results with specific message
            for key in results: results[key] = create_empty_figure(msg)
            return results

        # Remove rows with missing comparison column
        df_plot = df.dropna(subset=[compare_col]).copy()
        if df_plot.empty:
            msg = f"No data available for selected comparison: {compare_col} after dropping NA."
            logger.warning(msg)
            # Update results with specific message
            for key in results: results[key] = create_empty_figure(msg)
            return results

        # If comparing by building type, use friendly labels
        if compare_col == 'Bldg_Type':
            from dashboard.config import get_building_type_label
            df_plot['Bldg_Type_Display'] = df_plot['Bldg_Type'].apply(get_building_type_label)
            compare_col_display = 'Bldg_Type_Display'
        else:
            compare_col_display = compare_col

        # Default metrics if not specified
        if not metrics:
            available_cols = df_plot.columns.tolist()
            possible_metrics = [
                "Sale_Price", "Lot_Area", "Lot_Frontage", "Year_Built",
                "Total_Bsmt_SF", "First_Flr_SF", "Full_Bath", "Half_Bath",
                "Bedroom_AbvGr", "Fireplaces"
            ]
            metrics = [col for col in possible_metrics if col in available_cols][:5]
            # Ensure Sale_Price is included if available, for expected output keys
            if "Sale_Price" in available_cols and "Sale_Price" not in metrics:
                if len(metrics) >= 5: metrics.pop() # Make space if full
                metrics.insert(0, "Sale_Price") # Add Sale_Price

        # --- Plotting Logic Starts ---
        # Group comparison (boxplot and bar chart)
        for metric in metrics:
            if metric in df_plot.columns and df_plot[metric].dtype in ['int64', 'float64']:
                try:
                    # Box plot
                    fig_box = px.box(
                        df_plot,
                        x=compare_col_display,
                        y=metric,
                        color=compare_col_display,
                        title=f"{get_column_display_label(metric)} by {get_column_display_label(compare_col)}",
                        labels={
                            metric: get_column_display_label(metric),
                            compare_col_display: get_column_display_label(compare_col)
                        },
                        points="outliers"
                    )
                    fig_box.update_layout(
                        xaxis_title=get_column_display_label(compare_col),
                        yaxis_title=get_column_display_label(metric),
                        showlegend=False
                    )
                    # Update results if the metric matches an expected key prefix
                    if metric == "Sale_Price":
                       results["comparison-price-box"] = fig_box # Use layout ID

                    # Bar chart for averages
                    avg_by_group = df_plot.groupby(compare_col)[metric].mean().reset_index()
                    count_by_group = df_plot.groupby(compare_col)[metric].count().reset_index()
                    avg_by_group['count'] = count_by_group[metric]
                    if compare_col == 'Bldg_Type':
                        from dashboard.config import get_building_type_label
                        avg_by_group[compare_col_display] = avg_by_group[compare_col].apply(get_building_type_label)
                        avg_by_group['label'] = avg_by_group[compare_col_display] + ' (n=' + avg_by_group['count'].astype(str) + ')'
                    else:
                        avg_by_group['label'] = avg_by_group[compare_col].astype(str) + ' (n=' + avg_by_group['count'].astype(str) + ')'
                    fig_bar = px.bar(
                        avg_by_group,
                        x=compare_col_display if compare_col == 'Bldg_Type' else compare_col,
                        y=metric,
                        color=compare_col_display if compare_col == 'Bldg_Type' else compare_col,
                        text_auto=True,
                        title=f"Average {get_column_display_label(metric)} by {get_column_display_label(compare_col)}",
                        labels={
                            metric: f"Avg. {get_column_display_label(metric)}",
                            compare_col_display: get_column_display_label(compare_col),
                            compare_col: get_column_display_label(compare_col)
                        },
                        hover_data={
                            'count': True,
                            'label': False,
                            compare_col if compare_col != 'Bldg_Type' else compare_col_display: False
                        }
                    )
                    fig_bar.update_layout(
                        xaxis_title="",
                        yaxis_title=f"Average {get_column_display_label(metric)}",
                        showlegend=False,
                        xaxis={'categoryorder': 'total descending'}
                    )
                    # Update results if the metric matches an expected key prefix
                    if metric == "Sale_Price":
                        results["comparison-price-bar"] = fig_bar # Use layout ID
                except Exception as e:
                    msg = f"Error generating chart for {metric}: {str(e)}"
                    logger.error(msg, exc_info=True)
                    if metric == "Sale_Price":
                        results["comparison-price-box"] = create_empty_figure(msg) # Use layout ID
                        results["comparison-price-bar"] = create_empty_figure(msg) # Use layout ID

        # Scatter plot (price vs area)
        if "Sale_Price" in df_plot.columns and "Lot_Area" in df_plot.columns:
            try:
                fig = px.scatter(
                    df_plot,
                    x="Lot_Area",
                    y="Sale_Price",
                    color=compare_col_display,
                    opacity=0.7,
                    title=f"{get_column_display_label('Sale_Price')} vs {get_column_display_label('Lot_Area')} by {get_column_display_label(compare_col)}",
                    labels={
                        "Lot_Area": get_column_display_label("Lot_Area"),
                        "Sale_Price": get_column_display_label("Sale_Price"),
                        compare_col_display: get_column_display_label(compare_col),
                        compare_col: get_column_display_label(compare_col)
                    },
                    trendline="ols",
                    trendline_scope="overall"
                )
                fig.update_layout(
                    legend_title=get_column_display_label(compare_col)
                )
                results["comparison-scatter"] = fig # Use layout ID
            except Exception as e:
                msg = f"Error generating scatter plot: {str(e)}"
                logger.error(msg, exc_info=True)
                results["comparison-scatter"] = create_empty_figure(msg) # Use layout ID

        # Radar chart (normalized metrics)
        if len(metrics) >= 3:
            # Filter for numeric metrics among the selected ones
            numeric_metrics = [m for m in metrics if m in df_plot.columns and df_plot[m].dtype in ['int64', 'float64']][:5]
            
            if len(numeric_metrics) >= 3:
                try:
                    if compare_col == 'Bldg_Type':
                        from dashboard.config import get_building_type_label
                        groups = df_plot[compare_col].unique().tolist()
                        group_names = [get_building_type_label(g) for g in groups]
                    else:
                        groups = df_plot[compare_col].unique().tolist()
                        group_names = groups
                    
                    radar_df = df_plot.copy()
                    
                    # Normalize selected numeric metrics
                    for metric in numeric_metrics:
                        min_val = df_plot[metric].min()
                        max_val = df_plot[metric].max()
                        if max_val > min_val:
                            radar_df[metric] = (df_plot[metric] - min_val) / (max_val - min_val)
                        else:
                            radar_df[metric] = 0 # Avoid division by zero if all values are the same

                    group_avgs = radar_df.groupby(compare_col)[numeric_metrics].mean().reset_index()
                    
                    fig = go.Figure()
                    
                    # Add traces for each group
                    for i, group in enumerate(groups):
                        group_data = group_avgs[group_avgs[compare_col] == group]
                        if not group_data.empty:
                            values = group_data[numeric_metrics].iloc[0].values.flatten().tolist()
                            fig.add_trace(go.Scatterpolar(
                                r=values,
                                theta=numeric_metrics, # Use original metric names for labels
                                fill='toself',
                                name=group_names[i] # Use display name
                            ))
                            
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1] # Normalized range
                            )
                        ),
                        showlegend=True,
                        title="Normalized Metrics Comparison"
                    )
                    results["comparison-radar"] = fig # Use layout ID
                except Exception as e:
                    msg = f"Error generating radar chart: {str(e)}"
                    logger.error(msg, exc_info=True)
                    results["comparison-radar"] = create_empty_figure(msg) # Use layout ID
            else: # Not enough numeric metrics for radar
                results["comparison-radar"] = create_empty_figure("Need at least 3 numeric metrics for radar chart.") # Use layout ID
        else: # Not enough metrics selected overall
             results["comparison-radar"] = create_empty_figure("Need at least 3 metrics selected for radar chart.") # Use layout ID

        # --- Final check ---
        # Ensure all expected keys exist before returning, use empty if any failed
        final_results = {}
        expected_keys = ["comparison-price-box", "comparison-price-bar", "comparison-scatter", "comparison-radar"] # Use layout IDs
        for key in expected_keys:
            if key in results and isinstance(results[key], go.Figure):
                 final_results[key] = results[key]
            else:
                 # Log if a key is missing unexpectedly (should have been initialized)
                 if key not in results: 
                     logger.warning(f"Expected key '{key}' missing from results in generate_property_comparisons.")
                 # Use the imported standalone function
                 final_results[key] = create_empty_figure(f"Failed to generate comparison chart ({key}). Error: {e}")

        return final_results

    except Exception as e:
        msg = f"Unexpected error generating comparisons: {str(e)}"
        logger.error(msg, exc_info=True)
        # Update all figures in the initialized dict with an error message
        error_figure = create_empty_figure(msg)
        for key in results:
            results[key] = error_figure
        return results # Return the dict with error figures


def generate_year_trend_analysis(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Generate year-based trend analysis visualizations.
    
    Args:
        df: DataFrame containing housing data with year-related columns
        
    Returns:
        Dictionary of plotly figures with year-based trends
    """
    results = {}
    
    # Check if Year_Built column exists
    if 'Year_Built' not in df.columns:
        results["error"] = create_empty_figure("Year trend analysis requires Year_Built column")
        return results
        
    # Convert Year_Built to numeric if not already
    df = df.copy()
    try:
        df['Year_Built'] = pd.to_numeric(df['Year_Built'], errors='coerce')
    except:
        pass
    
    # Filter out invalid years
    df = df[df['Year_Built'] > 1800].copy()
    
    if df.empty:
        results["error"] = create_empty_figure("No valid year data available")
        return results
    
    # 1. Average price by year built
    if 'Sale_Price' in df.columns:
        # Group by year and calculate statistics
        year_stats = df.groupby('Year_Built').agg({
            'Sale_Price': ['mean', 'median', 'count']
        }).reset_index()
        
        # Flatten MultiIndex columns
        year_stats.columns = ['_'.join(col).strip('_') for col in year_stats.columns.values]
        
        # Create line chart for price trends by year built
        fig = px.line(
            year_stats,
            x='Year_Built',
            y=['Sale_Price_mean', 'Sale_Price_median'],
            labels={
                'Year_Built': get_column_display_label('Year_Built'),
                'Sale_Price_mean': f"Mean {get_column_display_label('Sale_Price')}",
                'Sale_Price_median': f"Median {get_column_display_label('Sale_Price')}",
                'value': f"{get_column_display_label('Sale_Price')} ($)"
            },
            title=f"{get_column_display_label('Sale_Price')} Trends by {get_column_display_label('Year_Built')}",
            markers=True
        )
        
        # Add property count as bar chart on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=year_stats['Year_Built'],
                y=year_stats['Sale_Price_count'],
                name='Number of Properties',
                opacity=0.3,
                yaxis='y2'
            )
        )
        
        # Update layout with secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title='Number of Properties',
                overlaying='y',
                side='right'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.3,
                xanchor='center',
                x=0.5
            )
        )
        
        results['price_by_year'] = fig
        
        # 2. Create heatmap of decade vs. building type if available
        if 'Bldg_Type' in df.columns:
            # Create decade column
            df['Decade'] = (df['Year_Built'] // 10) * 10
            
            # Create pivot table for heatmap
            decade_type_pivot = pd.pivot_table(
                df,
                values='Sale_Price',
                index='Decade',
                columns='Bldg_Type',
                aggfunc='mean'
            ).fillna(0)
            
            # Create heatmap
            fig = px.imshow(
                decade_type_pivot,
                text_auto='.0f',
                labels=dict(
                    x=get_column_display_label('Bldg_Type'),
                    y=get_column_display_label('Decade'),
                    color=f"Avg. {get_column_display_label('Sale_Price')} ($)"
                ),
                title=f"Average {get_column_display_label('Sale_Price')} by {get_column_display_label('Decade')} and {get_column_display_label('Bldg_Type')}",
                color_continuous_scale='Viridis',
                aspect='auto'
            )
            
            fig.update_layout(
                xaxis_title=get_column_display_label('Bldg_Type'),
                yaxis_title=get_column_display_label('Decade')
            )
            
            results['decade_bldg_heatmap'] = fig
    
    # 3. Create age vs price correlation scatter plot
    if 'Sale_Price' in df.columns:
        # Calculate age of property at time of dataset creation
        current_year = pd.to_datetime('now').year
        df['Property_Age'] = current_year - df['Year_Built']
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='Property_Age',
            y='Sale_Price',
            opacity=0.7,
            title=f"{get_column_display_label('Sale_Price')} vs. {get_column_display_label('Property_Age')}",
            labels={
                'Property_Age': f"{get_column_display_label('Property_Age')} (years)",
                'Sale_Price': f"{get_column_display_label('Sale_Price')} ($)"
            },
            trendline='ols',
            trendline_color_override='red'
        )
        
        # Calculate correlation coefficient
        corr = df['Property_Age'].corr(df['Sale_Price'])
        
        # Add annotation with correlation coefficient
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.02, y=0.98,
            text=f'Correlation: {corr:.2f}',
            showarrow=False,
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1,
            borderpad=4
        )
        
        results['age_price_correlation'] = fig
    
    return results
