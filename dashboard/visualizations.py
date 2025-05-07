"""
Visualizations module for the dashboard.
This module provides functions to generate Plotly figures for the dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
from dashboard.config import get_colorscale, get_chart_defaults, get_color_palette


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


def generate_price_distribution(df: pd.DataFrame, bin_size: int = 50000) -> go.Figure:
    """
    Generate a histogram showing the distribution of housing prices.
    
    Args:
        df: DataFrame containing housing data with Sale_Price column
        bin_size: Size of histogram bins
        
    Returns:
        Plotly figure object with the price distribution
    """
    from dashboard.visualizations_helpers import PriceDistribution
    
    # Use the PriceDistribution class to generate the visualization
    price_dist = PriceDistribution(df, bin_size)
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
    Generate summary statistics for dashboard cards.
    
    Args:
        df: DataFrame containing housing data
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    # Total properties
    summary["total_properties"] = {
        "value": len(df),
        "description": "Total Properties"
    }
    
    # Average price if Sale_Price column exists
    if 'Sale_Price' in df.columns:
        summary["avg_price"] = {
            "value": f"${df['Sale_Price'].mean():,.0f}",
            "description": "Average Price"
        }
        
        summary["median_price"] = {
            "value": f"${df['Sale_Price'].median():,.0f}",
            "description": "Median Price"
        }
        
        summary["price_range"] = {
            "value": f"${df['Sale_Price'].min():,.0f} - ${df['Sale_Price'].max():,.0f}",
            "description": "Price Range"
        }
    
    # Average area if Lot_Area column exists
    if 'Lot_Area' in df.columns:
        summary["avg_area"] = {
            "value": f"{df['Lot_Area'].mean():,.0f} sq.ft",
            "description": "Average Lot Area"
        }
    
    # Building type distribution if Bldg_Type column exists
    if 'Bldg_Type' in df.columns:
        most_common_type = df['Bldg_Type'].value_counts().idxmax()
        type_percentage = (df['Bldg_Type'].value_counts().max() / len(df)) * 100
        
        summary["common_type"] = {
            "value": f"{most_common_type} ({type_percentage:.1f}%)",
            "description": "Most Common Building Type"
        }
    
    return summary


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
    if compare_col not in df.columns:
        return {"error": _create_empty_figure(f"Comparison column {compare_col} not found")}
    
    results = {}
    
    # Default metrics if not specified
    if not metrics:
        available_cols = df.columns.tolist()
        possible_metrics = [
            "Sale_Price", "Lot_Area", "Lot_Frontage", "Year_Built", 
            "Total_Bsmt_SF", "First_Flr_SF", "Full_Bath", "Half_Bath", 
            "Bedroom_AbvGr", "Fireplaces"
        ]
        metrics = [col for col in possible_metrics if col in available_cols][:5]  # Use first 5 available
    
    # Group comparison (boxplot for distribution comparison)
    for metric in metrics:
        if metric in df.columns and df[metric].dtype in ['int64', 'float64']:
            # Create grouped box plot
            fig = px.box(
                df,
                x=compare_col,
                y=metric,
                color=compare_col,
                title=f"{metric} by {compare_col}",
                labels={
                    metric: metric.replace('_', ' '),
                    compare_col: compare_col.replace('_', ' ')
                },
                points="outliers"
            )
            
            fig.update_layout(
                xaxis_title=compare_col.replace('_', ' '),
                yaxis_title=metric.replace('_', ' '),
                showlegend=False
            )
            
            results[f"{metric}_box"] = fig
            
            # Create grouped bar chart for averages
            avg_by_group = df.groupby(compare_col)[metric].mean().reset_index()
            count_by_group = df.groupby(compare_col)[metric].count().reset_index()
            
            avg_by_group['count'] = count_by_group[metric]
            avg_by_group['label'] = avg_by_group[compare_col] + ' (n=' + avg_by_group['count'].astype(str) + ')'
            
            fig = px.bar(
                avg_by_group,
                x=compare_col,
                y=metric,
                color=compare_col,
                text_auto=True,
                title=f"Average {metric} by {compare_col}",
                labels={
                    metric: f"Avg. {metric.replace('_', ' ')}",
                    compare_col: compare_col.replace('_', ' ')
                },
                hover_data={
                    'count': True,
                    'label': False,
                    compare_col: False
                }
            )
            
            fig.update_layout(
                xaxis_title="",
                yaxis_title=f"Average {metric.replace('_', ' ')}",
                showlegend=False,
                xaxis={'categoryorder': 'total descending'}
            )
            
            results[f"{metric}_bar"] = fig
    
    # If Sale_Price is available, create a scatter plot showing price vs area colored by comparison column
    if "Sale_Price" in df.columns and "Lot_Area" in df.columns:
        fig = px.scatter(
            df,
            x="Lot_Area", 
            y="Sale_Price",
            color=compare_col,
            opacity=0.7,
            title=f"Price vs Area by {compare_col}",
            labels={
                "Lot_Area": "Lot Area (sq.ft)",
                "Sale_Price": "Sale Price ($)",
                compare_col: compare_col.replace('_', ' ')
            },
            trendline="ols",
            trendline_scope="overall"
        )
        
        fig.update_layout(
            legend_title=compare_col.replace('_', ' ')
        )
        
        results["price_vs_area"] = fig
    
    # Create a radar chart comparing the average metrics by property type
    if len(metrics) >= 3:
        # Filter to only numeric metrics
        numeric_metrics = [m for m in metrics if df[m].dtype in ['int64', 'float64']][:5]  # Limit to 5 metrics
        
        if len(numeric_metrics) >= 3:
            # Create radar chart data
            groups = df[compare_col].unique().tolist()
            
            # Normalize each metric to 0-1 scale for fair comparison
            radar_df = df.copy()
            for metric in numeric_metrics:
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:  # Avoid division by zero
                    radar_df[metric] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    radar_df[metric] = 0  # All values are the same
            
            # Calculate averages by group
            group_avgs = radar_df.groupby(compare_col)[numeric_metrics].mean().reset_index()
            
            # Create radar chart
            fig = go.Figure()
            
            for i, group in enumerate(groups):
                group_data = group_avgs[group_avgs[compare_col] == group]
                if not group_data.empty:
                    fig.add_trace(go.Scatterpolar(
                        r=group_data[numeric_metrics].values.flatten().tolist(),
                        theta=numeric_metrics,
                        fill='toself',
                        name=group
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title=f"Property Metrics Comparison by {compare_col} (Normalized)",
                showlegend=True
            )
            
            results["radar_comparison"] = fig
    
    return results


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
        results["error"] = _create_empty_figure("Year trend analysis requires Year_Built column")
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
        results["error"] = _create_empty_figure("No valid year data available")
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
                'Year_Built': 'Year Built',
                'Sale_Price_mean': 'Mean Price',
                'Sale_Price_median': 'Median Price',
                'value': 'Sale Price ($)'
            },
            title='Price Trends by Year Built',
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
                    x='Building Type',
                    y='Decade',
                    color='Avg. Price ($)'
                ),
                title='Average Price by Decade and Building Type',
                color_continuous_scale='Viridis',
                aspect='auto'
            )
            
            fig.update_layout(
                xaxis_title='Building Type',
                yaxis_title='Decade'
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
            title='Property Price vs. Age',
            labels={
                'Property_Age': 'Property Age (years)',
                'Sale_Price': 'Sale Price ($)'
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
