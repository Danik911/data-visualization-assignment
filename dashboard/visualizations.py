"""
Visualizations module for the dashboard.
This module provides functions to generate Plotly figures for the dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple


def generate_price_map(df: pd.DataFrame) -> go.Figure:
    """
    Generate a map visualization of housing prices by location.
    
    Args:
        df: DataFrame containing housing data with Latitude and Longitude columns
        
    Returns:
        Plotly figure object with the map visualization
    """
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns or 'Sale_Price' not in df.columns:
        # Return empty figure if required columns are not present
        return go.Figure().update_layout(
            title="Map visualization requires Latitude, Longitude, and Sale_Price columns"
        )
    
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Sale_Price",
        size="Sale_Price",
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=15,
        zoom=10,
        mapbox_style="open-street-map",
        hover_data={
            'Sale_Price': True,
            'Lot_Area': True,
            'Bldg_Type': True
        },
        title="Housing Prices by Location"
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Sale Price ($)")
    )
    
    return fig


def generate_price_distribution(df: pd.DataFrame, bin_size: int = 50000) -> go.Figure:
    """
    Generate a histogram showing the distribution of housing prices.
    
    Args:
        df: DataFrame containing housing data with Sale_Price column
        bin_size: Size of histogram bins
        
    Returns:
        Plotly figure object with the price distribution
    """
    if 'Sale_Price' not in df.columns:
        # Return empty figure if required column is not present
        return go.Figure().update_layout(
            title="Price distribution requires Sale_Price column"
        )
    
    fig = px.histogram(
        df,
        x="Sale_Price",
        nbins=int((df["Sale_Price"].max() - df["Sale_Price"].min()) / bin_size),
        title="Distribution of Housing Prices",
        labels={"Sale_Price": "Sale Price ($)"},
        opacity=0.8,
        color_discrete_sequence=['#19A7CE']
    )
    
    fig.update_layout(
        xaxis_title="Sale Price ($)",
        yaxis_title="Number of Properties",
        bargap=0.1
    )
    
    # Add a vertical line for the mean price
    mean_price = df["Sale_Price"].mean()
    fig.add_vline(
        x=mean_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${mean_price:.0f}",
        annotation_position="top right"
    )
    
    return fig


def generate_correlation_heatmap(df: pd.DataFrame, columns: List[str] = None) -> go.Figure:
    """
    Generate a correlation heatmap for numeric columns.
    
    Args:
        df: DataFrame containing housing data
        columns: List of columns to include in the heatmap (optional)
        
    Returns:
        Plotly figure object with the correlation heatmap
    """
    # Use only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if columns:
        # Filter to only specified columns that are also numeric
        numeric_df = numeric_df[[col for col in columns if col in numeric_df.columns]]
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap",
        labels=dict(color="Correlation")
    )
    
    fig.update_layout(
        height=800,  # Larger size to ensure readability
        width=800
    )
    
    return fig


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
    if x_col not in df.columns or y_col not in df.columns:
        # Return empty figure if required columns are not present
        return go.Figure().update_layout(
            title=f"Scatter plot requires both {x_col} and {y_col} columns"
        )
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in df.columns else None,
        opacity=0.7,
        title=f"{y_col} vs {x_col}",
        labels={x_col: x_col.replace('_', ' '), y_col: y_col.replace('_', ' ')},
        hover_data=df.columns[:5],  # Include first 5 columns in hover data
        trendline='ols' if trendline else None,
        trendline_color_override="red" if trendline else None
    )
    
    fig.update_traces(marker=dict(size=7, line=dict(width=1, color='DarkSlateGrey')))
    
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' '),
        yaxis_title=y_col.replace('_', ' '),
        legend_title=color_col.replace('_', ' ') if color_col else None
    )
    
    return fig


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
    if numeric_col not in df.columns or category_col not in df.columns:
        # Return empty figure if required columns are not present
        return go.Figure().update_layout(
            title=f"Box plot requires both {numeric_col} and {category_col} columns"
        )
    
    # For better visualization, limit to top categories if there are too many
    value_counts = df[category_col].value_counts()
    if len(value_counts) > 10:
        top_categories = value_counts.nlargest(10).index.tolist()
        filtered_df = df[df[category_col].isin(top_categories)].copy()
        filtered_df[category_col] = filtered_df[category_col].astype(str) + " "  # Add space to preserve category order
    else:
        filtered_df = df.copy()
        
    fig = px.box(
        filtered_df,
        x=category_col,
        y=numeric_col,
        color=category_col,
        title=f"Distribution of {numeric_col} by {category_col}",
        labels={
            numeric_col: numeric_col.replace('_', ' '),
            category_col: category_col.replace('_', ' ')
        },
        points="outliers"  # Only show outlier points
    )
    
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' '),
        yaxis_title=numeric_col.replace('_', ' '),
        showlegend=False,  # Hide legend as it's redundant with x-axis
        xaxis={'categoryorder': 'total descending'}  # Order categories by total value
    )
    
    return fig


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
    if date_col not in df.columns or value_col not in df.columns:
        # Return empty figure if required columns are not present
        return go.Figure().update_layout(
            title=f"Time series requires both {date_col} and {value_col} columns"
        )
    
    # Ensure date column is datetime
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    except:
        return go.Figure().update_layout(
            title=f"Could not convert {date_col} to datetime format"
        )
    
    # Create time series plot
    if group_col and group_col in df.columns:
        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            color=group_col,
            title=f"{value_col} Over Time by {group_col}",
            labels={
                date_col: date_col.replace('_', ' '),
                value_col: value_col.replace('_', ' '),
                group_col: group_col.replace('_', ' ')
            }
        )
    else:
        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            title=f"{value_col} Over Time",
            labels={
                date_col: date_col.replace('_', ' '),
                value_col: value_col.replace('_', ' ')
            }
        )
        
        # Add moving average trendline
        df_sorted = df.sort_values(by=date_col)
        window_size = max(7, len(df) // 20)  # Use either 7 or 5% of data points
        
        if len(df) > window_size:
            moving_avg = df_sorted[value_col].rolling(window=window_size).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[date_col],
                    y=moving_avg,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'{window_size}-point Moving Average'
                )
            )
    
    fig.update_layout(
        xaxis_title=date_col.replace('_', ' '),
        yaxis_title=value_col.replace('_', ' ')
    )
    
    return fig


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
            return go.Figure().update_layout(
                title=f"Feature importance requires numeric target column: {target_col}"
            )
        
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
        return go.Figure().update_layout(
            title=f"Missing columns for parallel coordinates: {', '.join(missing_columns)}"
        )
    
    # Create a copy of the dataframe with only the needed columns
    plot_df = df[columns].copy()
    
    # If color column is specified and exists, add it
    if color_col and color_col in df.columns:
        plot_df[color_col] = df[color_col]
        
        fig = px.parallel_coordinates(
            plot_df,
            color=color_col,
            labels={col: col.replace('_', ' ') for col in plot_df.columns},
            title="Parallel Coordinates Plot",
            color_continuous_scale=px.colors.sequential.Viridis
        )
    else:
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