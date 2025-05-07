"""
Visualizations module for the dashboard.
This module provides functions to generate Plotly figures for the dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple


def generate_price_map(df: pd.DataFrame, use_clustering: bool = True) -> go.Figure:
    """
    Generate a map visualization of housing prices by location.
    
    Args:
        df: DataFrame containing housing data with Latitude and Longitude columns
        use_clustering: Whether to use clustering for better visualization of many points
        
    Returns:
        Plotly figure object with the map visualization
    """
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns or 'Sale_Price' not in df.columns:
        # Return empty figure if required columns are not present
        return go.Figure().update_layout(
            title="Map visualization requires Latitude, Longitude, and Sale_Price columns"
        )
    
    if use_clustering and len(df) > 50:
        # Use a density heatmap approach for many points to avoid overcrowding
        fig = px.density_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            z="Sale_Price",
            radius=10,
            color_continuous_scale=px.colors.sequential.Viridis,
            zoom=11,
            mapbox_style="open-street-map",
            hover_data={
                'Sale_Price': True,
                'Bldg_Type': True if 'Bldg_Type' in df.columns else False,
                'Year_Built': True if 'Year_Built' in df.columns else False
            },
            title="Housing Price Density by Location"
        )
        
        # Add a scatter layer with reduced opacity for individual points
        scatter_layer = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            color="Sale_Price",
            size_max=6,
            opacity=0.4,
            zoom=11,
            mapbox_style="open-street-map"
        ).data[0]
        
        fig.add_trace(scatter_layer)
        
    else:
        # Use regular scatter plot for fewer points
        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            color="Sale_Price",
            size="Sale_Price",
            color_continuous_scale=px.colors.sequential.Viridis,
            size_max=15,
            zoom=11,
            mapbox_style="open-street-map",
            hover_data={
                'Sale_Price': True,
                'Lot_Area': True if 'Lot_Area' in df.columns else False,
                'Bldg_Type': True if 'Bldg_Type' in df.columns else False,
                'Year_Built': True if 'Year_Built' in df.columns else False
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
        return {"error": go.Figure().update_layout(title=f"Comparison column {compare_col} not found")}
    
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
        results["error"] = go.Figure().update_layout(
            title="Year trend analysis requires Year_Built column"
        )
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
        results["error"] = go.Figure().update_layout(
            title="No valid year data available"
        )
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
