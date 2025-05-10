"""
Helper classes for visualizations in the dashboard.
This module provides base classes for different visualization types.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple, Union
from dashboard.config import get_colorscale, get_chart_defaults, get_color_palette, get_column_display_label
import numpy as np


class Visualization:
    """Base class for all visualizations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame
        
        Args:
            df: DataFrame containing data to visualize
        """
        self.df = df
    
    def validate_columns(self, required_cols: List[str]) -> Tuple[bool, str]:
        """
        Validate if the DataFrame contains all required columns.
        
        Args:
            required_cols: List of column names that must be present
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {', '.join(missing_cols)}"
            return False, error_msg
        return True, ""
    
    def create_empty_figure(self, message: str) -> go.Figure:
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
    
    def format_hover_data(self, 
                         base_cols: List[str], 
                         optional_cols: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Format hover data for Plotly figures.
        
        Args:
            base_cols: List of column names to always include
            optional_cols: List of column names to include if they exist in the DataFrame
            
        Returns:
            Dictionary suitable for hover_data parameter in Plotly Express
        """
        hover_data = {col: True for col in base_cols if col in self.df.columns}
        
        if optional_cols:
            for col in optional_cols:
                hover_data[col] = True if col in self.df.columns else False
                
        return hover_data
    
    def apply_standard_layout(self, 
                             fig: go.Figure, 
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
            margin=layout_defaults.get("margin", {"r": 10, "t": 40, "l": 10, "b": 10}),
            font=layout_defaults.get("font", {"family": "Open Sans, sans-serif", "size": 12}),
            template=layout_defaults.get("template", "plotly_white")
        )
        
        # Add more layout options based on chart type if needed
        if chart_type == "maps":
            fig.update_layout(
                coloraxis_colorbar=dict(title="Sale Price ($)")
            )
        
        return fig
    
    def generate(self) -> Union[go.Figure, Dict[str, Any]]:
        """
        Generate the visualization. Must be implemented by subclasses.
        
        Returns:
            A Plotly figure or dictionary containing visualization data
        """
        raise NotImplementedError("Subclasses must implement generate()")


class GeographicVisualization(Visualization):
    """Base class for geographic visualizations"""
    
    def get_map_defaults(self):
        """
        Get default settings for maps
        
        Returns:
            Dictionary with map defaults
        """
        map_defaults = get_chart_defaults("maps")
        return {
            "map_style": map_defaults.get("mapbox_style", "open-street-map"),
            "zoom_level": map_defaults.get("zoom", 11),
            "use_clustering": map_defaults.get("use_clustering", True)
        }


class StatisticalVisualization(Visualization):
    """Base class for statistical visualizations"""
    
    def get_stats_defaults(self):
        """
        Get default settings for statistical charts
        
        Returns:
            Dictionary with statistical chart defaults
        """
        stats_defaults = get_chart_defaults("statistics")
        return {
            "bin_size": stats_defaults.get("bin_size", 50000),
            "opacity": stats_defaults.get("opacity", 0.7),
            "color_discrete_sequence": stats_defaults.get("color_discrete_sequence", ['#19A7CE'])
        }


class RelationshipVisualization(Visualization):
    """Base class for relationship visualizations"""
    
    def get_relationship_defaults(self):
        """
        Get default settings for relationship charts
        
        Returns:
            Dictionary with relationship chart defaults
        """
        rel_defaults = get_chart_defaults("relationship")
        return {
            "trendline": rel_defaults.get("trendline", True),
            "opacity": rel_defaults.get("opacity", 0.7),
            "marker_size": rel_defaults.get("marker_size", 7)
        }


class TimeSeriesVisualization(Visualization):
    """Base class for time series visualizations"""
    
    def get_time_defaults(self):
        """
        Get default settings for time series charts
        
        Returns:
            Dictionary with time series chart defaults
        """
        time_defaults = get_chart_defaults("time_series")
        return {
            "moving_avg_window": time_defaults.get("moving_avg_window", 7),
            "line_width": time_defaults.get("line_width", 2),
            "line_color": time_defaults.get("line_color", "blue")
        }


# ---- Concrete Visualization Classes ----

class PriceMap(GeographicVisualization):
    """Housing price map visualization"""
    
    def __init__(self, df: pd.DataFrame, use_clustering: bool = None):
        """
        Initialize price map visualization
        
        Args:
            df: DataFrame with housing data
            use_clustering: Whether to use clustering for better visualization
        """
        super().__init__(df)
        self.use_clustering = use_clustering
        
    def generate(self) -> go.Figure:
        """Generate price map visualization"""
        # Validate required columns
        is_valid, error_msg = self.validate_columns(['Latitude', 'Longitude', 'Sale_Price'])
        if not is_valid:
            return self.create_empty_figure(error_msg)
        
        # Get map configuration settings
        map_defaults = self.get_map_defaults()
        map_style = map_defaults["map_style"]
        zoom_level = map_defaults["zoom_level"]
        
        # Use configuration default if not specified
        if self.use_clustering is None:
            self.use_clustering = map_defaults["use_clustering"]
        
        # Get colorscale from config
        colorscale = get_colorscale("price_map")
        
        # Format hover data
        hover_data = self.format_hover_data(['Sale_Price'], ['Bldg_Type', 'Lot_Area', 'Year_Built'])
        
        if self.use_clustering and len(self.df) > 50:
            # Use a density heatmap approach for many points
            fig = px.density_mapbox(
                self.df,
                lat="Latitude",
                lon="Longitude",
                z="Sale_Price",
                radius=10,
                color_continuous_scale=colorscale,
                zoom=zoom_level,
                mapbox_style=map_style,
                hover_data=hover_data,
                title="Housing Price Density by Location"
            )
            
            # Add a scatter layer with reduced opacity
            scatter_layer = px.scatter_mapbox(
                self.df,
                lat="Latitude",
                lon="Longitude",
                color="Sale_Price",
                size_max=6,
                opacity=0.4,
                zoom=zoom_level,
                mapbox_style=map_style
            ).data[0]
            
            fig.add_trace(scatter_layer)
            
        else:
            # Use regular scatter plot for fewer points
            fig = px.scatter_mapbox(
                self.df,
                lat="Latitude",
                lon="Longitude",
                color="Sale_Price",
                size="Sale_Price",
                color_continuous_scale=colorscale,
                size_max=15,
                zoom=zoom_level,
                mapbox_style=map_style,
                hover_data=hover_data,
                title="Housing Prices by Location"
            )
        
        # Apply standard layout
        fig = self.apply_standard_layout(fig, "Housing Prices by Location", "maps")
        
        return fig


class GooglePriceMap(GeographicVisualization):
    """Google Maps housing price map visualization"""
    
    def __init__(self, df: pd.DataFrame, max_points: int = 1000, use_clustering: bool = True):
        """
        Initialize Google Maps price map visualization
        
        Args:
            df: DataFrame with housing data
            max_points: Maximum number of points to display on map
            use_clustering: Whether to use clustering for better visualization
        """
        super().__init__(df)
        self.max_points = max_points
        self.use_clustering = use_clustering
        
    def generate(self) -> Dict[str, Any]:
        """
        Generate Google Maps visualization data
        
        Returns:
            Dictionary with map configuration data
        """
        # Validate required columns
        is_valid, error_msg = self.validate_columns(['Latitude', 'Longitude', 'Sale_Price'])
        if not is_valid:
            return {"error": error_msg, "data": [], "center": {"lat": 0, "lng": 0}, "zoom": 10}
        
        # DEBUGGING: Print data summary before processing
        print(f"GooglePriceMap: Processing {len(self.df)} records")
        
        # IMPORTANT: Ensure latitude and longitude are proper numeric values
        # Convert to float explicitly to avoid type issues
        if 'Latitude' in self.df.columns and 'Longitude' in self.df.columns:
            self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
            self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
            
            # Print sample of lat/long values for verification
            print(f"Sample latitude values: {self.df['Latitude'].head(3).tolist()}")
            print(f"Sample longitude values: {self.df['Longitude'].head(3).tolist()}")
        
        # Clean data to ensure no null values or invalid types
        map_data = self.df.dropna(subset=['Latitude', 'Longitude', 'Sale_Price']).copy()
        
        # Ensure we have data after cleaning
        if len(map_data) == 0:
            return {"error": "No valid data points found after cleaning", "data": [], 
                   "center": {"lat": 0, "lng": 0}, "zoom": 10}
                   
        print(f"GooglePriceMap: After cleaning, {len(map_data)} valid records remain")
        
        # Apply sampling for large datasets to prevent browser freezing
        original_count = len(map_data)
        need_sampling = original_count > self.max_points
        
        if need_sampling:
            # Use either random sampling or a smart sampling strategy
            if self.use_clustering and original_count > self.max_points * 2:
                # Use K-means clustering for smart sampling when dataset is very large
                try:
                    from sklearn.cluster import KMeans
                    
                    # Select only the geospatial columns for clustering
                    geo_data = map_data[['Latitude', 'Longitude']].copy()
                    
                    # Determine number of clusters based on data size
                    n_clusters = min(self.max_points, len(geo_data) // 3)
                    
                    # Fit K-means to find cluster centers
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    map_data['Cluster'] = kmeans.fit_predict(geo_data)
                    
                    # Sample from each cluster proportionally
                    sampled_data = []
                    for cluster_id in range(n_clusters):
                        cluster_points = map_data[map_data['Cluster'] == cluster_id]
                        sample_size = max(1, int((len(cluster_points) / original_count) * self.max_points))
                        cluster_sample = cluster_points.sample(n=min(sample_size, len(cluster_points)))
                        sampled_data.append(cluster_sample)
                    
                    # Combine sampled points from each cluster
                    map_data = pd.concat(sampled_data)
                    
                    # Add clustering info to the result
                    clustering_applied = True
                    
                except (ImportError, Exception) as e:
                    # Fall back to random sampling if clustering fails
                    print(f"Clustering failed, falling back to random sampling: {str(e)}")
                    map_data = map_data.sample(n=self.max_points, random_state=42)
                    clustering_applied = False
            else:
                # Use simple random sampling
                map_data = map_data.sample(n=self.max_points, random_state=42)
                clustering_applied = False
            
            print(f"Applied sampling to Google Maps data: {original_count} -> {len(map_data)} points")
        else:
            clustering_applied = False
        
        # CRITICAL: Ensure data types for proper JSON serialization
        # Explicitly convert to float and validate
        map_data['Latitude'] = map_data['Latitude'].astype(float)
        map_data['Longitude'] = map_data['Longitude'].astype(float)
        map_data['Sale_Price'] = map_data['Sale_Price'].astype(float)
        
        # Verify no NaN values are present
        map_data = map_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Latitude', 'Longitude', 'Sale_Price'])
        
        # Add ID for each point
        map_data['id'] = range(len(map_data))
        
        # Add additional properties for display if available
        optional_columns = ['Bldg_Type', 'Year_Built', 'Lot_Area', 'Neighborhood']
        for col in optional_columns:
            if col in self.df.columns:
                map_data[col] = self.df[col]
        
        # Apply user-friendly building type labels if available
        if 'Bldg_Type' in map_data.columns:
            from dashboard.config import get_building_type_label
            map_data['Bldg_Type_Display'] = map_data['Bldg_Type'].apply(get_building_type_label)
            # Keep both columns - original for data integrity and display for UI
        
        # Get map configuration settings
        map_defaults = self.get_map_defaults()
        zoom_level = map_defaults["zoom_level"]
        
        # Calculate map center
        lat_mean = float(map_data['Latitude'].mean())
        lng_mean = float(map_data['Longitude'].mean())
        
        # DEBUGGING: Print coordinate range to verify validity
        lat_min, lat_max = float(map_data['Latitude'].min()), float(map_data['Latitude'].max())
        lng_min, lng_max = float(map_data['Longitude'].min()), float(map_data['Longitude'].max())
        print(f"Latitude range: {lat_min} to {lat_max}")
        print(f"Longitude range: {lng_min} to {lng_max}")
        
        # Add data statistics that might be useful for the frontend
        stats = {
            "price_min": float(map_data['Sale_Price'].min()),
            "price_max": float(map_data['Sale_Price'].max()),
            "price_avg": float(map_data['Sale_Price'].mean()),
            "count": len(map_data),
            "total_count": original_count,
            "sampled": need_sampling,
            "clustering_applied": clustering_applied,
            "use_marker_clustering": self.use_clustering
        }
        
        # Convert to records, ensuring each record has valid coordinates
        data_records = []
        for _, row in map_data.iterrows():
            try:
                record = row.to_dict()
                # Ensure coordinates are valid numbers
                if (isinstance(record['Latitude'], float) and 
                    isinstance(record['Longitude'], float) and
                    not pd.isna(record['Latitude']) and 
                    not pd.isna(record['Longitude'])):
                    data_records.append(record)
            except Exception as e:
                print(f"Error processing row: {e}")
        
        print(f"Prepared Google Maps data with {len(data_records)} properties" + 
              (f" (sampled from {original_count})" if need_sampling else ""))
        
        if len(data_records) > 0:
            print(f"First data point: Lat={data_records[0]['Latitude']}, Lng={data_records[0]['Longitude']}")
        
        # Add timestamp to ensure each output is unique
        import time
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
        
        return {
            "data": data_records,
            "center": {
                "lat": lat_mean,
                "lng": lng_mean
            },
            "zoom": zoom_level,
            "stats": stats,
            "use_clustering": self.use_clustering,
            "timestamp": timestamp  # Add timestamp to make each output unique
        }


class PriceDistribution(StatisticalVisualization):
    """Housing price distribution visualization"""
    
    def __init__(self, df: pd.DataFrame, bin_size: int = None):
        """
        Initialize price distribution visualization
        
        Args:
            df: DataFrame with housing data
            bin_size: Size of histogram bins
        """
        super().__init__(df)
        self.bin_size = bin_size
        
    def generate(self) -> go.Figure:
        """Generate price distribution visualization"""
        # Validate required columns
        is_valid, error_msg = self.validate_columns(['Sale_Price'])
        if not is_valid:
            return self.create_empty_figure(error_msg)
        
        # Get stats defaults
        stats_defaults = self.get_stats_defaults()
        bin_size = self.bin_size or stats_defaults["bin_size"]
        colors = stats_defaults["color_discrete_sequence"]
        
        # Create histogram
        fig = px.histogram(
            self.df,
            x="Sale_Price",
            nbins=int((self.df["Sale_Price"].max() - self.df["Sale_Price"].min()) / bin_size),
            title=f"Distribution of {get_column_display_label('Sale_Price')}",
            labels={"Sale_Price": get_column_display_label("Sale_Price")},
            opacity=stats_defaults["opacity"],
            color_discrete_sequence=colors
        )
        
        # Add a vertical line for the mean price
        mean_price = self.df["Sale_Price"].mean()
        fig.add_vline(
            x=mean_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mean_price:.0f}",
            annotation_position="top right"
        )
        
        # Apply standard layout
        fig = self.apply_standard_layout(fig, f"Distribution of {get_column_display_label('Sale_Price')}", "statistics")
        fig.update_layout(
            xaxis_title=get_column_display_label("Sale_Price"),
            yaxis_title="Number of Properties",
            bargap=0.1
        )
        
        return fig


class ScatterComparison(RelationshipVisualization):
    """Scatter plot comparison visualization"""
    
    def __init__(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None, trendline: bool = None):
        """
        Initialize scatter plot visualization
        
        Args:
            df: DataFrame with housing data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            color_col: Column name for color coding points (optional)
            trendline: Whether to add a trendline
        """
        super().__init__(df)
        self.x_col = x_col
        self.y_col = y_col
        self.color_col = color_col
        self.trendline = trendline
        
    def generate(self) -> go.Figure:
        """Generate scatter plot visualization"""
        # Validate required columns
        is_valid, error_msg = self.validate_columns([self.x_col, self.y_col])
        if not is_valid:
            return self.create_empty_figure(error_msg)
        
        # Get relationship defaults
        rel_defaults = self.get_relationship_defaults()
        trendline = self.trendline if self.trendline is not None else rel_defaults["trendline"]
        
        # Create scatter plot
        fig = px.scatter(
            self.df,
            x=self.x_col,
            y=self.y_col,
            color=self.color_col if self.color_col in self.df.columns else None,
            opacity=rel_defaults["opacity"],
            title=f"{get_column_display_label(self.y_col)} vs {get_column_display_label(self.x_col)}",
            labels={
                self.x_col: get_column_display_label(self.x_col), 
                self.y_col: get_column_display_label(self.y_col),
                self.color_col: get_column_display_label(self.color_col) if self.color_col in self.df.columns else None
            },
            hover_data=self.df.columns[:5],  # Include first 5 columns in hover data
            trendline='ols' if trendline else None,
            trendline_color_override="red" if trendline else None
        )
        
        # Apply additional styling
        fig.update_traces(marker=dict(
            size=rel_defaults["marker_size"], 
            line=dict(width=1, color='DarkSlateGrey')
        ))
        
        # Apply standard layout
        fig = self.apply_standard_layout(fig, f"{get_column_display_label(self.y_col)} vs {get_column_display_label(self.x_col)}", "relationship")
        fig.update_layout(
            xaxis_title=get_column_display_label(self.x_col),
            yaxis_title=get_column_display_label(self.y_col),
            legend_title=get_column_display_label(self.color_col) if self.color_col else None
        )
        
        return fig


class TimeSeries(TimeSeriesVisualization):
    """Time series visualization"""
    
    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str, group_col: Optional[str] = None):
        """
        Initialize time series visualization
        
        Args:
            df: DataFrame with housing data
            date_col: Column name containing date information
            value_col: Column name containing the values to plot
            group_col: Column name to group by (optional)
        """
        super().__init__(df)
        self.date_col = date_col
        self.value_col = value_col
        self.group_col = group_col
        
    def generate(self) -> go.Figure:
        """Generate time series visualization"""
        # Validate required columns
        is_valid, error_msg = self.validate_columns([self.date_col, self.value_col])
        if not is_valid:
            return self.create_empty_figure(error_msg)
        
        # Ensure date column is datetime
        try:
            df = self.df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        except:
            return self.create_empty_figure(f"Could not convert {self.date_col} to datetime format")
        
        # Get time series defaults
        time_defaults = self.get_time_defaults()
        
        # Create time series plot
        if self.group_col and self.group_col in df.columns:
            fig = px.line(
                df,
                x=self.date_col,
                y=self.value_col,
                color=self.group_col,
                title=f"{get_column_display_label(self.value_col)} Over Time by {get_column_display_label(self.group_col)}",
                labels={
                    self.date_col: get_column_display_label(self.date_col),
                    self.value_col: get_column_display_label(self.value_col),
                    self.group_col: get_column_display_label(self.group_col)
                }
            )
        else:
            fig = px.line(
                df,
                x=self.date_col,
                y=self.value_col,
                title=f"{get_column_display_label(self.value_col)} Over Time",
                labels={
                    self.date_col: get_column_display_label(self.date_col),
                    self.value_col: get_column_display_label(self.value_col)
                }
            )
            
            # Add moving average trendline
            df_sorted = df.sort_values(by=self.date_col)
            window_size = time_defaults["moving_avg_window"] or max(7, len(df) // 20)
            
            if len(df) > window_size:
                moving_avg = df_sorted[self.value_col].rolling(window=window_size).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted[self.date_col],
                        y=moving_avg,
                        mode='lines',
                        line=dict(
                            color=time_defaults["line_color"], 
                            width=time_defaults["line_width"]
                        ),
                        name=f'{window_size}-point Moving Average'
                    )
                )
        
        # Apply standard layout
        fig = self.apply_standard_layout(fig, f"{get_column_display_label(self.value_col)} Over Time", "time_series")
        fig.update_layout(
            xaxis_title=get_column_display_label(self.date_col),
            yaxis_title=get_column_display_label(self.value_col)
        )
        
        return fig


class CorrelationHeatmap(RelationshipVisualization):
    """Correlation heatmap visualization"""
    
    def __init__(self, df: pd.DataFrame, columns: List[str] = None):
        """
        Initialize correlation heatmap visualization
        
        Args:
            df: DataFrame with housing data
            columns: List of columns to include in the heatmap (optional)
        """
        super().__init__(df)
        self.columns = columns
        
    def generate(self) -> go.Figure:
        """Generate correlation heatmap visualization"""
        # Use only numeric columns
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if self.columns:
            # Filter to only specified columns that are also numeric
            numeric_df = numeric_df[[col for col in self.columns if col in numeric_df.columns]]
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return self.create_empty_figure("Not enough numeric columns for correlation analysis")
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",  # This ensures the cells are auto-sized
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap",
            labels=dict(color="Correlation")
        )
        
        # Apply standard layout
        fig = self.apply_standard_layout(fig, "Correlation Heatmap", "relationship")
        
        # Make the figure responsive to container size
        fig.update_layout(
            height=800,  # Maintain height for readability
            autosize=True,  # Enable autosize to fill container
            margin={"r": 20, "t": 50, "l": 20, "b": 20}  # Adjust margins for better fit
        )
        
        return fig


class BoxPlotVisualization(StatisticalVisualization):
    """Box plot visualization for comparing distributions across categories"""
    
    def __init__(self, df: pd.DataFrame, numeric_col: str, category_col: str):
        """
        Initialize box plot visualization
        
        Args:
            df: DataFrame with housing data
            numeric_col: Column name for the numeric variable
            category_col: Column name for the categorical variable
        """
        super().__init__(df)
        self.numeric_col = numeric_col
        self.category_col = category_col
        
    def generate(self) -> go.Figure:
        """Generate box plot visualization"""
        # Validate required columns
        is_valid, error_msg = self.validate_columns([self.numeric_col, self.category_col])
        if not is_valid:
            return self.create_empty_figure(error_msg)
        
        # Apply building type labels if the category column is Bldg_Type
        if self.category_col == 'Bldg_Type':
            from dashboard.config import get_building_type_label
            df_plot = self.df.copy()
            
            # Create a new column for display with friendly building type names
            df_plot['Bldg_Type_Display'] = df_plot['Bldg_Type'].apply(get_building_type_label)
            
            # Use the display column for visualization but keep original for data
            category_col_display = 'Bldg_Type_Display'
        else:
            df_plot = self.df.copy()
            category_col_display = self.category_col
        
        # For better visualization, limit to top categories if there are too many
        value_counts = df_plot[self.category_col].value_counts()
        if len(value_counts) > 10:
            top_categories = value_counts.nlargest(10).index.tolist()
            filtered_df = df_plot[df_plot[self.category_col].isin(top_categories)].copy()
            if self.category_col == 'Bldg_Type':
                # Keep the display names but add space to preserve category order
                filtered_df[category_col_display] = filtered_df[category_col_display].astype(str) + " "
            else:
                filtered_df[category_col_display] = filtered_df[category_col_display].astype(str) + " "
        else:
            filtered_df = df_plot.copy()
        
        # Get stats defaults
        stats_defaults = self.get_stats_defaults()
            
        fig = px.box(
            filtered_df,
            x=category_col_display,
            y=self.numeric_col,
            color=category_col_display,
            title=f"Distribution of {get_column_display_label(self.numeric_col)} by {get_column_display_label(self.category_col)}",
            labels={
                self.numeric_col: get_column_display_label(self.numeric_col),
                category_col_display: get_column_display_label(self.category_col)
            },
            points="outliers"  # Only show outlier points
        )
        
        # Apply standard layout
        fig = self.apply_standard_layout(fig, f"Distribution of {get_column_display_label(self.numeric_col)} by {get_column_display_label(self.category_col)}", "statistics")
        fig.update_layout(
            xaxis_title=get_column_display_label(self.category_col),
            yaxis_title=get_column_display_label(self.numeric_col),
            showlegend=False,  # Hide legend as it's redundant with x-axis
            xaxis={'categoryorder': 'total descending'}  # Order categories by total value
        )
        
        return fig