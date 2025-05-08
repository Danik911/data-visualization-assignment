"""
Configuration module for the dashboard.
This module defines settings, themes, and defaults for the dashboard application.
"""

import os
from typing import Dict, List, Any, Optional

# Environment settings
ENV = os.environ.get("DASHBOARD_ENV", "development")
DEBUG = ENV == "development"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.environ.get("DASHBOARD_PORT", "8050"))
DEFAULT_DATA_PATH = "data/Housing Data_cleaned_for_dashboard.csv"
METADATA_PATH = "data/Housing Data_dashboard_metadata.json"
ASSETS_FOLDER = "dashboard/assets"

# Theme settings
THEMES = {
    "default": {
        "external_stylesheets": ["https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css"],
        "template": "plotly_white",
        "primary_color": "#2c6e91",  # Dark blue - matches our new CSS theme
        "secondary_color": "#6c7a89",  # Secondary gray-blue
        "accent_color": "#e67e22",     # Accent orange
        "text_color": "#2c3e50",       # Dark text
        "background_color": "#f9fafb"  # Light background
    },
    "dark": {
        "external_stylesheets": ["https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css"],
        "template": "plotly_dark",
        "primary_color": "#6C63FF",  # Purple
        "secondary_color": "#4641AA",  # Dark purple
        "accent_color": "#FFD166",  # Yellow
        "text_color": "#E0E0E0",
        "background_color": "#222222"
    }
}

# Set active theme
ACTIVE_THEME = "default"

# Color settings for visualizations
VISUALIZATION_COLORS = {
    "sequential": [
        "#2c6e91",  # dark blue - primary
        "#3498db",  # medium blue
        "#6c7a89",  # gray blue - secondary
        "#a3c9e3"   # light blue
    ],
    "categorical": [
        "#2c6e91",  # dark blue - primary
        "#e67e22",  # orange - accent
        "#27ae60",  # green - success
        "#8e44ad",  # purple
        "#f39c12",  # yellow
        "#16a085",  # teal
        "#c0392b",  # red
        "#7f8c8d",  # gray
        "#3498db",  # medium blue
        "#d35400"   # dark orange
    ],
    "diverging": [
        "#c0392b",  # red
        "#e74c3c",  # light red
        "#f9fafb",  # white - bg light
        "#3498db",  # medium blue
        "#2c6e91"   # dark blue - primary
    ],
    "colorscales": {
        "price_map": "Blues",        # Changed from Viridis to Blues for real estate theme
        "correlation": "RdBu_r",
        "feature_importance": "Blues" # Changed from Viridis to Blues for consistency
    }
}

# Default chart settings
CHART_DEFAULTS = {
    "layout": {
        "margin": {"r": 20, "t": 40, "l": 20, "b": 20},  # Improved margins for readability
        "font": {"family": "Nunito Sans, Open Sans, sans-serif", "size": 12},  # Matching our new font
        "legend": {"orientation": "h", "y": -0.2},
        "colorway": VISUALIZATION_COLORS["categorical"],
        "paper_bgcolor": "rgba(0,0,0,0)",  # Transparent background
        "plot_bgcolor": "rgba(0,0,0,0)",   # Transparent background
        "title_font": {"family": "Nunito Sans, sans-serif", "size": 16, "color": "#2c6e91"} # Matching header style
    },
    "maps": {
        "mapbox_style": "open-street-map",
        "zoom": 11,
        "use_clustering": True
    },
    "histograms": {
        "bin_size": 50000,
        "opacity": 0.85,  # Increased opacity for better visibility
        "marker": {
            "line": {"width": 1, "color": "#ffffff"}  # White border for better separation
        }
    },
    "scatter": {
        "marker_size": 8,  # Slightly larger markers
        "opacity": 0.75,
        "trendline": True,
        "marker": {
            "line": {"width": 1, "color": "#ffffff"}  # White border for better visibility
        }
    },
    "box_plots": {
        "points": "outliers",
        "notched": False,
        "line_width": 2  # Thicker lines for better visibility
    },
    "bar_charts": {
        "opacity": 0.85,
        "text_auto": True,
        "marker": {
            "line": {"width": 1, "color": "#ffffff"}  # White border for better separation
        }
    }
}

# Default filter settings
DEFAULT_FILTERS = {
    "price_range": [0, 1000000],
    "bedrooms": [1, 6],
    "bathrooms": [1, 4],
    "building_types": ["Single-family", "Multi-family", "Apartment", "Condo"],
    "year_built_min": 1900,
    "max_results": 1000
}

# Default visualizations to display
DEFAULT_VISUALIZATIONS = [
    "price_map",
    "price_distribution",
    "price_by_feature_correlation",
    "building_type_comparison"
]

# Tab configuration
TABS_CONFIG = [
    {
        "id": "overview",
        "label": "Overview",
        "icon": "fas fa-home",
        "default_charts": ["price_map", "price_distribution", "summary_stats", "year_built_histogram"]
    },
    {
        "id": "property_analysis",
        "label": "Property Analysis",
        "icon": "fas fa-chart-bar",
        "default_charts": ["feature_importance", "price_scatters", "property_type_comparison"]
    },
    {
        "id": "market_trends",
        "label": "Market Trends",
        "icon": "fas fa-chart-line",
        "default_charts": ["price_by_year", "decade_bldg_heatmap", "age_price_correlation"]
    },
    {
        "id": "advanced",
        "label": "Advanced Analysis",
        "icon": "fas fa-analytics",
        "default_charts": ["correlation_heatmap", "parallel_coordinates", "regression_results"]
    }
]

# Define which columns should be available for certain visualization types
VISUALIZATION_COLUMNS = {
    "map": ["Latitude", "Longitude", "Sale_Price", "Bldg_Type"],
    "scatter_x": ["Lot_Area", "Year_Built", "Total_Bsmt_SF", "Gr_Liv_Area", "Garage_Area"],
    "scatter_y": ["Sale_Price"],
    "correlation_target": ["Sale_Price"],
    "box_plot_categories": ["Bldg_Type", "Neighborhood", "Overall_Qual", "Central_Air"],
    "time_series_date": ["Year_Built", "Year_Sold", "Mo_Sold"],
    "parallel_coordinates": ["Sale_Price", "Lot_Area", "Year_Built", "Overall_Qual", "Gr_Liv_Area"]
}

# Caching settings
CACHE_CONFIG = {
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": ".dash_cache",
    "CACHE_DEFAULT_TIMEOUT": 3600  # 1 hour cache timeout
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "logs/dashboard.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

# Define building type mappings for clearer labels
BUILDING_TYPE_LABELS = {
    "OneFam": "Single Family Home",
    "TwnhsE": "End Unit Townhouse",
    "Twnhs": "Interior Townhouse",
    "Duplex": "Duplex",
    "TwoFmCon": "Two-Family Conversion"
}

# Define column display names for better readability
COLUMN_DISPLAY_LABELS = {
    "Lot_Frontage": "Lot Frontage (ft)",
    "Lot_Area": "Lot Area (sq ft)",
    "Bldg_Type": "Building Type",
    "House_Style": "House Style",
    "Overall_Cond": "Overall Condition",
    "Year_Built": "Year Built",
    "Exter_Cond": "Exterior Condition",
    "Total_Bsmt_SF": "Total Basement Area (sq ft)",
    "First_Flr_SF": "First Floor Area (sq ft)",
    "Second_Flr_SF": "Second Floor Area (sq ft)",
    "Full_Bath": "Full Bathrooms",
    "Half_Bath": "Half Bathrooms",
    "Bedroom_AbvGr": "Bedrooms (Above Ground)",
    "Kitchen_AbvGr": "Kitchens (Above Ground)",
    "Fireplaces": "Fireplaces",
    "Sale_Price": "Sale Price ($)"
}

# Define available filter components with their types
FILTER_COMPONENTS = {
    "Sale_Price": {
        "type": "range",
        "min": 0,
        "max": 1000000,
        "step": 10000,
        "marks": {0: "$0", 250000: "$250K", 500000: "$500K", 750000: "$750K", 1000000: "$1M"}
    },
    "Bldg_Type": {
        "type": "dropdown",
        "multi": True
    },
    "Neighborhood": {
        "type": "dropdown",
        "multi": True,
        "search_box": True
    },
    "Year_Built": {
        "type": "range",
        "min": 1900,
        "max": 2025,
        "step": 1,
        "marks": {1900: "1900", 1950: "1950", 2000: "2000", 2025: "2025"}
    },
    "Bedrooms": {
        "type": "range",
        "min": 0,
        "max": 10,
        "step": 1,
        "marks": {i: str(i) for i in range(0, 11, 2)}
    },
    "Bathrooms": {
        "type": "range",
        "min": 0,
        "max": 8,
        "step": 0.5,
        "marks": {i: str(i) for i in range(0, 9, 2)}
    },
    "Lot_Area": {
        "type": "range",
        "min": 0,
        "max": 50000,
        "step": 1000,
        "marks": {0: "0", 10000: "10K", 20000: "20K", 30000: "30K", 40000: "40K", 50000: "50K"}
    },
    "Overall_Qual": {
        "type": "dropdown",
        "multi": True
    }
}

# Helper functions to access configuration
def get_theme() -> Dict[str, Any]:
    """Get the current theme configuration"""
    return THEMES.get(ACTIVE_THEME, THEMES["default"])

def get_color_palette(palette_type: str = "categorical") -> List[str]:
    """Get a color palette by type"""
    return VISUALIZATION_COLORS.get(palette_type, VISUALIZATION_COLORS["categorical"])

def get_colorscale(chart_type: str) -> str:
    """Get the appropriate colorscale for a chart type"""
    return VISUALIZATION_COLORS["colorscales"].get(chart_type, "Viridis")

def get_chart_defaults(chart_type: str = None) -> Dict[str, Any]:
    """Get default settings for a chart type"""
    if chart_type and chart_type in CHART_DEFAULTS:
        return CHART_DEFAULTS[chart_type]
    return CHART_DEFAULTS["layout"]

def get_tabs_config() -> List[Dict[str, Any]]:
    """Get the tabs configuration"""
    return TABS_CONFIG

def get_filter_config(column_name: str) -> Dict[str, Any]:
    """Get filter configuration for a specific column"""
    return FILTER_COMPONENTS.get(column_name, {"type": "dropdown", "multi": False})

def get_columns_for_visualization(viz_type: str) -> List[str]:
    """Get recommended columns for a specific visualization type"""
    return VISUALIZATION_COLUMNS.get(viz_type, [])

def get_building_type_label(building_type: str) -> str:
    """
    Get the user-friendly label for a building type code.
    
    Args:
        building_type: The original building type code from the dataset
        
    Returns:
        User-friendly building type label if available, otherwise the original code
    """
    return BUILDING_TYPE_LABELS.get(building_type, building_type)

def get_column_display_label(column_name: str) -> str:
    """
    Get the user-friendly display label for a column name.
    
    Args:
        column_name: The original column name from the dataset
        
    Returns:
        User-friendly column display label if available, otherwise the original name
    """
    return COLUMN_DISPLAY_LABELS.get(column_name, column_name)