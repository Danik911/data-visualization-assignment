"""
Layout module for the dashboard.
This module defines the UI components and overall structure of the dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State, no_update
import plotly.graph_objects as go # Import Plotly graph objects

# Import necessary functions from other modules
from dashboard.data_provider import load_and_preprocess_data, get_feature_names, get_numeric_features, get_categorical_features
from dashboard.google_maps_component import create_google_map
from dashboard.pandas_helper import load_data

# Load data to get initial values for filters if needed
# df = load_data() # Potentially load df here if needed for initial filter values
# min_price, max_price = df['Price'].min(), df['Price'].max()
# min_area, max_area = df['Area'].min(), df['Area'].max()
# building_types = df['Building_Type'].unique()

# Placeholder for empty figure to avoid errors on initial load
empty_fig = go.Figure(data=[], layout={"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No data selected", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]})


# Placeholder or function to create a default empty figure with a message
def create_empty_figure(message="Please select data or wait for loading."):
    """Creates an empty Plotly figure with a centered message."""
    return {
        "data": [],
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16},
                "align": "center"
            }],
            "paper_bgcolor": 'rgba(0,0,0,0)',
            "plot_bgcolor": 'rgba(0,0,0,0)',
            "height": 300 # Adjust height as needed
        }
    }

# Create default empty figures for each chart ID
default_figures = {
    "google-price-map": None, # Map handled differently
    "price-distribution": create_empty_figure("Price distribution requires data."),
    "feature-importance": create_empty_figure("Feature importance requires data."),
    "building-type-distribution": create_empty_figure("Building type distribution requires data."),
    "neighborhood-pie-chart": create_empty_figure("Neighborhood distribution requires data."),
    "price-per-sqft-analysis": create_empty_figure("Price/SqFt analysis requires data."),
    "building-type-comparison": create_empty_figure("Building type comparison requires data."),
    "age-price-correlation": create_empty_figure("Age-Price correlation requires data."),
    "price-vs-area-scatter": create_empty_figure("Price vs Area requires data."),
    "property-comparison-table": None, # Table handled differently
    "year-trend-analysis": create_empty_figure("Year trend analysis requires data."),
    "monthly-trend-analysis": create_empty_figure("Monthly trend analysis requires data."),
    "data-preview-table": None, # Table handled differently
}


def create_header() -> dbc.NavbarSimple:
    """
    Create the header section of the dashboard with title and description.
    
    Returns:
        A Bootstrap navbar with the header content
    """
    header = dbc.NavbarSimple(
        children=[
            dbc.NavbarBrand("Amsterdam Housing Dashboard", href="#"),
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="#"),
                    dbc.NavLink("About", href="#"),
                    dbc.NavLink("Contact", href="#")
                ],
                className="ms-auto"
            ),
            dbc.Nav(
                [
                    dbc.NavLink(html.I(className="fas fa-download me-1"), href="#"),
                    dbc.NavLink(html.I(className="fas fa-chart-line me-1"), href="#")
                ],
                className="ms-auto"
            )
        ],
        color="primary",
        dark=True,
        className="mb-4", # Add margin bottom
    )
    
    return header


def create_filters(df) -> dbc.Card:
    """
    Create the filters sidebar for the dashboard.
    
    Args:
        df: DataFrame with housing data (used for filter options)
        
    Returns:
        A Bootstrap card containing filter components
    """
    min_area, max_area = int(df['Area'].min()), int(df['Area'].max())
    building_types = df['Building_Type'].unique()

    return dbc.Card(
        dbc.CardBody([
            html.H4("Filters", className="card-title"),
            dbc.Row([
                dbc.Col([
                    html.Label("Building Type:"),
                    dcc.Dropdown(
                        id='building-type-filter',
                        options=[{'label': b_type, 'value': b_type} for b_type in sorted(building_types)],
                        value=None, # Default to no selection
                        multi=True,
                        placeholder="Select building types...",
                    ),
                ], width=12, md=4),
                dbc.Col([
                    html.Label("Price Range (€):"),
                    dcc.RangeSlider(
                        id='price-range-filter',
                        min=min_price,
                        max=max_price,
                        value=[min_price, max_price],
                        marks={i: f'€{i // 1000}k' for i in range(min_price, max_price + 1, (max_price - min_price) // 5)},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], width=12, md=4),
                dbc.Col([
                    html.Label("Area Range (m²):"),
                    dcc.RangeSlider(
                        id='area-range-filter',
                        min=min_area,
                        max=max_area,
                        value=[min_area, max_area],
                        marks={i: f'{i} m²' for i in range(min_area, max_area + 1, (max_area - min_area) // 5)},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], width=12, md=4),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        "Reset Filters", 
                        id="reset-filters-button", 
                        color="secondary", 
                        className="mt-3",
                        n_clicks=0
                    ),
                    width="auto",
                    className="d-flex align-items-end" # Align button nicely
                ),
                 dbc.Col(
                     html.Div(id='filter-count-display', className='mt-3 align-self-end') # Placeholder for filter count
                 )
            ], className="mt-2", justify="start")
        ]),
        className="mb-4" # Add margin bottom
    )


def create_summary_cards(summary_data: dict = None) -> dbc.Row:
    """
    Create summary statistic cards to display at the top of the dashboard.
    
    Args:
        summary_data: Dictionary containing summary statistics
        
    Returns:
        A Bootstrap row containing summary cards
    """
    if not summary_data:
        summary_data = {
            "total_properties": {"value": "Loading...", "description": "Total Properties"},
            "avg_price": {"value": "Loading...", "description": "Average Price"},
            "median_price": {"value": "Loading...", "description": "Median Price"},
            "common_type": {"value": "Loading...", "description": "Most Common Type"}
        }
    
    # Define icons for each card type
    card_icons = {
        "total_properties": "fas fa-home",
        "avg_price": "fas fa-dollar-sign",
        "median_price": "fas fa-chart-line",
        "common_type": "fas fa-building",
        "avg_area": "fas fa-ruler-combined",
        "price_range": "fas fa-exchange-alt"
    }
    
    summary_cards = []
    
    # Create a card for each summary statistic
    for key, data in summary_data.items():
        icon = card_icons.get(key, "fas fa-chart-bar")  # Default icon if not found
        
        card = dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([
                            html.Div([
                                html.I(className=f"{icon} fa-2x text-primary opacity-75")
                            ], className="summary-icon me-3"),
                            html.Div([
                                html.H2(data["value"], className="card-title mb-0"),
                                html.P(data["description"], className="card-text text-muted mb-0 small")
                            ])
                        ], className="d-flex align-items-center fade-in")
                    ]
                ),
                className="mb-4 summary-card"
            ),
            width=3
        )
        summary_cards.append(card)
    
    return dbc.Row(summary_cards, className="mb-4", id="summary-cards-row")


def create_tab_content():
    """
    Create the content for each tab of the dashboard.
    
    Returns:
        Dictionary containing the content for each tab
    """
    from dashboard.google_maps_component import create_google_map

    # Overview Tab Content
    overview_tab = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Housing Prices by Location"),
                                create_google_map(id="google-price-map"),
                                html.Div(id="google-price-map-fallback")
                            ]),
                            className="mb-4"
                        ),
                        width=12
                    )
                ]
            ),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("Price Distribution"), dcc.Loading(dcc.Graph(id="price-distribution"))])), width=6, className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("Building Type Distribution"), dcc.Loading(dcc.Graph(id="building-type-distribution"))])), width=6, className="mb-4"),
            ]),
            dbc.Row([
                 dbc.Col(dbc.Card(dbc.CardBody([html.H4("Feature Importance"), dcc.Loading(dcc.Graph(id="feature-importance"))])), width=6, className="mb-4"),
                 dbc.Col(dbc.Card(dbc.CardBody([html.H4("Distribution by Category"), dcc.Loading(dcc.Graph(id="neighborhood-pie-chart"))])), width=6, className="mb-4"), # Placeholder for pie chart
            ])
        ],
        fluid=True
    )

    # Property Analysis Tab Content
    property_analysis_tab = dbc.Container(
        [
            dbc.Row( # First row: Correlation and Parallel Coords
                [
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4("Correlation Analysis"), dcc.Loading(dcc.Graph(id="correlation-heatmap"))])), width=6, className="mb-4"),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4("Parallel Coordinates"), dcc.Loading(dcc.Graph(id="parallel-coordinates"))])), width=6, className="mb-4"),
                ]
            ),
            dbc.Row( # Second row: Scatter and Box plots with controls
                [
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Scatter Plot"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Dropdown(id="scatter-x-axis", placeholder="Select X-axis"), width=4),
                                    dbc.Col(dcc.Dropdown(id="scatter-y-axis", placeholder="Select Y-axis"), width=4),
                                    dbc.Col(dcc.Dropdown(id="scatter-color", placeholder="Select Color"), width=4)
                                ]),
                                dcc.Loading(dcc.Graph(id="property-scatter-plot"))
                            ])
                        ])
                    ], width=6, className="mb-4"),
                    dbc.Col([
                        dbc.Card([
                             dbc.CardHeader("Box Plot"),
                             dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dcc.Dropdown(id="box-plot-numeric", placeholder="Select Numeric Var"), width=6),
                                    dbc.Col(dcc.Dropdown(id="box-plot-category", placeholder="Select Category Var"), width=6)
                                ]),
                                dcc.Loading(dcc.Graph(id="property-box-plot"))
                             ])
                        ])
                    ], width=6, className="mb-4"),
                ]
            )
        ],
        fluid=True
    )

    # Market Trends Tab Content
    market_trends_tab = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4("Price Trends by Year Built"), dcc.Loading(dcc.Graph(id="price-by-year"))])), width=12, className="mb-4")
                ]
            ),
            dbc.Row(
                [
                     dbc.Col(dbc.Card(dbc.CardBody([html.H4("Age vs Price Correlation"), dcc.Loading(dcc.Graph(id="age-price-correlation"))])), width=6, className="mb-4"),
                     dbc.Col(dbc.Card(dbc.CardBody([html.H4("Decade vs Building Type Heatmap"), dcc.Loading(dcc.Graph(id="decade-bldg-heatmap"))])), width=6, className="mb-4"),
                ]
            ),
            # Add Price per SqFt and Building Type comparison charts
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4("Price per SqFt Analysis"), dcc.Loading(dcc.Graph(id="price-per-sqft-analysis"))])), width=6, className="mb-4"),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H4("Building Type Average Price"), dcc.Loading(dcc.Graph(id="building-type-comparison"))])), width=6, className="mb-4"),
                ]
            )
        ],
        fluid=True
    )

    # Property Comparison Tab Content
    property_comparison_tab = dbc.Container(
        [
            # Controls Row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Compare By"),
                            dcc.Dropdown(
                                id="comparison-column",
                                options=[
                                    {"label": "Building Type", "value": "Bldg_Type"},
                                    {"label": "House Style", "value": "House_Style"},
                                    {"label": "Neighborhood", "value": "Neighborhood"},
                                    {"label": "Sale Condition", "value": "Sale_Condition"}
                                ],
                                value="Bldg_Type", # Default value
                                clearable=False
                            ),
                            dbc.Button(
                                "Generate Comparisons",
                                id="generate-comparison-button",
                                color="primary",
                                className="mt-3 w-100" # Make button full width of column
                            )
                        ],
                        width=3, # Control column takes 3/12 width
                        className="mb-4" # Add margin below controls
                    ),
                    dbc.Col( # Placeholder or initial message for charts area
                        html.Div("Select a category and click 'Generate Comparisons' to view charts.",
                                 style={'textAlign': 'center', 'marginTop': '20px', 'color': '#888'}),
                        width=9,
                        id="comparison-charts-area" # We might still need this ID if we want to update the placeholder dynamically
                    )
                ],
                className="mb-3 align-items-start" # Align items at the start
            ),
            # Charts Row (Initially maybe hidden or empty, updated by callback)
             dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("Price Distribution by Category"), dcc.Loading(dcc.Graph(id="comparison-price-box"))])), width=6, className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("Average Price by Category"), dcc.Loading(dcc.Graph(id="comparison-price-bar"))])), width=6, className="mb-4"),
            ]),
             dbc.Row([
                 dbc.Col(dbc.Card(dbc.CardBody([html.H4("Price vs Area by Category"), dcc.Loading(dcc.Graph(id="comparison-scatter"))])), width=6, className="mb-4"),
                 dbc.Col(dbc.Card(dbc.CardBody([html.H4("Multi-Metric Comparison"), dcc.Loading(dcc.Graph(id="comparison-radar"))])), width=6, className="mb-4"),
            ]),
        ],
        fluid=True
    )

    # Data Table Tab Content (Remains simple)
    data_table_tab = dbc.Container(
        [
            dbc.Card(
                dbc.CardBody([
                    html.H4("Property Data Table"),
                    dcc.Loading(
                        id="data-table-loading",
                        children=[
                            html.Div(id="data-table-container") # Container for the DataTable
                        ],
                        type="circle"
                    )
                ]),
                className="mb-4"
            )
        ],
        fluid=True
    )

    return {
        "overview": overview_tab,
        "property_analysis": property_analysis_tab,
        "market_trends": market_trends_tab,
        "property_comparison": property_comparison_tab,
        "data_table": data_table_tab
    }


def create_tabs(tab_content: dict) -> dbc.Container:
    """
    Create the tabs for the dashboard.
    
    Args:
        tab_content: Dictionary containing the content for each tab
        
    Returns:
        A Bootstrap container with the tabs and their content
    """
    tabs = dbc.Container(
        [
            dbc.Tabs(
                [
                    dbc.Tab(html.Div(tab_content["overview"], id="tab-content-overview", className="tab-content"), label="Overview", tab_id="tab-overview"),
                    dbc.Tab(html.Div(tab_content["property_analysis"], id="tab-content-property", className="tab-content"), label="Property Analysis", tab_id="tab-property"),
                    dbc.Tab(html.Div(tab_content["market_trends"], id="tab-content-market", className="tab-content"), label="Market Trends", tab_id="tab-market"),
                    dbc.Tab(html.Div(tab_content["property_comparison"], id="tab-content-comparison", className="tab-content"), label="Property Comparison", tab_id="tab-comparison"),
                    dbc.Tab(html.Div(tab_content["data_table"], id="tab-content-data", className="tab-content"), label="Data Table", tab_id="tab-data")
                ],
                id="dashboard-tabs",
                active_tab="tab-overview"
            )
        ],
        fluid=True
    )
    
    return tabs


def create_footer() -> dbc.Container:
    """
    Create the footer section of the dashboard.
    
    Returns:
        A Bootstrap container with the footer content
    """
    footer = dbc.Container(
        [
            html.Hr(),
            html.P(
                "Amsterdam Housing Data Dashboard © 2024",
                className="text-center text-muted"
            )
        ],
        fluid=True,
        className="py-3"
    )
    
    return footer


def create_layout(data_provider=None):
    """
    Create the overall layout of the dashboard.
    
    Args:
        data_provider: Data provider instance to get filter options and summary data
        
    Returns:
        A Bootstrap container with the complete dashboard layout
    """
    # Get filter options and summary data from data_provider if available
    filter_options = {}
    summary_data = {}
    
    if data_provider:
        filter_options = data_provider.get_column_options()
        df = data_provider.get_data()
        
        # Import visualization module here to avoid circular import
        from dashboard.visualizations import generate_summary_cards
        summary_data = generate_summary_cards(df)
    
    # Create tab content (now includes all charts)
    tab_content = create_tab_content()
    
    # Add a dcc.Store to track the active tab
    active_tab_store = dcc.Store(id="active-tab-store", data="tab-overview")

    layout = dbc.Container(
        [
            create_header(),
            dbc.Row(
                [
                    # Filters Sidebar (3 cols)
                    dbc.Col(
                        create_filters(df),
                        width=12, lg=3, 
                        className="sidebar"
                    ),
                    
                    # Main Content Area (9 cols)
                    dbc.Col(
                        [
                            create_summary_cards(summary_data),
                            create_tabs(tab_content) # Tabs now contain full content
                        ],
                        width=12, lg=9, 
                        className="main-content"
                    )
                ],
                className="mb-4"
            ),
            create_footer(),
            
            # Stores
            dcc.Store(id="filtered-data-store"),
            active_tab_store,
        ],
        fluid=True,
        className="dashboard-container"
    )
    
    return layout


@callback(
    Output('tabs-content', 'children'),
    Input('dashboard-tabs', 'active_tab')
)
def render_tab_content(active_tab):
    """Renders the content based on the selected tab."""
    if active_tab == 'tab-overview':
        return create_overview_tab()
    elif active_tab == 'tab-property':
        return create_property_analysis_tab()
    elif active_tab == 'tab-market':
        return create_market_trends_tab()
    elif active_tab == 'tab-comparison':
        return create_property_comparison_tab()
    elif active_tab == 'tab-data':
        return create_data_view_tab()
    return html.P("This tab content hasn't been implemented yet.") # Fallback
