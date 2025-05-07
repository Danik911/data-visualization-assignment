"""
Layout module for the dashboard.
This module defines the UI components and overall structure of the dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_header() -> dbc.Container:
    """
    Create the header section of the dashboard with title and description.
    
    Returns:
        A Bootstrap container with the header content
    """
    header = dbc.Container(
        [
            html.H1("Housing Data Dashboard", className="dashboard-title"),
            html.P(
                "Interactive visualization of housing data with filters and analysis.",
                className="lead"
            ),
            html.Hr()
        ],
        fluid=True,
        className="py-3"
    )
    
    return header


def create_filters(options: dict = None) -> dbc.Card:
    """
    Create the filters sidebar for the dashboard.
    
    Args:
        options: Dictionary containing filter options from data_provider
        
    Returns:
        A Bootstrap card containing filter components
    """
    if not options:
        options = {}
        
    # Building Type filter if available
    building_type_filter = html.Div([])
    if "Bldg_Type" in options:
        building_type_filter = html.Div(
            [
                dbc.Label("Building Type"),
                dcc.Dropdown(
                    id="building-type-filter",
                    options=[{"label": bt, "value": bt} for bt in options["Bldg_Type"]],
                    multi=True,
                    placeholder="Select building types..."
                ),
                html.Div(style={"height": "20px"})  # Add some spacing
            ]
        )
    
    # Price Range filter if available
    price_filter = html.Div([])
    if "Sale_Price" in options:
        min_price = options["Sale_Price"]["min"]
        max_price = options["Sale_Price"]["max"]
        
        price_filter = html.Div(
            [
                dbc.Label("Price Range"),
                dcc.RangeSlider(
                    id="price-range-filter",
                    min=min_price,
                    max=max_price,
                    step=(max_price - min_price) / 100,
                    marks={
                        min_price: f"${min_price:,.0f}",
                        max_price: f"${max_price:,.0f}"
                    },
                    value=[min_price, max_price],
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Div(style={"height": "20px"})  # Add some spacing
            ]
        )
    
    # Lot Area filter if available
    area_filter = html.Div([])
    if "Lot_Area" in options:
        min_area = options["Lot_Area"]["min"]
        max_area = options["Lot_Area"]["max"]
        
        area_filter = html.Div(
            [
                dbc.Label("Lot Area"),
                dcc.RangeSlider(
                    id="area-range-filter",
                    min=min_area,
                    max=max_area,
                    step=(max_area - min_area) / 100,
                    marks={
                        min_area: f"{min_area:,.0f}",
                        max_area: f"{max_area:,.0f}"
                    },
                    value=[min_area, max_area],
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Div(style={"height": "20px"})  # Add some spacing
            ]
        )
    
    # Reset button
    reset_button = dbc.Button(
        "Reset Filters",
        id="reset-filters-button",
        color="secondary",
        className="mt-3 w-100"
    )
    
    # Create filters card
    filters_card = dbc.Card(
        dbc.CardBody(
            [
                html.H4("Filters", className="card-title"),
                html.Hr(),
                building_type_filter,
                price_filter,
                area_filter,
                reset_button
            ]
        ),
        className="mb-4"
    )
    
    return filters_card


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
    
    summary_cards = []
    
    # Create a card for each summary statistic
    for key, data in summary_data.items():
        card = dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H2(data["value"], className="card-title text-center"),
                        html.P(data["description"], className="card-text text-center")
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
    # Overview Tab
    overview_tab = dbc.Container(
        [
            dbc.Row(
                [
                    # Price Map
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Housing Prices by Location"),
                                dcc.Loading(
                                    dcc.Graph(id="price-map")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=8
                    ),
                    # Price Distribution
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Price Distribution"),
                                dcc.Loading(
                                    dcc.Graph(id="price-distribution")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=4
                    )
                ]
            ),
            dbc.Row(
                [
                    # Feature Importance
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Feature Importance for Price"),
                                dcc.Loading(
                                    dcc.Graph(id="feature-importance")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    ),
                    # Building Type Distribution
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Distribution by Building Type"),
                                dcc.Loading(
                                    dcc.Graph(id="building-type-distribution")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    )
                ]
            )
        ],
        fluid=True
    )
    
    # Property Analysis Tab
    property_analysis_tab = dbc.Container(
        [
            dbc.Row(
                [
                    # Scatter Plot with Property Attributes
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Property Attribute Analysis"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("X-Axis"),
                                        dcc.Dropdown(
                                            id="scatter-x-axis",
                                            placeholder="Select x-axis attribute..."
                                        )
                                    ], width=5),
                                    dbc.Col([
                                        dbc.Label("Y-Axis"),
                                        dcc.Dropdown(
                                            id="scatter-y-axis",
                                            placeholder="Select y-axis attribute..."
                                        )
                                    ], width=5),
                                    dbc.Col([
                                        dbc.Label("Color By"),
                                        dcc.Dropdown(
                                            id="scatter-color",
                                            placeholder="Select color attribute..."
                                        )
                                    ], width=2)
                                ], className="mb-3"),
                                dcc.Loading(
                                    dcc.Graph(id="property-scatter-plot")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=8
                    ),
                    # Box Plot for Distribution Analysis
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Distribution Analysis"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Numeric Variable"),
                                        dcc.Dropdown(
                                            id="box-plot-numeric",
                                            placeholder="Select numeric variable..."
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Group By"),
                                        dcc.Dropdown(
                                            id="box-plot-category",
                                            placeholder="Select category..."
                                        )
                                    ], width=6)
                                ], className="mb-3"),
                                dcc.Loading(
                                    dcc.Graph(id="property-box-plot")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=4
                    )
                ]
            ),
            dbc.Row(
                [
                    # Correlation Heatmap
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Correlation Analysis"),
                                dcc.Loading(
                                    dcc.Graph(id="correlation-heatmap")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    ),
                    # Parallel Coordinates Plot
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Multi-Dimensional Analysis"),
                                dcc.Loading(
                                    dcc.Graph(id="parallel-coordinates")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    )
                ]
            )
        ],
        fluid=True
    )
    
    # Market Trends Tab
    market_trends_tab = dbc.Container(
        [
            dbc.Row(
                [
                    # Price by Year Chart
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Price Trends by Year Built"),
                                dcc.Loading(
                                    dcc.Graph(id="price-by-year")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=12
                    )
                ]
            ),
            dbc.Row(
                [
                    # Price vs Property Age
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Price vs Property Age"),
                                dcc.Loading(
                                    dcc.Graph(id="age-price-correlation")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    ),
                    # Decade & Building Type Heatmap
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Price by Decade & Building Type"),
                                dcc.Loading(
                                    dcc.Graph(id="decade-bldg-heatmap")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    )
                ]
            ),
            dbc.Row(
                [
                    # Area-Price Ratio Trend
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Price per Square Foot Analysis"),
                                dcc.Loading(
                                    dcc.Graph(id="price-per-sqft-analysis")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    ),
                    # Building Type Comparison
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Building Type Comparison"),
                                dcc.Loading(
                                    dcc.Graph(id="building-type-comparison")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    )
                ]
            )
        ],
        fluid=True
    )
    
    # Property Comparison Tab - New tab for enhanced comparison features
    property_comparison_tab = dbc.Container(
        [
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
                                value="Bldg_Type",
                                clearable=False
                            )
                        ],
                        width=4
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Generate Comparisons",
                            id="generate-comparison-button",
                            color="primary",
                            className="mt-4"
                        ),
                        width=4
                    )
                ],
                className="mb-4"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Price Distribution by Category"),
                                dcc.Loading(
                                    dcc.Graph(id="comparison-price-box")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Average Price by Category"),
                                dcc.Loading(
                                    dcc.Graph(id="comparison-price-bar")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Price vs Area by Category"),
                                dcc.Loading(
                                    dcc.Graph(id="comparison-scatter")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Multi-Metric Comparison"),
                                dcc.Loading(
                                    dcc.Graph(id="comparison-radar")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    )
                ]
            )
        ],
        fluid=True
    )
    
    # Data Table Tab
    data_table_tab = dbc.Container(
        [
            dbc.Card(
                dbc.CardBody([
                    html.H4("Property Data Table"),
                    dcc.Loading(
                        id="data-table-loading",
                        children=[
                            html.Div(id="data-table-container")
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
        "property_comparison": property_comparison_tab,  # Add the new tab
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
                    dbc.Tab(tab_content["overview"], label="Overview", tab_id="tab-overview"),
                    dbc.Tab(tab_content["property_analysis"], label="Property Analysis", tab_id="tab-property"),
                    dbc.Tab(tab_content["market_trends"], label="Market Trends", tab_id="tab-market"),
                    dbc.Tab(tab_content["property_comparison"], label="Property Comparison", tab_id="tab-comparison"),
                    dbc.Tab(tab_content["data_table"], label="Data Table", tab_id="tab-data")
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
                "Housing Data Dashboard - Created with Dash and Plotly",
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
    
    # Create tab content
    tab_content = create_tab_content()
    
    # Create main layout
    layout = html.Div(
        [
            # Header
            create_header(),
            
            # Main Content
            dbc.Container(
                [
                    # Summary Cards
                    create_summary_cards(summary_data),
                    
                    dbc.Row(
                        [
                            # Filters Sidebar
                            dbc.Col(
                                create_filters(filter_options),
                                width=3
                            ),
                            
                            # Main Dashboard Content
                            dbc.Col(
                                create_tabs(tab_content),
                                width=9
                            )
                        ]
                    )
                ],
                fluid=True
            ),
            
            # Footer
            create_footer(),
            
            # Store for sharing data between callbacks
            dcc.Store(id="filtered-data-store")
        ]
    )
    
    return layout