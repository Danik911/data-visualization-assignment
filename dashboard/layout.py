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
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1("Housing Data Dashboard", className="dashboard-title"),
                        html.P(
                            "Interactive visualization of housing data with filters and analysis.",
                            className="lead"
                        ),
                        html.Div([
                            html.Span("Last Updated: May 7, 2025", className="text-muted me-3"),
                            html.Span(html.I(className="fas fa-info-circle me-1"), id="info-icon"),
                            dbc.Tooltip(
                                [
                                    html.P("This dashboard visualizes housing market data for property analysis."),
                                    html.P("Use the filters on the left to refine the data view."),
                                    html.P("Data source: Housing Data CSV")
                                ],
                                target="info-icon",
                                placement="bottom"
                            )
                        ], className="d-flex align-items-center small")
                    ], className="header-content fade-in")
                ], width=9),
                dbc.Col([
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button(html.I(className="fas fa-download me-1"), id="download-button", color="outline-secondary", size="sm", className="me-2"),
                            dbc.Button(html.I(className="fas fa-chart-line me-1"), id="view-trends-button", color="outline-primary", size="sm")
                        ]),
                        dbc.Tooltip("Download Data", target="download-button"),
                        dbc.Tooltip("View Market Trends", target="view-trends-button")
                    ], className="d-flex justify-content-end")
                ], width=3)
            ]),
            html.Hr()
        ],
        fluid=True,
        className="py-4"
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
        from dashboard.config import get_building_type_label
        building_type_filter = html.Div(
            [
                dbc.Label("Building Type", className="form-label"),
                html.Div([
                    html.I(className="fas fa-building text-secondary me-2"),
                    dcc.Dropdown(
                        id="building-type-filter",
                        options=[{"label": get_building_type_label(bt), "value": bt} for bt in options["Bldg_Type"]],
                        multi=True,
                        placeholder="Select building types...",
                        style={"width": "100%"},
                    )
                ], className="d-flex align-items-center", style={"overflow": "visible", "position": "relative", "zIndex": 1000}),
                html.Div(style={"height": "45px"})  # Increased from 20px to 45px
            ],
            style={"marginBottom": "40px", "position": "relative", "zIndex": 999}
        )
    
    # Price Range filter if available
    price_filter = html.Div([])
    if "Sale_Price" in options:
        min_price = options["Sale_Price"]["min"]
        max_price = options["Sale_Price"]["max"]
        
        price_filter = html.Div(
            [
                dbc.Label("Price Range", className="form-label"),
                html.Div([
                    html.I(className="fas fa-dollar-sign text-secondary me-2"),
                    html.Div([
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
                        )
                    ], style={"width": "100%"})
                ], className="d-flex align-items-center"),
                html.Div([
                    html.Span(f"${min_price:,.0f}", id="price-range-min", className="small text-muted"),
                    html.Span(className="ms-auto", children=[
                        f"${max_price:,.0f}", 
                        html.Span(id="price-range-max", className="small text-muted")
                    ])
                ], className="d-flex justify-content-between px-2 mt-1"),
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
                dbc.Label("Lot Area", className="form-label"),
                html.Div([
                    html.I(className="fas fa-ruler-combined text-secondary me-2"),
                    html.Div([
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
                        )
                    ], style={"width": "100%"})
                ], className="d-flex align-items-center"),
                html.Div([
                    html.Span(f"{min_area:,.0f} sq.ft", id="area-range-min", className="small text-muted"),
                    html.Span(className="ms-auto", children=[
                        f"{max_area:,.0f} sq.ft", 
                        html.Span(id="area-range-max", className="small text-muted")
                    ])
                ], className="d-flex justify-content-between px-2 mt-1"),
                html.Div(style={"height": "20px"})  # Add some spacing
            ]
        )
    
    # Reset button with icon
    reset_button = dbc.Button(
        [
            html.I(className="fas fa-undo me-2"),
            "Reset Filters"
        ],
        id="reset-filters-button",
        color="secondary",
        className="mt-3 w-100"
    )
    
    # Filter count badge
    filter_badge = html.Div([
        dbc.Badge("0 active filters", id="filter-count-badge", color="light", className="text-secondary mb-3")
    ], className="d-flex justify-content-end")
    
    # Create filters card
    filters_card = dbc.Card(
        dbc.CardBody(
            [
                html.Div([
                    html.H4("Filters", className="card-title d-inline me-2"),
                    html.I(className="fas fa-filter text-secondary")
                ], className="d-flex align-items-center mb-2"),
                filter_badge,
                html.Hr(),
                html.Div(building_type_filter, style={"marginBottom": "20px"}),
                html.Div(price_filter, style={"marginBottom": "20px"}),
                html.Div(area_filter, style={"marginBottom": "20px"}),
                reset_button
            ]
        ),
        className="mb-4 shadow-sm"
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
    # Import Google Maps component
    from dashboard.google_maps_component import create_google_map
    
    # Overview Tab
    overview_tab = dbc.Container(
        [
            dbc.Row(
                [
                    # Google Maps Price Map - Full Width
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Housing Prices by Location"),
                                create_google_map(id="google-price-map"),
                                html.Div(id="google-price-map-fallback")  # Fallback element for static map
                            ]),
                            className="mb-4"
                        ),
                        width=12  # Changed from 8 to 12 for full width
                    )
                ]
            ),
            # Add the four overview charts here
            dbc.Row(
                [
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
                        width=6
                    ),
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
                    )
                ],
                className="mb-4"
            ),
            dbc.Row(
                [
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
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Neighborhood Distribution"),
                                dcc.Loading(
                                    dcc.Graph(id="neighborhood-pie-chart")
                                )
                            ]),
                            className="mb-4"
                        ),
                        width=6
                    )
                ],
                className="mb-4"
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
                        width=12  # Changed from 6 to 12 for full width
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
            ),
            # Add the two market charts here
            dbc.Row(
                [
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
                ],
                className="mb-4"
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
                    
                    dbc.Row( # Row to hold filters and main content
                        [
                            dbc.Col( # Column for Filters
                                create_filters(filter_options),
                                width=3, # Filters take 3/12 of the width
                                className="filters-column" # Optional: for specific styling
                            ),
                            dbc.Col( # Column for Main Content (Tabs)
                                create_tabs(tab_content),
                                width=9 # Main content takes 9/12 of the width
                            )
                        ],
                        className="mb-4" 
                    )
                ],
                fluid=True
            ),
            
            # Footer
            create_footer(),
            
            # Store for sharing data between callbacks
            dcc.Store(id="filtered-data-store", data=data_provider.get_data().to_json(date_format='iso', orient='split'))
        ]
    )
    
    return layout
