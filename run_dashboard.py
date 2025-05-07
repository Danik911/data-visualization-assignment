"""
Run script for Housing Data Dashboard.
This script launches the dashboard with the housing data.
"""

from dashboard.dash_app import run_dashboard

if __name__ == "__main__":
    # Path to the housing data CSV file
    data_path = "data/Housing Data_cleaned_for_dashboard.csv"
    
    # Run the dashboard
    print("Starting Housing Data Dashboard...")
    print(f"Using data from: {data_path}")
    print("Dashboard will be available at: http://localhost:8050")
    
    run_dashboard(
        data_path=data_path,
        host="localhost", 
        port=8050,
        debug=True
    )