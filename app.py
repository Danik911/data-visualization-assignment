"""
Housing Data Dashboard

This is the main entry point for the Housing Data Dashboard application.
"""

from simplified_dash_app import run_dashboard

# Import server for gunicorn
app = run_dashboard(
    data_path="data/Housing Data_cleaned_for_dashboard.csv",
    host="0.0.0.0",  # Use 0.0.0.0 instead of localhost for production
    port=10000,      # Render will override this with PORT environment variable
    debug=False,     # Disable debug mode in production
    return_app=True  # Return the app object instead of running it
)

# Get the Flask server for gunicorn
server = app.server

# This is only used when running locally
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=10000, debug=False) 