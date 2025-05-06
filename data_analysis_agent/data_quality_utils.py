import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_valid_modes():
    """Return the set of valid mode values."""
    return {'Car', 'Bus', 'Cycle', 'Walk', 'Train', 'Tram', 'Subway'}

def get_mode_correction(invalid_mode, valid_modes=None):
    """Suggest a correction for an invalid mode based on first letter."""
    if valid_modes is None:
        valid_modes = get_valid_modes()
    if not isinstance(invalid_mode, str) or not invalid_mode:
        return "Unknown"
    first_letter = invalid_mode[0].upper()
    for mode in valid_modes:
        if mode.startswith(first_letter):
            return mode
    return "Unknown"

def convert_column_type(series: pd.Series, to_type: str) -> pd.Series:
    """Convert a pandas Series to the specified type."""
    try:
        if to_type == 'int64':
            return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)
        elif to_type == 'float64':
            return pd.to_numeric(series, errors='coerce')
        # Add more conversions as needed
        return series
    except Exception as e:
        print(f"Error converting series to {to_type}: {e}")
        return series

def plot_before_after_hist(original, cleaned, col, plots_dir):
    """Plot before/after histogram for a numeric column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(original.dropna(), color='red', alpha=0.5, label='Before', kde=True)
    sns.histplot(cleaned.dropna(), color='blue', alpha=0.5, label='After', kde=True)
    plt.title(f'Before/After: {col} Distribution')
    plt.legend()
    plot_path = os.path.join(plots_dir, f'{col}_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_mode_comparison(original_mode, cleaned_mode, plots_dir):
    """Plot before/after countplot for the Mode column."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.countplot(x=original_mode, ax=axes[0])
    axes[0].set_title('Mode Values: Before')
    axes[0].tick_params(axis='x', rotation=45)
    sns.countplot(x=cleaned_mode, ax=axes[1])
    axes[1].set_title('Mode Values: After')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'mode_comparison.png')
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path
