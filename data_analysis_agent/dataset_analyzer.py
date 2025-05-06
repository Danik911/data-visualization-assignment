"""
Dataset Analyzer module for automatically detecting dataset properties.
This module provides functions to analyze any dataset and determine its characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from config import get_config

def analyze_dataset(df: pd.DataFrame, 
                   save_report: bool = True,
                   report_path: str = "reports/dataset_analysis.json",
                   generate_plots: bool = True,
                   plots_dir: str = "plots/dataset_analysis") -> Dict[str, Any]:
    """
    Analyze a dataset to determine its properties and structure.
    
    Args:
        df: The DataFrame to analyze
        save_report: Whether to save the analysis report to a file
        report_path: Path to save the report
        generate_plots: Whether to generate analysis plots
        plots_dir: Directory to save plots in
        
    Returns:
        Dictionary with dataset properties and analysis results
    """
    print(f"[ANALYZER] Starting dataset analysis of DataFrame with shape {df.shape}")
    
    # Get configuration
    config = get_config()
    
    # Initialize analysis results
    analysis_results = {
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "column_types": {},
        "missing_values": {},
        "numeric_stats": {},
        "categorical_stats": {},
        "correlations": {},
        "potential_targets": {},
        "dataset_domain": {},
        "summary": []
    }
    
    # 1. Analyze column types and missing values
    column_types = analyze_column_types(df, config)
    analysis_results["column_types"] = column_types
    
    # 2. Analyze missing values
    missing_values = analyze_missing_values(df)
    analysis_results["missing_values"] = missing_values
    
    # 3. Analyze numeric columns
    numeric_columns = [col for col, info in column_types.items() if info["type"] == "numeric"]
    analysis_results["numeric_stats"] = analyze_numeric_columns(df, numeric_columns)
    
    # 4. Analyze categorical columns
    categorical_columns = [col for col, info in column_types.items() 
                          if info["type"] in ["categorical", "boolean", "datetime"]]
    analysis_results["categorical_stats"] = analyze_categorical_columns(df, categorical_columns)
    
    # 5. Analyze correlations between numeric columns
    if len(numeric_columns) > 1:
        analysis_results["correlations"] = analyze_correlations(df[numeric_columns])
    
    # 6. Detect potential target variables
    analysis_results["potential_targets"] = detect_potential_targets(
        df, numeric_columns, column_types, analysis_results["correlations"], config)
    
    # 7. Detect dataset domain
    analysis_results["dataset_domain"] = detect_dataset_domain(df, config)
    
    # 8. Generate overview summary
    analysis_results["summary"] = generate_summary(analysis_results)
    
    # 9. Generate plots if requested
    plot_paths = []
    if generate_plots:
        # Create plots directory if it doesn't exist
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate distribution plots for numeric columns
        for col in numeric_columns[:5]:  # Limit to 5 columns for brevity
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(plots_dir, f"{col.lower().replace(' ', '_')}_distribution.png")
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(plot_path)
            
        # Generate correlation heatmap
        if len(numeric_columns) > 1:
            plt.figure(figsize=(12, 10))
            corr = df[numeric_columns].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, 
                      square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            
            corr_path = os.path.join(plots_dir, "correlation_heatmap.png")
            plt.savefig(corr_path)
            plt.close()
            plot_paths.append(corr_path)
            
        # Generate bar plot of missing values
        plt.figure(figsize=(12, 8))
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if not missing.empty:
            sns.barplot(x=missing.index, y=missing.values)
            plt.title('Missing Values by Column')
            plt.xlabel('Column')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            missing_path = os.path.join(plots_dir, "missing_values.png")
            plt.savefig(missing_path)
            plt.close()
            plot_paths.append(missing_path)
        
        analysis_results["plot_paths"] = plot_paths
    
    # Save report if requested
    if save_report:
        # Create report directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Convert any NumPy types to native Python types for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Recursively convert all values to serializable format
        def serialize_dict(d):
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    result[key] = serialize_dict(value)
                elif isinstance(value, list):
                    result[key] = [make_serializable(item) if not isinstance(item, dict) 
                                   else serialize_dict(item) for item in value]
                else:
                    result[key] = make_serializable(value)
            return result
        
        serializable_results = serialize_dict(analysis_results)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"[ANALYZER] Analysis report saved to {report_path}")
        except Exception as e:
            print(f"[ANALYZER] Error saving analysis report to {report_path}: {str(e)}")
    
    # Return the analysis results
    return analysis_results


def analyze_column_types(df: pd.DataFrame, config: Any) -> Dict[str, Dict[str, str]]:
    """
    Analyze and classify column types in the DataFrame.
    
    Args:
        df: DataFrame to analyze
        config: Configuration object
        
    Returns:
        Dictionary with column types and details
    """
    column_types = {}
    
    for col in df.columns:
        # Skip columns with all missing values
        if df[col].isna().all():
            column_types[col] = {
                "type": "unknown",
                "subtype": "unknown",
                "reason": "All values missing"
            }
            continue
            
        # Check data type
        dtype = df[col].dtype
        
        # Check for numeric columns
        if pd.api.types.is_numeric_dtype(dtype):
            # Check if it's likely to be categorical
            unique_count = df[col].nunique()
            if unique_count <= config.config["categorical_threshold"]:
                if unique_count <= 2:
                    column_types[col] = {
                        "type": "boolean",
                        "subtype": "binary_numeric",
                        "unique_values": unique_count
                    }
                else:
                    column_types[col] = {
                        "type": "categorical",
                        "subtype": "numeric_categorical",
                        "unique_values": unique_count
                    }
            else:
                column_types[col] = {
                    "type": "numeric",
                    "subtype": "float" if pd.api.types.is_float_dtype(dtype) else "integer",
                    "unique_values": unique_count
                }
                
                # Check if it might be an ID column
                if col.lower().endswith('id') or 'id_' in col.lower() or col.lower() == 'id':
                    if df[col].nunique() / len(df) > 0.9:
                        column_types[col]["likely_id"] = True
        
        # Check for datetime columns
        elif pd.api.types.is_datetime64_dtype(dtype):
            column_types[col] = {
                "type": "datetime",
                "subtype": "datetime",
                "unique_values": df[col].nunique()
            }
        
        # Check for string/object columns
        else:
            # Try to convert to datetime
            try:
                # Check if values match common date formats
                date_formats = config.config["date_detection_patterns"]
                is_date = False
                
                # Sample some values for date detection
                sample_size = min(100, len(df))
                sample_values = df[col].dropna().sample(sample_size) if len(df) > sample_size else df[col].dropna()
                
                for date_format in date_formats:
                    try:
                        # Try to parse the first few non-null values
                        for val in sample_values:
                            datetime.strptime(str(val), date_format)
                        
                        # If no exception was raised, it's likely a date column
                        column_types[col] = {
                            "type": "datetime",
                            "subtype": "string_date",
                            "format": date_format,
                            "unique_values": df[col].nunique()
                        }
                        is_date = True
                        break
                    except (ValueError, TypeError):
                        continue
                
                if not is_date:
                    # If not a date, check if categorical or text
                    unique_count = df[col].nunique()
                    
                    if unique_count <= config.config["categorical_threshold"]:
                        column_types[col] = {
                            "type": "categorical",
                            "subtype": "string_categorical",
                            "unique_values": unique_count
                        }
                    else:
                        # If many unique values, likely text data
                        avg_len = df[col].astype(str).str.len().mean()
                        
                        if avg_len > 100:  # Long text suggests descriptive content
                            column_types[col] = {
                                "type": "text",
                                "subtype": "long_text",
                                "avg_length": avg_len,
                                "unique_values": unique_count
                            }
                        else:
                            column_types[col] = {
                                "type": "text",
                                "subtype": "short_text",
                                "avg_length": avg_len,
                                "unique_values": unique_count
                            }
            
            except Exception:
                # If datetime conversion fails, treat as categorical or text
                unique_count = df[col].nunique()
                
                if unique_count <= config.config["categorical_threshold"]:
                    column_types[col] = {
                        "type": "categorical",
                        "subtype": "string_categorical",
                        "unique_values": unique_count
                    }
                else:
                    column_types[col] = {
                        "type": "text",
                        "subtype": "general_text",
                        "unique_values": unique_count
                    }
        
        # Check for geographic data
        geo_keywords = config.config["geo_column_keywords"]
        if any(keyword in col.lower() for keyword in geo_keywords):
            column_types[col]["geographic"] = True
    
    return column_types


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing values in the DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with missing value statistics
    """
    missing_info = {
        "total_missing": df.isna().sum().sum(),
        "total_missing_percentage": (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        "columns_with_missing": {}
    }
    
    # Get counts and percentages by column
    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / df.shape[0]) * 100
    
    for col in df.columns:
        if missing_counts[col] > 0:
            missing_info["columns_with_missing"][col] = {
                "count": int(missing_counts[col]),
                "percentage": float(missing_percentages[col])
            }
    
    # Identify columns with high percentages of missing values
    high_missing_threshold = 30.0  # 30% or more
    high_missing_cols = [col for col, info in missing_info["columns_with_missing"].items() 
                        if info["percentage"] >= high_missing_threshold]
    
    if high_missing_cols:
        missing_info["high_missing_columns"] = high_missing_cols
    
    return missing_info


def analyze_numeric_columns(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Analyze statistics for numeric columns.
    
    Args:
        df: DataFrame to analyze
        numeric_columns: List of numeric column names
        
    Returns:
        Dictionary with numeric column statistics
    """
    if not numeric_columns:
        return {}
    
    numeric_stats = {}
    
    for col in numeric_columns:
        # Calculate basic statistics
        stats = df[col].describe().to_dict()
        
        # Add additional statistics
        stats["skewness"] = float(df[col].skew())
        stats["kurtosis"] = float(df[col].kurtosis())
        
        # Add IQR and detect potential outliers
        Q1 = float(df[col].quantile(0.25))
        Q3 = float(df[col].quantile(0.75))
        IQR = Q3 - Q1
        
        stats["iqr"] = IQR
        stats["lower_bound"] = Q1 - 1.5 * IQR
        stats["upper_bound"] = Q3 + 1.5 * IQR
        
        # Count potential outliers
        outliers = df[(df[col] < stats["lower_bound"]) | (df[col] > stats["upper_bound"])][col]
        stats["outlier_count"] = len(outliers)
        stats["outlier_percentage"] = (len(outliers) / df[col].count()) * 100 if df[col].count() > 0 else 0
        
        numeric_stats[col] = stats
    
    return numeric_stats


def analyze_categorical_columns(df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze statistics for categorical columns.
    
    Args:
        df: DataFrame to analyze
        categorical_columns: List of categorical column names
        
    Returns:
        Dictionary with categorical column statistics
    """
    if not categorical_columns:
        return {}
    
    categorical_stats = {}
    
    for col in categorical_columns:
        # Get value counts
        value_counts = df[col].value_counts().to_dict()
        
        # Convert keys to strings for JSON compatibility
        value_counts = {str(k): v for k, v in value_counts.items()}
        
        # Calculate percentage for each value
        total_non_null = df[col].count()
        value_percentages = {k: (v / total_non_null) * 100 for k, v in value_counts.items()}
        
        # Store statistics
        stats = {
            "unique_count": df[col].nunique(),
            "top_value": str(df[col].mode()[0]) if not df[col].mode().empty else None,
            "top_value_count": int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0,
            "value_counts": value_counts,
            "value_percentages": value_percentages,
            "missing_count": int(df[col].isna().sum()),
            "missing_percentage": float((df[col].isna().sum() / len(df)) * 100)
        }
        
        categorical_stats[col] = stats
    
    return categorical_stats


def analyze_correlations(df_numeric: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze correlations between numeric columns.
    
    Args:
        df_numeric: DataFrame with numeric columns only
        
    Returns:
        Dictionary with correlation statistics
    """
    # Calculate correlation matrix
    corr_matrix = df_numeric.corr()
    
    # Convert to dictionary format
    corr_dict = {}
    for col1 in corr_matrix.columns:
        corr_dict[col1] = {}
        for col2 in corr_matrix.columns:
            corr_dict[col1][col2] = float(corr_matrix.loc[col1, col2])
    
    # Find highest correlations (excluding self-correlations)
    high_correlations = []
    
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2:
                corr_value = abs(corr_matrix.loc[col1, col2])
                if corr_value >= 0.5:  # Threshold for strong correlation
                    high_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": float(corr_matrix.loc[col1, col2]),
                        "abs_correlation": float(corr_value)
                    })
    
    # Sort by absolute correlation, descending
    high_correlations = sorted(high_correlations, key=lambda x: x["abs_correlation"], reverse=True)
    
    return {
        "correlation_matrix": corr_dict,
        "high_correlations": high_correlations
    }


def detect_potential_targets(df: pd.DataFrame, 
                           numeric_columns: List[str], 
                           column_types: Dict[str, Dict[str, str]],
                           correlations: Dict[str, Any],
                           config: Any) -> Dict[str, Any]:
    """
    Detect potential target variables in the dataset.
    
    Args:
        df: DataFrame to analyze
        numeric_columns: List of numeric column names
        column_types: Dictionary with column type information
        correlations: Dictionary with correlation information
        config: Configuration object
        
    Returns:
        Dictionary with potential target variables
    """
    potential_targets = {}
    
    # Skip if no numeric columns
    if not numeric_columns:
        potential_targets["error"] = "No numeric columns found for target detection"
        return potential_targets
    
    # Get domain indicators from config
    domain_info = config.config.get("target_indicators", {})
    detected_domain = detect_dataset_domain(df, config)["detected_domain"]
    
    domain_target_indicators = domain_info.get(detected_domain, [])
    
    # Score each numeric column as a potential target
    target_scores = {}
    
    for col in numeric_columns:
        # Skip likely ID columns
        if column_types[col].get("likely_id", False):
            continue
        
        score = 0
        
        # 1. Name-based score: Check if column name indicates a target
        col_lower = col.lower()
        
        # Check domain-specific target indicators
        for indicator in domain_target_indicators:
            if indicator.lower() in col_lower:
                score += 10
                break
        
        # Generic target indicators (output, target, result, etc.)
        for generic_target in ["output", "target", "result", "outcome", "dependent", "response"]:
            if generic_target in col_lower:
                score += 5
                break
        
        # 2. Correlation-based score
        if correlations and "high_correlations" in correlations:
            # Columns with many high correlations might be targets
            col_high_corrs = sum(1 for corr in correlations["high_correlations"] 
                              if corr["column1"] == col or corr["column2"] == col)
            score += min(col_high_corrs, 5)  # Cap at 5 points
        
        # 3. Statistical properties score
        if col in df.columns:
            # Prefer columns with reasonable variance (not too high or low)
            try:
                variance = df[col].var()
                if variance > 0:
                    # Normalize variance to the mean for better comparison
                    relative_variance = variance / (df[col].mean() ** 2) if df[col].mean() != 0 else variance
                    if 0.01 < relative_variance < 100:
                        score += 3
            except:
                pass
            
            # Prefer columns with fewer zeros (if dealing with positive metrics)
            zero_percentage = (df[col] == 0).mean() * 100
            if zero_percentage < 10:
                score += 2
        
        # 4. Position score: Last column is often a target
        if list(df.columns).index(col) == len(df.columns) - 1:
            score += 2
        
        # Store the score
        target_scores[col] = score
    
    # Get top potential targets
    sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Store top 3 potential targets
    top_targets = [{"column": col, "score": score} for col, score in sorted_targets[:3]]
    potential_targets["potential_target_columns"] = top_targets
    
    # Recommend the highest-scoring target
    if top_targets:
        potential_targets["recommended_target"] = top_targets[0]["column"]
        
        # Also suggest predictor columns (correlated with target)
        recommended_target = potential_targets["recommended_target"]
        
        if correlations and "correlation_matrix" in correlations:
            predictor_correlations = []
            
            for col in numeric_columns:
                if col != recommended_target and col in correlations["correlation_matrix"].get(recommended_target, {}):
                    correlation = correlations["correlation_matrix"][recommended_target][col]
                    abs_correlation = abs(correlation)
                    
                    if abs_correlation > 0.1:  # Minimal correlation threshold
                        predictor_correlations.append({
                            "column": col,
                            "correlation": correlation,
                            "abs_correlation": abs_correlation
                        })
            
            # Sort by absolute correlation, descending
            sorted_predictors = sorted(predictor_correlations, key=lambda x: x["abs_correlation"], reverse=True)
            
            # Store top predictors
            potential_targets["recommended_predictors"] = [p["column"] for p in sorted_predictors[:5]]
            potential_targets["predictor_correlations"] = sorted_predictors[:5]
    
    return potential_targets


def detect_dataset_domain(df: pd.DataFrame, config: Any) -> Dict[str, str]:
    """
    Detect the likely domain of the dataset.
    
    Args:
        df: DataFrame to analyze
        config: Configuration object
        
    Returns:
        Dictionary with detected domain information
    """
    domain_scores = {}
    
    # Get domain indicators from config
    domain_indicators = config.config.get("domain_indicators", {})
    
    for domain, indicators in domain_indicators.items():
        score = 0
        
        # Check column names
        for col in df.columns:
            col_lower = col.lower()
            for indicator in indicators:
                if indicator.lower() in col_lower:
                    score += 2
        
        # Check for categorical values (in string columns)
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Sample some values
                sample_size = min(100, len(df))
                sample_values = df[col].dropna().astype(str).sample(sample_size) if len(df) > sample_size else df[col].dropna().astype(str)
                
                for val in sample_values:
                    val_lower = str(val).lower()
                    for indicator in indicators:
                        if indicator.lower() in val_lower:
                            score += 1
                            break
            except:
                continue
        
        domain_scores[domain] = score
    
    # Determine the most likely domain
    if domain_scores:
        max_score = max(domain_scores.values())
        
        # If all scores are 0, use "generic"
        if max_score == 0:
            detected_domain = "generic"
        else:
            # Get domain with highest score
            detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
    else:
        detected_domain = "generic"
    
    return {
        "detected_domain": detected_domain,
        "domain_scores": domain_scores
    }


def generate_summary(analysis_results: Dict[str, Any]) -> List[str]:
    """
    Generate a summary of the dataset analysis.
    
    Args:
        analysis_results: Dictionary with analysis results
        
    Returns:
        List of summary points
    """
    summary = []
    
    # Basic dataset info
    dataset_info = analysis_results.get("dataset_info", {})
    if dataset_info:
        summary.append(
            f"Dataset contains {dataset_info.get('rows', 0):,} rows and "
            f"{dataset_info.get('columns', 0)} columns, using "
            f"{dataset_info.get('memory_usage', 0):.2f} MB of memory."
        )
    
    # Column type summary
    column_types = analysis_results.get("column_types", {})
    if column_types:
        # Count columns by type
        type_counts = {}
        for col, info in column_types.items():
            col_type = info.get("type", "unknown")
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        type_summary = ", ".join([f"{count} {col_type}" for col_type, count in type_counts.items()])
        summary.append(f"Column types: {type_summary}.")
    
    # Missing values summary
    missing_values = analysis_results.get("missing_values", {})
    if missing_values:
        total_missing = missing_values.get("total_missing", 0)
        missing_percentage = missing_values.get("total_missing_percentage", 0)
        
        if total_missing > 0:
            summary.append(
                f"Dataset contains {total_missing:,} missing values ({missing_percentage:.2f}% of all cells)."
            )
            
            high_missing_cols = missing_values.get("high_missing_columns", [])
            if high_missing_cols:
                if len(high_missing_cols) <= 3:
                    col_list = ", ".join(high_missing_cols)
                    summary.append(f"Columns with high missing rates: {col_list}.")
                else:
                    summary.append(f"{len(high_missing_cols)} columns have high missing rates (>30%).")
    
    # Potential target summary
    potential_targets = analysis_results.get("potential_targets", {})
    if potential_targets:
        recommended_target = potential_targets.get("recommended_target")
        if recommended_target:
            summary.append(f"Recommended target variable: '{recommended_target}'.")
            
            # Add predictor information if available
            recommended_predictors = potential_targets.get("recommended_predictors", [])
            if recommended_predictors:
                if len(recommended_predictors) <= 3:
                    pred_list = ", ".join([f"'{pred}'" for pred in recommended_predictors])
                    summary.append(f"Top predictors for '{recommended_target}': {pred_list}.")
                else:
                    summary.append(f"Found {len(recommended_predictors)} potential predictor variables.")
    
    # Domain summary
    dataset_domain = analysis_results.get("dataset_domain", {})
    if dataset_domain:
        detected_domain = dataset_domain.get("detected_domain")
        if detected_domain and detected_domain != "generic":
            summary.append(f"Dataset appears to be from the {detected_domain} domain.")
    
    # Add recommendations based on data quality
    numeric_stats = analysis_results.get("numeric_stats", {})
    if numeric_stats:
        outlier_cols = []
        for col, stats in numeric_stats.items():
            if stats.get("outlier_percentage", 0) > 5:
                outlier_cols.append(col)
        
        if outlier_cols:
            if len(outlier_cols) <= 3:
                col_list = ", ".join([f"'{col}'" for col in outlier_cols])
                summary.append(f"Columns with significant outliers: {col_list}.")
            else:
                summary.append(f"{len(outlier_cols)} numeric columns have significant outliers.")
    
    return summary