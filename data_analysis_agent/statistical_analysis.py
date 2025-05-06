import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Any, Union

def calculate_advanced_statistics(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate advanced statistics for numeric columns including skewness, kurtosis, and confidence intervals.
    
    Args:
        df: The DataFrame to analyze
        numeric_columns: Optional list of numeric columns to analyze. If None, all numeric columns are used.
        
    Returns:
        A dictionary with column names as keys and dictionaries of statistics as values
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    results = {}
    
    for column in numeric_columns:
        # Skip if column doesn't exist
        if column not in df.columns:
            continue
            
        # Skip if not numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
            
        # Get non-null values for the column
        values = df[column].dropna()
        
        if len(values) < 2:
            continue
            
        stats_dict = {}
        
        # Basic statistics
        stats_dict['mean'] = values.mean()
        stats_dict['median'] = values.median()
        stats_dict['std'] = values.std()
        
        # Advanced statistics
        stats_dict['skewness'] = values.skew()
        stats_dict['kurtosis'] = values.kurtosis()
        
        # 95% confidence interval for the mean
        ci_low, ci_high = stats.t.interval(
            confidence=0.95,
            df=len(values)-1,
            loc=values.mean(),
            scale=stats.sem(values)
        )
        stats_dict['ci_95_low'] = ci_low
        stats_dict['ci_95_high'] = ci_high
        
        # Normality test (Shapiro-Wilk)
        shapiro_test = stats.shapiro(values)
        stats_dict['shapiro_stat'] = shapiro_test[0]
        stats_dict['shapiro_p_value'] = shapiro_test[1]
        stats_dict['is_normal'] = shapiro_test[1] > 0.05  # p > 0.05 suggests normality
        
        results[column] = stats_dict
    
    return results

def calculate_mode_statistics(df: pd.DataFrame, group_column: str, 
                              numeric_columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate statistics for each group in group_column for the specified numeric columns.
    
    Args:
        df: The DataFrame to analyze
        group_column: Column to group by (e.g., 'Mode')
        numeric_columns: Optional list of numeric columns to analyze. If None, all numeric columns are used.
        
    Returns:
        A nested dictionary with structure {column: {group: {statistic: value}}}
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
        
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    results = {}
    
    for column in numeric_columns:
        # Skip if column doesn't exist or is not numeric
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue
            
        results[column] = {}
        
        # For each group in the group column
        for group_value, group_df in df.groupby(group_column):
            values = group_df[column].dropna()
            
            if len(values) < 2:
                continue
                
            stats_dict = {}
            
            # Basic statistics
            stats_dict['mean'] = values.mean()
            stats_dict['median'] = values.median()
            stats_dict['std'] = values.std()
            stats_dict['count'] = len(values)
            
            # Advanced statistics
            stats_dict['skewness'] = values.skew()
            stats_dict['kurtosis'] = values.kurtosis()
            
            # 95% confidence interval for the mean
            ci_low, ci_high = stats.t.interval(
                confidence=0.95,
                df=len(values)-1,
                loc=values.mean(),
                scale=stats.sem(values)
            )
            stats_dict['ci_95_low'] = ci_low
            stats_dict['ci_95_high'] = ci_high
            
            results[column][group_value] = stats_dict
    
    return results

def perform_anova(df: pd.DataFrame, group_column: str, numeric_column: str) -> Dict[str, Any]:
    """
    Perform one-way ANOVA test to check if there's a significant difference between groups.
    
    Args:
        df: The DataFrame to analyze
        group_column: Column containing the groups (e.g., 'Mode')
        numeric_column: Numeric column to analyze (e.g., 'Time')
        
    Returns:
        Dictionary with ANOVA results
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
        
    if numeric_column not in df.columns:
        raise ValueError(f"Numeric column '{numeric_column}' not found in DataFrame")
        
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        raise ValueError(f"Column '{numeric_column}' is not numeric")
    
    # Create lists of data for each group
    groups = []
    group_names = []
    
    for group_value, group_df in df.groupby(group_column):
        values = group_df[numeric_column].dropna()
        if len(values) > 1:  # Need at least 2 values for variance
            groups.append(values)
            group_names.append(group_value)
    
    if len(groups) < 2:
        return {
            "error": "Not enough groups with sufficient data for ANOVA",
            "is_significant": False
        }
    
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "is_significant": p_value < 0.05,  # p < 0.05 indicates significant difference
        "groups": group_names
    }

def perform_tukey_hsd(df: pd.DataFrame, group_column: str, numeric_column: str) -> Dict[str, Any]:
    """
    Perform Tukey's HSD (Honest Significant Difference) test for pairwise comparisons 
    after ANOVA indicates significant differences.
    
    Args:
        df: The DataFrame to analyze
        group_column: Column containing the groups (e.g., 'Mode')
        numeric_column: Numeric column to analyze (e.g., 'Time')
        
    Returns:
        Dictionary with Tukey HSD results
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
        
    if numeric_column not in df.columns:
        raise ValueError(f"Numeric column '{numeric_column}' not found in DataFrame")
        
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        raise ValueError(f"Column '{numeric_column}' is not numeric")
    
    # First check if ANOVA shows significant differences
    anova_result = perform_anova(df, group_column, numeric_column)
    
    if not anova_result.get("is_significant", False):
        return {
            "message": "ANOVA did not show significant differences between groups",
            "anova_result": anova_result,
            "pairwise_results": {}
        }
    
    # Prepare data for Tukey's HSD
    # Create a DataFrame with the group as a separate column
    data = []
    for group_value, group_df in df.groupby(group_column):
        for value in group_df[numeric_column].dropna():
            data.append({'group': group_value, 'value': value})
    
    if not data:
        return {
            "error": "No valid data for Tukey HSD test",
            "anova_result": anova_result
        }
    
    data_df = pd.DataFrame(data)
    
    # Perform Tukey's HSD test
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey_result = pairwise_tukeyhsd(data_df['value'], data_df['group'], alpha=0.05)
        
        # Format results
        pairwise_results = []
        for i, (group1, group2, reject, _, _, _) in enumerate(zip(
            tukey_result.groupsunique[tukey_result.pairindices[:, 0]],
            tukey_result.groupsunique[tukey_result.pairindices[:, 1]],
            tukey_result.reject
        )):
            mean1 = df[df[group_column] == group1][numeric_column].mean()
            mean2 = df[df[group_column] == group2][numeric_column].mean()
            
            pairwise_results.append({
                "group1": group1,
                "group2": group2,
                "mean_difference": abs(mean1 - mean2),
                "is_significant": bool(reject),
                "p_value": tukey_result.pvalues[i]
            })
        
        return {
            "anova_result": anova_result,
            "pairwise_results": pairwise_results
        }
    except Exception as e:
        return {
            "error": f"Error performing Tukey HSD test: {str(e)}",
            "anova_result": anova_result
        }

def generate_advanced_plots(df: pd.DataFrame, output_dir: str = "plots/advanced", 
                       focus_columns: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Generate advanced statistical plots including density plots, Q-Q plots, violin plots,
    correlation heatmaps, and pair plots. Works with any dataset by automatically
    detecting appropriate columns.
    
    Args:
        df: The DataFrame to visualize
        output_dir: Directory to save plots in
        focus_columns: Optional dictionary with keys 'numeric' and 'categorical'
                      containing lists of column names to focus on
        
    Returns:
        List of saved plot file paths
    """
    plot_paths = []
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    try:
        # Detect numeric and categorical columns automatically
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Use focus columns if provided
        if focus_columns:
            if 'numeric' in focus_columns and focus_columns['numeric']:
                numeric_cols = [col for col in focus_columns['numeric'] if col in df.columns]
            if 'categorical' in focus_columns and focus_columns['categorical']:
                categorical_cols = [col for col in focus_columns['categorical'] if col in df.columns]
        
        # Limit to columns with reasonable number of unique values for categorical columns
        categorical_cols = [col for col in categorical_cols 
                           if col in df.columns and df[col].nunique() <= 10]
        
        # Select top numeric columns (by non-null count) if we have more than 3
        if len(numeric_cols) > 3:
            numeric_cols = sorted(
                numeric_cols, 
                key=lambda col: df[col].count(), 
                reverse=True
            )[:3]
        
        # Select primary categorical column (if available)
        primary_cat_col = None
        if categorical_cols:
            # Prefer column with 3-7 categories as primary grouping variable
            good_range_cols = [col for col in categorical_cols 
                              if 3 <= df[col].nunique() <= 7]
            primary_cat_col = good_range_cols[0] if good_range_cols else categorical_cols[0]
        
        # 1. Density plots for numeric columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            # Plot the density for the whole dataset
            sns.kdeplot(data=df, x=col, fill=True, color='gray', alpha=0.5, label='Overall')
            
            # Plot density by primary categorical column if available
            if primary_cat_col:
                for category in df[primary_cat_col].dropna().unique():
                    subset = df[df[primary_cat_col] == category]
                    if len(subset) > 1:  # Need at least 2 points for density
                        sns.kdeplot(data=subset, x=col, fill=True, alpha=0.3, label=str(category))
            
            plt.title(f'Density Plot of {col}' + 
                     (f' by {primary_cat_col}' if primary_cat_col else ''))
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()
            
            density_path = os.path.join(output_dir, f"{col.lower()}_density_plot.png")
            plt.savefig(density_path)
            plt.close()
            plot_paths.append(density_path)
            print(f"Saved plot: {density_path}")
        
        # 2. Q-Q plots for numeric columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 8))
            stats.probplot(df[col].dropna(), dist="norm", plot=plt)
            plt.title(f'Q-Q Plot of {col}')
            
            qq_path = os.path.join(output_dir, f"{col.lower()}_qq_plot.png")
            plt.savefig(qq_path)
            plt.close()
            plot_paths.append(qq_path)
            print(f"Saved plot: {qq_path}")
        
        # 3. Violin plots (if we have both numeric and categorical columns)
        if numeric_cols and primary_cat_col:
            for num_col in numeric_cols:
                plt.figure(figsize=(12, 7))
                sns.violinplot(data=df, x=primary_cat_col, y=num_col, inner='quart', hue=primary_cat_col, legend=False)
                plt.title(f'Violin Plot of {num_col} by {primary_cat_col}')
                plt.xlabel(primary_cat_col)
                plt.ylabel(num_col)
                
                violin_path = os.path.join(output_dir, f"{num_col.lower()}_{primary_cat_col.lower()}_violin_plot.png")
                plt.savefig(violin_path)
                plt.close()
                plot_paths.append(violin_path)
                print(f"Saved plot: {violin_path}")
        
        # 4. Correlation heatmap
        numeric_df = df[numeric_cols] if numeric_cols else df.select_dtypes(include=['number'])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = numeric_df.corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                linewidths=0.5, 
                fmt='.2f',
                center=0
            )
            plt.title('Correlation Heatmap')
            
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            plot_paths.append(heatmap_path)
            print(f"Saved plot: {heatmap_path}")
        
        # 5. Pair plot with regression lines
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            plot_vars = numeric_cols[:min(len(numeric_cols), 3)]  # Limit to maximum 3 numeric columns
            
            if primary_cat_col:
                pair_plot = sns.pairplot(
                    data=df, 
                    vars=plot_vars, 
                    hue=primary_cat_col, 
                    kind='scatter',
                    diag_kind='kde',
                    plot_kws={'alpha': 0.6},
                    height=2.5
                )
            else:
                pair_plot = sns.pairplot(
                    data=df, 
                    vars=plot_vars, 
                    kind='scatter',
                    diag_kind='kde',
                    plot_kws={'alpha': 0.6},
                    height=3
                )
            
            pair_plot.fig.suptitle('Pair Plot with Relationships', y=1.02)
            
            pairplot_path = os.path.join(output_dir, "pair_plot.png")
            pair_plot.savefig(pairplot_path)
            plt.close()
            plot_paths.append(pairplot_path)
            print(f"Saved plot: {pairplot_path}")
        
        return plot_paths
        
    except Exception as e:
        print(f"Error generating advanced plots: {str(e)}")
        return plot_paths

def generate_statistical_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive statistical report including advanced statistics 
    and significance testing.
    
    Args:
        df: The DataFrame to analyze
        
    Returns:
        Dictionary with statistical report data
    """
    report = {
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        },
        "advanced_statistics": {},
        "group_statistics": {},
        "significance_tests": {}
    }
    
    # Calculate advanced statistics for numeric columns
    report["advanced_statistics"] = calculate_advanced_statistics(df)
    
    # Calculate statistics by Mode (if present)
    if 'Mode' in df.columns:
        report["group_statistics"]["Mode"] = calculate_mode_statistics(df, 'Mode')
        
        # Perform ANOVA and Tukey HSD for Time and Distance by Mode
        for col in ['Time', 'Distance']:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                anova_result = perform_anova(df, 'Mode', col)
                
                if anova_result.get("is_significant", False):
                    tukey_result = perform_tukey_hsd(df, 'Mode', col)
                    report["significance_tests"][col] = tukey_result
                else:
                    report["significance_tests"][col] = {
                        "anova_result": anova_result,
                        "message": "No significant differences found between modes"
                    }
    
    return report