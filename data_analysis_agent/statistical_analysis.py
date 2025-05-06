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

def generate_advanced_plots(df: pd.DataFrame, output_dir: str = "plots/advanced") -> List[str]:
    """
    Generate advanced statistical plots for the given DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to generate plots for
        output_dir (str): Directory to save plots in
        
    Returns:
        List[str]: List of plot file paths
    """
    print("[ADVANCED ANALYSIS] Generating advanced visualizations...")
    plot_paths = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep track of plots to generate
    plot_generators = []
    
    # Use a plot cache to avoid duplicate generation
    plot_cache = {}
    
    try:
        # Select numeric columns for plotting
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select categorical columns for grouping
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Filter to important columns to avoid generating too many plots
        # For numeric columns, prefer those with variation and fewer missing values
        if len(numeric_cols) > 3:
            # Calculate coefficient of variation and missing percentage for each column
            col_metrics = {}
            for col in numeric_cols:
                if df[col].std() == 0 or df[col].mean() == 0:
                    cv = 0
                else:
                    cv = df[col].std() / abs(df[col].mean())
                missing_pct = df[col].isna().mean()
                col_metrics[col] = (cv, missing_pct)
            
            # Sort columns by coefficient of variation (higher is better) and missing percentage (lower is better)
            sorted_cols = sorted(col_metrics.items(), key=lambda x: (x[1][0], -x[1][1]), reverse=True)
            numeric_cols = [col for col, _ in sorted_cols[:3]]
        
        # For categorical columns, prefer those with reasonable number of categories
        if len(categorical_cols) > 2:
            # Filter to columns with reasonable number of categories (2-10)
            filtered_cat_cols = []
            for col in categorical_cols:
                n_unique = df[col].nunique()
                if 2 <= n_unique <= 10:
                    filtered_cat_cols.append((col, n_unique))
            
            # Sort by number of categories (closer to 5 is better)
            sorted_cat_cols = sorted(filtered_cat_cols, key=lambda x: abs(x[1] - 5))
            categorical_cols = [col for col, _ in sorted_cat_cols[:2]]
        
        # --- OPTIMIZATION: Prepare all plot generation tasks ---
        
        # 1. Density plots for numeric columns
        for col in numeric_cols:
            # Skip if already in cache
            density_path = os.path.join(output_dir, f"{col.lower().replace(' ', '_')}_density_plot.png")
            if density_path in plot_cache:
                plot_paths.append(density_path)
                continue
                
            plot_generators.append({
                'type': 'density',
                'data': df[col].dropna(),
                'col': col,
                'path': density_path
            })
        
        # 2. Q-Q plots for numeric columns
        for col in numeric_cols:
            # Skip if already in cache
            qq_path = os.path.join(output_dir, f"{col.lower().replace(' ', '_')}_qq_plot.png")
            if qq_path in plot_cache:
                plot_paths.append(qq_path)
                continue
                
            plot_generators.append({
                'type': 'qq',
                'data': df[col].dropna(),
                'col': col,
                'path': qq_path
            })
        
        # 3. Violin plots for numeric columns by categorical columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    # Skip if categorical column has too many unique values
                    if df[cat_col].nunique() > 10:
                        continue
                        
                    # Skip if already in cache
                    violin_path = os.path.join(output_dir, f"{num_col.lower().replace(' ', '_')}_{cat_col.lower().replace(' ', '_')}_violin_plot.png")
                    if violin_path in plot_cache:
                        plot_paths.append(violin_path)
                        continue
                        
                    plot_generators.append({
                        'type': 'violin',
                        'data': df,
                        'x': cat_col,
                        'y': num_col,
                        'path': violin_path
                    })
        
        # 4. Correlation heatmap for numeric columns
        if len(numeric_cols) > 1:
            # Skip if already in cache
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            if heatmap_path not in plot_cache:
                plot_generators.append({
                    'type': 'heatmap',
                    'data': df[numeric_cols],
                    'path': heatmap_path
                })
            else:
                plot_paths.append(heatmap_path)
        
        # 5. Pair plot for numeric columns and one categorical column
        if len(numeric_cols) > 1:
            # Skip if already in cache
            pair_path = os.path.join(output_dir, "pair_plot.png")
            if pair_path not in plot_cache:
                # Use first categorical column for hue if available
                hue_col = categorical_cols[0] if categorical_cols else None
                plot_generators.append({
                    'type': 'pair',
                    'data': df,
                    'cols': numeric_cols,
                    'hue': hue_col,
                    'path': pair_path
                })
            else:
                plot_paths.append(pair_path)
        
        # --- OPTIMIZATION: Execute all plot generators with error handling for each ---
        for generator in plot_generators:
            try:
                # Create a new figure for each plot to ensure clean state
                plt.figure(figsize=(10, 6))
                
                try:
                    # Generate appropriate plot type
                    if generator['type'] == 'density':
                        sns.kdeplot(generator['data'], fill=True)
                        plt.title(f'Density Plot of {generator["col"]}')
                        plt.xlabel(generator['col'])
                        plt.ylabel('Density')
                        plt.tight_layout()
                        
                    elif generator['type'] == 'qq':
                        from scipy import stats
                        # Ensure data is sorted and not empty
                        data = generator['data'].dropna()
                        if len(data) < 2:
                            print(f"[ADVANCED ANALYSIS] Skipping Q-Q plot for {generator['col']} - insufficient data points")
                            continue
                            
                        # Generate Q-Q plot
                        stats.probplot(data, plot=plt)
                        plt.title(f'Q-Q Plot of {generator["col"]}')
                        plt.tight_layout()
                        
                    elif generator['type'] == 'violin':
                        sns.violinplot(data=generator['data'], x=generator['x'], y=generator['y'])
                        plt.title(f'Violin Plot of {generator["y"]} by {generator["x"]}')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                    elif generator['type'] == 'heatmap':
                        # Calculate correlation matrix
                        corr = generator['data'].corr()
                        # Generate heatmap
                        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                        plt.title('Correlation Heatmap')
                        plt.tight_layout()
                        
                    elif generator['type'] == 'pair':
                        # Close current figure as pairplot creates its own figure
                        plt.close()
                        
                        # For large datasets with many numeric columns, limit the data points and columns
                        # to avoid overwhelming memory
                        data = generator['data']
                        if len(data) > 1000 or len(generator['cols']) > 5:
                            # Sample data if too large
                            if len(data) > 1000:
                                data = data.sample(1000, random_state=42)
                            
                            # Limit columns if too many
                            cols = generator['cols'][:5] if len(generator['cols']) > 5 else generator['cols']
                        else:
                            cols = generator['cols']
                            
                        # Generate pairplot with optional hue
                        if generator['hue'] is not None:
                            # Ensure hue column doesn't have too many unique values
                            unique_hues = data[generator['hue']].nunique()
                            if unique_hues <= 10:
                                g = sns.pairplot(data, vars=cols, hue=generator['hue'])
                            else:
                                g = sns.pairplot(data, vars=cols)
                        else:
                            g = sns.pairplot(data, vars=cols)
                            
                        g.fig.suptitle('Pair Plot of Numeric Variables', y=1.02)
                        
                        # Save the pairplot figure directly
                        g.savefig(generator['path'])
                        plot_paths.append(generator['path'])
                        plot_cache[generator['path']] = True
                        print(f"Saved plot: {generator['path']}")
                        continue  # Skip the regular saving code for pairplot
                    
                    # Save the plot
                    plt.savefig(generator['path'])
                    plot_paths.append(generator['path'])
                    plot_cache[generator['path']] = True
                    print(f"Saved plot: {generator['path']}")
                    
                except Exception as e:
                    print(f"[ADVANCED ANALYSIS] Error generating {generator['type']} plot: {str(e)}")
                    traceback.print_exc()
                    
            finally:
                plt.close()  # Always close the figure to free resources
        
        print(f"[ADVANCED ANALYSIS] Generated {len(plot_paths)} advanced plots")
        
        return plot_paths
        
    except Exception as e:
        error_msg = f"[ADVANCED ANALYSIS] Error in advanced plot generation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return plot_paths  # Return any plots that were successfully generated
    finally:
        # Reset matplotlib to default state
        plt.style.use('default')

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