# filepath: c:\Users\anteb\Desktop\Courses\Projects\data_analysis_ai\data_analysis_agent\pandas_helper.py
import pandas as pd
import os
import traceback
import re
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Optional, Any
from llama_index.experimental.query_engine import PandasQueryEngine
from data_analysis_agent.statistical_analysis import (
    calculate_advanced_statistics,
    calculate_mode_statistics,
    perform_anova,
    perform_tukey_hsd,
    generate_advanced_plots,
    generate_statistical_report
)

class PandasHelper:
    """
    Helper class to manage pandas DataFrame operations and interactions 
    with a PandasQueryEngine, independent of the main workflow context.
    """
    def __init__(self, df: pd.DataFrame, query_engine: PandasQueryEngine):
        self.query_engine = query_engine
        # Work on a copy internally to avoid modifying the original df passed during init
        self._current_df = df.copy() 
        # Attempt to keep query engine synced with the internal df state
        try:
            # This assumes the experimental engine uses _df internally
            self.query_engine._df = self._current_df 
        except AttributeError:
            print("Warning: Could not set initial query_engine._df in PandasHelper.")

    

    async def execute_pandas_query(self, query_str: str) -> str:
        """
        Executes a pandas query string against the current DataFrame state.
        Attempts to distinguish between queries returning results and modifications.
        Args:
            query_str (str): The pandas query/command to execute. Must use 'df'.
        """
        try:
            print(f"Helper executing query: {query_str}")
            
            # --- Refined Heuristic for Modification ---
            # More specific check for assignments or inplace operations
            is_modification = (
                re.search(r"\bdf\s*\[.*\]\s*=", query_str) or  # df['col'] = ...
                re.search(r"\bdf\s*=\s*", query_str) or        # df = ...
                'inplace=True' in query_str or
                re.search(r"\.drop\(.*\)", query_str) or      # .drop(...) might be inplace or assignment
                re.search(r"\.fillna\(.*\)", query_str) or    # .fillna(...) might be inplace or assignment
                re.search(r"\.rename\(.*\)", query_str) or    # .rename(...) might be inplace or assignment
                re.search(r"\.replace\(.*\)", query_str)      # .replace(...) might be inplace or assignment
                # Add other modification patterns if needed
            )
            # --- End Refined Heuristic ---

            if is_modification:
                # --- Modification Logic (using exec) ---
                try:
                    local_vars = {'df': self._current_df.copy()} 
                    global_vars = {'pd': pd}
                    
                    exec(query_str, global_vars, local_vars)
                    
                    modified_df = local_vars['df']
                    
                    # Check if the DataFrame object actually changed
                    # This helps differentiate queries like `x = df[...]` from actual modifications
                    if not self._current_df.equals(modified_df):
                        self._current_df = modified_df
                        try:
                            self.query_engine._df = modified_df
                        except AttributeError:
                            print("Warning: Could not directly update query_engine._df after modification.")
                        result = "Executed modification successfully."
                        print(f"Helper modification result: {result}")
                    else:
                        # If df didn't change, it was likely a query assigning to a variable
                        # Try to capture the result if it's simple (e.g., unique_modes = df['Mode'].unique())
                        # This part is tricky and might need more robust handling
                        result_var_name = query_str.split('=')[0].strip()
                        if result_var_name in local_vars and result_var_name != 'df':
                             result = f"Executed query, result stored in '{result_var_name}': {str(local_vars[result_var_name])[:500]}..."
                             print(f"Helper query (via exec) result: {result}")
                        else:
                             result = "Executed command (likely query assignment), DataFrame unchanged."
                             print(f"Helper exec result: {result}")

                    return result
                    
                except Exception as e:
                    print(f"Helper exec error for query '{query_str}': {e}\n{traceback.format_exc()}")
                    error_msg = f"Error executing modification '{query_str}': {e}"
                    # Handle specific FutureWarning for fillna inplace (Example)
                    if "FutureWarning" in str(e) and "fillna" in query_str and "inplace=True" in query_str:
                         print("Note: Detected FutureWarning with inplace fillna. Consider using assignment syntax like 'df[col] = df[col].fillna(value)' instead.")
                    
                    return error_msg
            else: 
                # --- Query Logic (using query_engine) ---
                try:
                    try:
                         self.query_engine._df = self._current_df # Ensure engine has latest df
                    except AttributeError:
                         print("Warning: Could not directly update query_engine._df before query.")
                         
                    response = await self.query_engine.aquery(query_str) 
                    result = str(response)
                    
                    # Check for known error patterns from the engine's response string
                    if "error" in result.lower() or "Traceback" in result.lower() or "invalid syntax" in result.lower():
                         error_msg = f"Query engine failed for '{query_str}': {result}"
                         print(error_msg)
                         return error_msg
                         
                    print(f"Helper query engine result: {result[:500]}...")
                    return result
                except Exception as e:
                     print(f"Helper error during query_engine.aquery('{query_str}'): {e}\n{traceback.format_exc()}")
                     return f"Error during query_engine.aquery('{query_str}'): {e}"

        except Exception as e:
            print(f"Helper general error processing query '{query_str}': {e}\n{traceback.format_exc()}")
            return f"Error processing query '{query_str}': {e}"
    async def save_dataframe(self, file_path: str) -> str:
        """
        Saves the current DataFrame state to a CSV file.
        Args:
            file_path (str): The full path where the CSV should be saved.
        """
        try:
            output_dir = os.path.dirname(file_path)
            if output_dir: # Check if path includes a directory
                 os.makedirs(output_dir, exist_ok=True)

            print(f"Helper attempting to save DataFrame to: {file_path}")
            # Save the current internal DataFrame state
            self._current_df.to_csv(file_path, index=False) 
            result = f"DataFrame successfully saved to {file_path}"
            print(result)
            return result
        except Exception as e:
            error_msg = f"Error saving DataFrame to '{file_path}': {e}"
            print(error_msg)
            return error_msg

    def get_final_dataframe(self) -> pd.DataFrame:
        """Returns the final state of the DataFrame managed by the helper."""
        return self._current_df
    
    async def generate_plots(self, output_dir: str = "plots", focus_columns: list = None) -> list[str]:
        """
        Generates standard plots (histogram, countplot, scatterplot, boxplot)
        for the current DataFrame based on the data types and structure.
        Automatically determines appropriate columns for visualization if focus_columns is not provided.

        Args:
            output_dir (str): The directory to save the plots in. Defaults to 'plots'.
            focus_columns (list): Optional list of column names to focus on for visualization.
                                  If not provided, columns will be auto-selected based on data types.
        
        Returns:
            list[str]: List of saved plot file paths.
        """
        plot_paths = []
        df = self._current_df # Use the helper's current dataframe

        if df is None:
            return ["Error: DataFrame not available in helper."]

        try:
            os.makedirs(output_dir, exist_ok=True)
            # Set style once at the beginning
            sns.set_theme(style="whitegrid")
            
            # Cache for plot paths to avoid duplicate generation
            plot_cache = {}
            
            # Auto-detect appropriate columns if focus_columns not provided
            if not focus_columns:
                # Find numeric columns for distribution and correlation plots
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                # Find categorical columns for countplots
                categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                
                # Select columns based on availability
                if not numeric_cols and not categorical_cols:
                    return ["Error: No suitable numeric or categorical columns found for plotting."]
                
                # Use at most 3 columns to avoid generating too many plots
                focus_numeric = numeric_cols[:2] if numeric_cols else []
                focus_categorical = categorical_cols[:1] if categorical_cols else []
                focus_columns = focus_numeric + focus_categorical
            
            print(f"Generating plots for columns: {focus_columns}")
            
            # --- OPTIMIZATION: Prepare all figures at once ---
            # This reduces the overhead of creating and destroying figure objects
            plot_configs = []
            
            # Generate plots based on column data types
            for col in focus_columns:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                    continue
                    
                # Check column data type
                if pd.api.types.is_numeric_dtype(df[col]):
                    # 1. Histogram for numeric columns - add to configs
                    plot_configs.append({
                        'type': 'histogram',
                        'data': df[col].dropna(),
                        'col': col,
                        'path': os.path.join(output_dir, f"{col.lower().replace(' ', '_')}_histogram.png")
                    })
                
                elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                    # 2. Countplot for categorical columns
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 20:  # Too many categories
                        print(f"Skipping countplot for column '{col}' - too many unique values ({len(value_counts)})")
                        continue
                    
                    plot_configs.append({
                        'type': 'countplot', 
                        'data': df,
                        'x': col,
                        'order': df[col].value_counts().iloc[:15].index,
                        'col': col,
                        'path': os.path.join(output_dir, f"{col.lower().replace(' ', '_')}_countplot.png")
                    })
            
            # Generate relationship plots if we have numeric columns
            numeric_cols = [col for col in focus_columns if pd.api.types.is_numeric_dtype(df[col])]
            categorical_cols = [col for col in focus_columns if pd.api.types.is_categorical_dtype(df[col]) or 
                              pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])]
            
            # 3. Scatter plots between numeric columns
            if len(numeric_cols) >= 2:
                for i in range(len(numeric_cols)-1):
                    for j in range(i+1, len(numeric_cols)):
                        x_col = numeric_cols[i]
                        y_col = numeric_cols[j]
                        
                        # If we have a categorical column, use it for hue
                        if categorical_cols:
                            hue_col = categorical_cols[0]
                            unique_values = df[hue_col].nunique()
                            # Only use hue if not too many categories and not too few data points
                            if unique_values <= 10 and df.shape[0] > unique_values * 5:
                                plot_configs.append({
                                    'type': 'scatter',
                                    'data': df,
                                    'x': x_col,
                                    'y': y_col,
                                    'hue': hue_col,
                                    'title': f'{x_col} vs. {y_col} by {hue_col}',
                                    'path': os.path.join(output_dir, 
                                                        f"{x_col.lower().replace(' ', '_')}_{y_col.lower().replace(' ', '_')}_scatter.png")
                                })
                            else:
                                plot_configs.append({
                                    'type': 'scatter',
                                    'data': df,
                                    'x': x_col,
                                    'y': y_col,
                                    'title': f'{x_col} vs. {y_col}',
                                    'path': os.path.join(output_dir, 
                                                        f"{x_col.lower().replace(' ', '_')}_{y_col.lower().replace(' ', '_')}_scatter.png")
                                })
                        else:
                            plot_configs.append({
                                'type': 'scatter',
                                'data': df,
                                'x': x_col,
                                'y': y_col,
                                'title': f'{x_col} vs. {y_col}',
                                'path': os.path.join(output_dir, 
                                                    f"{x_col.lower().replace(' ', '_')}_{y_col.lower().replace(' ', '_')}_scatter.png")
                            })
            
            # 4. Box plots of numeric columns by categorical columns
            if numeric_cols and categorical_cols:
                for num_col in numeric_cols:
                    for cat_col in categorical_cols:
                        # Check if categorical column doesn't have too many unique values
                        if df[cat_col].nunique() <= 10:
                            plot_configs.append({
                                'type': 'boxplot',
                                'data': df,
                                'x': cat_col,
                                'y': num_col,
                                'title': f'Distribution of {num_col} by {cat_col}',
                                'path': os.path.join(output_dir, 
                                                   f"{num_col.lower().replace(' ', '_')}_{cat_col.lower().replace(' ', '_')}_boxplot.png")
                            })
            
            # --- OPTIMIZATION: Batch process all plots ---
            # This reduces matplotlib figure creation/destruction overhead
            for config in plot_configs:
                # Skip if already in cache
                if config['path'] in plot_cache:
                    plot_paths.append(config['path'])
                    continue
                
                # Create the figure
                plt.figure(figsize=(10, 6))
                
                try:
                    # Generate the appropriate plot type
                    if config['type'] == 'histogram':
                        # Use a data-driven approach to determine bin count
                        bin_count = min(max(10, int(len(config['data']) / 10)), 30)
                        sns.histplot(config['data'], kde=True, bins=bin_count)
                        plt.title(f'Distribution of {config["col"]}')
                        plt.xlabel(config["col"])
                        plt.ylabel('Frequency')
                        
                    elif config['type'] == 'countplot':
                        sns.countplot(data=config['data'], x=config['x'], order=config['order'])
                        plt.title(f'Count of {config["col"]}')
                        plt.xlabel(config["col"])
                        plt.ylabel('Count')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                    elif config['type'] == 'scatter':
                        if 'hue' in config:
                            sns.scatterplot(data=config['data'], x=config['x'], y=config['y'], hue=config['hue'])
                            plt.legend(title=config['hue'])
                        else:
                            sns.scatterplot(data=config['data'], x=config['x'], y=config['y'])
                        plt.title(config['title'])
                        plt.xlabel(config['x'])
                        plt.ylabel(config['y'])
                        
                    elif config['type'] == 'boxplot':
                        sns.boxplot(data=config['data'], x=config['x'], y=config['y'])
                        plt.title(config['title'])
                        plt.xlabel(config['x'])
                        plt.ylabel(config['y'])
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                    
                    # Save the plot
                    plt.savefig(config['path'])
                    plot_paths.append(config['path'])
                    plot_cache[config['path']] = True
                    print(f"Saved plot: {config['path']}")
                    
                finally:
                    plt.close()  # Ensure we close the figure
            
            return plot_paths

        except Exception as e:
            error_msg = f"Error generating plots: {e}\n{traceback.format_exc()}"
            print(error_msg)
            # Ensure plot is closed in case of error during saving
            plt.close()
            return [error_msg]
        finally:
            # Reset matplotlib state if necessary
            plt.rcdefaults()

    async def generate_advanced_plots(self, output_dir: str = "plots/advanced") -> List[str]:
        """
        Generates advanced statistical plots including density plots, Q-Q plots, violin plots,
        correlation heatmaps, and pair plots. Uses functions from statistical_analysis module.

        Args:
            output_dir (str): The directory to save the advanced plots in. Defaults to 'plots/advanced'.
            
        Returns:
            List of saved plot file paths
        """
        df = self._current_df # Use the helper's current dataframe
        
        if df is None:
            return ["Error: DataFrame not available in helper."]
            
        try:
            # Call the generate_advanced_plots function from statistical_analysis module
            plot_paths = generate_advanced_plots(df, output_dir)
            return plot_paths
        except Exception as e:
            error_msg = f"Error generating advanced plots: {e}\n{traceback.format_exc()}"
            print(error_msg)
            return [error_msg]
            
    async def perform_advanced_analysis(self, save_report: bool = True, 
                                 report_path: str = "reports/statistical_analysis_report.json") -> Dict[str, Any]:
        """
        Performs advanced statistical analysis on the current DataFrame including:
        - Advanced statistics (skewness, kurtosis, confidence intervals)
        - Group-based statistics for each Mode
        - Statistical significance testing (ANOVA and Tukey's HSD)
        
        Args:
            save_report (bool): Whether to save the analysis report to a JSON file
            report_path (str): Path to save the report if save_report is True
            
        Returns:
            Dictionary containing the statistical analysis report
        """
        df = self._current_df
        
        if df is None:
            return {"error": "DataFrame not available in helper."}
            
        try:
            print("[DEBUG] Starting advanced statistical analysis...")
            print(f"[DEBUG] DataFrame shape: {df.shape}")
            print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
            print(f"[DEBUG] DataFrame data types: {df.dtypes.to_dict()}")
            
            print("[DEBUG] Performing advanced statistical analysis...")
            
            # Generate comprehensive statistical report
            overall_stats = calculate_advanced_statistics(df)
            mode_stats = calculate_mode_statistics(df)
            normality_test_results = perform_anova(df)
            
            # Fix any non-serializable values (convert to string or other serializable types)
            def make_json_serializable(obj):
                if isinstance(obj, bool):
                    return str(obj)  # Convert boolean to string
                elif isinstance(obj, (int, float, str, type(None))):
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                else:
                    return str(obj)  # Convert any other type to string
            
            # Prepare report
            report = {
                'overall_statistics': make_json_serializable(overall_stats),
                'mode_statistics': make_json_serializable(mode_stats),
                'normality_tests': make_json_serializable(normality_test_results)
            }
            
            # Save report if requested
            if save_report:
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"[DEBUG] Advanced statistical report saved to {report_path}")
                
            return report
        except Exception as e:
            error_msg = f"Error performing advanced analysis: {e}\n{traceback.format_exc()}"
            print(error_msg)
            return {"error": error_msg}