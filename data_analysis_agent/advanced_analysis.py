import os
import json
import pandas as pd
import traceback
import numpy as np
from typing import Dict, Any, List, Optional
from pandas_helper import PandasHelper
from llama_index.experimental.query_engine import PandasQueryEngine
from statistical_analysis import generate_statistical_report

# New imports for advanced regression models
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class UtilsManager:
    """Utility class for managing common operations like JSON serialization"""
    
    @staticmethod
    def make_json_serializable(obj) -> Any:
        """
        Convert any non-JSON-serializable values to serializable types
        
        Args:
            obj: The object to convert to JSON serializable format
        
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, (str, int, float, type(None))):
            return obj
        elif isinstance(obj, bool):
            return obj  # Return boolean as-is for proper JSON serialization
        elif isinstance(obj, (list, tuple)):
            return [UtilsManager.make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: UtilsManager.make_json_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)  # Convert any other type to string


class StatisticalAnalyzer:
    """Class for advanced statistical analysis functionality"""
    
    def __init__(self, df: pd.DataFrame, llm=None):
        """
        Initialize the statistical analyzer
        
        Args:
            df: The DataFrame to analyze
            llm: The language model to use for the PandasQueryEngine
        """
        self.df = df
        self.llm = llm
        self.query_engine = PandasQueryEngine(df=df, llm=llm, verbose=False) if llm else None
        self.pandas_helper = PandasHelper(df, self.query_engine) if self.query_engine else None
    
    async def perform_analysis(self, original_path: str, modification_summary: str = None) -> Dict[str, Any]:
        """
        Perform advanced statistical analysis on the data
        
        Args:
            original_path: Path to the original data file
            modification_summary: Summary of data modifications performed
            
        Returns:
            Dictionary containing statistical analysis results
        """
        print("[ADVANCED ANALYSIS] Starting advanced statistical analysis")
        print(f"[ADVANCED ANALYSIS] DataFrame shape: {self.df.shape}")
        print(f"[ADVANCED ANALYSIS] DataFrame columns: {self.df.columns.tolist()}")
        print(f"[ADVANCED ANALYSIS] DataFrame data types: {self.df.dtypes.to_dict()}")
        
        # Create directory for reports if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Define the path for the statistical analysis report
        statistical_report_path = "reports/statistical_analysis_report.json"
        
        statistical_report = self._generate_statistical_report(statistical_report_path)
        summary = self._generate_summary(statistical_report)
        plot_info = await self._generate_plots()
        
        # Prepare path for modified file
        path_parts = os.path.splitext(original_path)
        modified_file_path = f"{path_parts[0]}_modified{path_parts[1]}"
        print(f"[ADVANCED ANALYSIS] Modified file path: {modified_file_path}")
        
        print("[ADVANCED ANALYSIS] Advanced statistical analysis completed")
        
        return {
            "statistical_report": statistical_report,
            "summary": summary,
            "plot_info": plot_info,
            "statistical_report_path": statistical_report_path,
            "modified_data_path": modified_file_path
        }
    
    def _generate_statistical_report(self, report_path: str) -> Dict[str, Any]:
        """
        Generate and save the statistical report
        
        Args:
            report_path: Path to save the report
            
        Returns:
            Dictionary containing the statistical report
        """
        try:
            from statistical_analysis import (
                calculate_advanced_statistics,
                calculate_mode_statistics,
                perform_anova,
                perform_tukey_hsd
            )
            
            # Generate comprehensive statistical report components
            advanced_statistics = calculate_advanced_statistics(self.df)
            
            # Fix: Specify 'Mode' as the group_column parameter
            group_statistics = {"Mode": calculate_mode_statistics(self.df, group_column='Mode')} if 'Mode' in self.df.columns else {}
            
            significance_tests = {}
            if 'Mode' in self.df.columns:
                for col in ['Time', 'Distance']:
                    if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                        anova_result = perform_anova(self.df, 'Mode', col)
                        
                        if anova_result.get("is_significant", False):
                            tukey_result = perform_tukey_hsd(self.df, 'Mode', col)
                            significance_tests[col] = tukey_result
                        else:
                            significance_tests[col] = {
                                "anova_result": anova_result,
                                "message": "No significant differences found between modes"
                            }
            
            # Prepare report
            statistical_report = {
                "advanced_statistics": advanced_statistics,
                "group_statistics": group_statistics,
                "significance_tests": significance_tests
            }
            
            # Save report - Make all values JSON serializable
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            json_serializable_report = UtilsManager.make_json_serializable(statistical_report)
            with open(report_path, 'w') as f:
                json.dump(json_serializable_report, f, indent=2)
            print(f"[ADVANCED ANALYSIS] Statistical report saved to {report_path}")
            
            return statistical_report
        
        except Exception as e:
            print(f"[ADVANCED ANALYSIS ERROR] Error in statistical analysis: {e}")
            print(traceback.format_exc())
            return {"error": str(e)}
    
    def _generate_summary(self, statistical_report: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the statistical report
        
        Args:
            statistical_report: The statistical report to summarize
            
        Returns:
            String containing a formatted summary
        """
        summary = "Advanced Statistical Analysis Complete\n\n"
        
        try:
            # Add advanced statistics summary
            if "advanced_statistics" in statistical_report:
                print(f"[ADVANCED ANALYSIS] Processing advanced statistics for {len(statistical_report['advanced_statistics'])} columns")
                summary += "## Advanced Statistics\n\n"
                for column, stats in statistical_report["advanced_statistics"].items():
                    summary += f"### {column}\n"
                    summary += f"- Mean: {stats.get('mean', 'N/A'):.2f}\n"
                    summary += f"- Median: {stats.get('median', 'N/A'):.2f}\n"
                    summary += f"- Standard Deviation: {stats.get('std', 'N/A'):.2f}\n"
                    summary += f"- Skewness: {stats.get('skewness', 'N/A'):.2f}\n"
                    summary += f"- Kurtosis: {stats.get('kurtosis', 'N/A'):.2f}\n"
                    
                    # Add confidence intervals if available
                    if 'ci_95_low' in stats and 'ci_95_high' in stats:
                        summary += f"- 95% Confidence Interval: ({stats['ci_95_low']:.2f}, {stats['ci_95_high']:.2f})\n"
                    
                    # Add normality test results if available
                    if 'is_normal' in stats:
                        is_normal_value = stats['is_normal']
                        # Handle both string and boolean representations
                        if isinstance(is_normal_value, str):
                            is_normal_text = "Normal" if is_normal_value.lower() == "true" else "Not normal"
                        else:
                            is_normal_text = "Normal" if is_normal_value else "Not normal"
                        summary += f"- Normality (Shapiro-Wilk): {is_normal_text}\n"
                    
                    summary += "\n"
            
            # Add significance test results
            if "significance_tests" in statistical_report:
                print(f"[ADVANCED ANALYSIS] Processing significance tests for {len(statistical_report['significance_tests'])} columns")
                summary += "## Significance Tests\n\n"
                for column, test_results in statistical_report["significance_tests"].items():
                    summary += f"### {column}\n"
                    
                    # Add ANOVA results
                    if "anova_result" in test_results:
                        anova = test_results["anova_result"]
                        if "is_significant" in anova:
                            # Handle both string and boolean representations
                            if isinstance(anova["is_significant"], str):
                                is_significant = anova["is_significant"].lower() == "true"
                            else:
                                is_significant = anova["is_significant"]
                            
                            summary += f"- ANOVA: {'Significant differences found' if is_significant else 'No significant differences'}\n"
                            if "f_statistic" in anova and "p_value" in anova:
                                summary += f"  - F-statistic: {anova['f_statistic']:.2f}, p-value: {anova['p_value']:.4f}\n"
                    
                    # Add Tukey HSD results if available
                    if "pairwise_results" in test_results and test_results["pairwise_results"]:
                        print(f"[ADVANCED ANALYSIS] Processing {len(test_results['pairwise_results'])} pairwise comparisons for {column}")
                        summary += "- Tukey HSD Pairwise Comparisons:\n"
                        for pair in test_results["pairwise_results"]:
                            if "group1" in pair and "group2" in pair and "is_significant" in pair:
                                # Handle both string and boolean representations
                                if isinstance(pair["is_significant"], str):
                                    is_sig_pair = pair["is_significant"].lower() == "true"
                                else:
                                    is_sig_pair = pair["is_significant"]
                                    
                                sig_text = "Significant" if is_sig_pair else "Not significant"
                                summary += f"  - {pair['group1']} vs {pair['group2']}: {sig_text}\n"
                                if "mean_difference" in pair:
                                    summary += f"    Mean difference: {pair['mean_difference']:.2f}\n"
                    
                    summary += "\n"
            
            # Add group statistics summary if available
            if "group_statistics" in statistical_report and "Mode" in statistical_report["group_statistics"]:
                print("[ADVANCED ANALYSIS] Processing group statistics by Mode")
                mode_stats = statistical_report["group_statistics"]["Mode"]
                summary += "## Statistics by Mode\n\n"
                
                for column, modes in mode_stats.items():
                    summary += f"### {column} by Mode\n"
                    for mode, stats in modes.items():
                        summary += f"- {mode}:\n"
                        summary += f"  - Mean: {stats.get('mean', 'N/A'):.2f}\n"
                        summary += f"  - Count: {stats.get('count', 'N/A')}\n"
                        if 'ci_95_low' in stats and 'ci_95_high' in stats:
                            summary += f"  - 95% CI: ({stats['ci_95_low']:.2f}, {stats['ci_95_high']:.2f})\n"
                    
                    summary += "\n"
        except Exception as e:
            print(f"[ADVANCED ANALYSIS ERROR] Error generating statistical summary: {e}")
            print(traceback.format_exc())
            summary += f"Error generating statistical summary: {e}\n"
            summary += "Full statistical report saved to JSON file.\n"
        
        print(f"[ADVANCED ANALYSIS] Statistical summary generated with length {len(summary)}")
        return summary
    
    async def _generate_plots(self) -> str:
        """
        Generate advanced visualization plots
        
        Returns:
            String containing information about generated plots
        """
        print("[ADVANCED ANALYSIS] Generating advanced visualizations...")
        os.makedirs("plots/advanced", exist_ok=True)
        
        if self.pandas_helper:
            advanced_plot_paths = await self.pandas_helper.generate_advanced_plots(output_dir="plots/advanced")
            
            if advanced_plot_paths:
                print(f"[ADVANCED ANALYSIS] Generated {len(advanced_plot_paths)} advanced plots")
                plot_info = "Advanced visualizations generated:\n"
                for path in advanced_plot_paths:
                    if isinstance(path, str) and not path.startswith("Error"):
                        plot_info += f"- {path}\n"
                        print(f"[ADVANCED ANALYSIS] Generated plot: {path}")
                
                print(f"[ADVANCED ANALYSIS] Advanced visualization summary: {len(plot_info)} characters")
            else:
                print("[ADVANCED ANALYSIS WARNING] No advanced plots were generated or an error occurred")
                plot_info = "No advanced plots were generated."
        else:
            plot_info = "Unable to generate plots: PandasHelper not initialized."
            
        return plot_info
    
    @staticmethod
    def summarize_findings(statistical_report: Dict[str, Any]) -> str:
        """
        Create a concise summary of key statistical findings
        
        Args:
            statistical_report: Dictionary containing statistical analysis results
            
        Returns:
            String containing a formatted summary of key statistical findings
        """
        print("[ADVANCED ANALYSIS] Creating statistical findings summary")
        summary = "Key Statistical Findings:\n\n"
        
        # Extract and summarize basic statistics
        if "advanced_statistics" in statistical_report:
            for column, stats in statistical_report["advanced_statistics"].items():
                summary += f"- {column}: Mean = {stats.get('mean', 'N/A'):.2f}, "
                summary += f"Median = {stats.get('median', 'N/A'):.2f}, "
                summary += f"Std Dev = {stats.get('std', 'N/A'):.2f}\n"
        
        # Extract and summarize significant findings
        if "significance_tests" in statistical_report:
            significant_findings = []
            for column, test_results in statistical_report["significance_tests"].items():
                if "anova_result" in test_results:
                    is_sig = test_results["anova_result"].get("is_significant", False)
                    # Handle both string and boolean representations
                    if isinstance(is_sig, str):
                        is_sig = is_sig.lower() == "true"
                    
                    if is_sig:
                        significant_findings.append(f"Significant differences found in {column} across transport modes")
                        
                        if "pairwise_results" in test_results:
                            sig_pairs = []
                            for pair in test_results["pairwise_results"]:
                                pair_sig = pair.get("is_significant", False)
                                # Handle both string and boolean representations
                                if isinstance(pair_sig, str):
                                    pair_sig = pair_sig.lower() == "true"
                                    
                                if pair_sig:
                                    sig_pairs.append(f"{pair.get('group1', '')} vs {pair.get('group2', '')}")
                            
                            if sig_pairs:
                                summary += f"- Significant differences in {column} between: {', '.join(sig_pairs)}\n"
        
        # Add any normality findings
        non_normal_vars = []
        if "advanced_statistics" in statistical_report:
            for column, stats in statistical_report["advanced_statistics"].items():
                if "is_normal" in stats:
                    is_normal = stats["is_normal"]
                    # Handle both string and boolean representations
                    if isinstance(is_normal, str):
                        is_normal = is_normal.lower() == "true"
                        
                    if not is_normal:
                        non_normal_vars.append(column)
            
            if non_normal_vars:
                summary += f"- Non-normally distributed variables: {', '.join(non_normal_vars)}\n"
        
        print(f"[ADVANCED ANALYSIS] Statistical findings summary created with length {len(summary)}")
        return summary


class RegressionModeler:
    """Class for implementing and comparing regression models"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def identify_modeling_columns(self) -> Dict[str, str]:
        """
        Automatically identify appropriate target and predictor columns for modeling
        based on dataset characteristics
        
        Returns:
            Dictionary with 'target' and 'predictor' keys containing column names
        """
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least two numeric columns for regression modeling")
            
        # Try to find the most likely target and predictor based on column names and correlations
        potential_target_keywords = ['target', 'output', 'result', 'dependent', 'price', 'cost', 'time', 
                                    'value', 'rating', 'score', 'performance', 'efficiency']
        
        # Find potential target columns by name
        potential_targets = []
        for col in numeric_cols:
            if any(keyword.lower() in col.lower() for keyword in potential_target_keywords):
                potential_targets.append(col)
                
        # If no target found by name, use correlation analysis
        if not potential_targets:
            # Find column with highest average correlation with other columns
            corr_matrix = self.df[numeric_cols].corr().abs()
            avg_corr = corr_matrix.mean()
            potential_targets = [avg_corr.idxmax()]
            
        # Select the column with the highest variance as the target
        if len(potential_targets) > 1:
            variances = self.df[potential_targets].var()
            target_column = variances.idxmax()
        else:
            target_column = potential_targets[0]
            
        # Remove target from numeric columns to find predictor
        remaining_cols = [col for col in numeric_cols if col != target_column]
        
        # Find predictor with highest correlation to target
        if remaining_cols:
            correlations = self.df[remaining_cols].corrwith(self.df[target_column]).abs()
            predictor_column = correlations.idxmax()
        else:
            raise ValueError("No suitable predictor column found")
            
        print(f"[AUTO-DETECT] Selected target column: {target_column}, predictor column: {predictor_column}")
        return {
            "target": target_column,
            "predictor": predictor_column
        }
    
    def fit_polynomial_regression(self, 
                                 target_column: str = None, 
                                 predictor_column: str = None,
                                 degrees: List[int] = [2, 3, 4],
                                 test_size: float = 0.2) -> Dict[str, Any]:
        """
        Fit polynomial regression models of varying degrees
        
        Args:
            target_column: Column to predict (if None, will be auto-detected)
            predictor_column: Column to use as predictor (if None, will be auto-detected)
            degrees: List of polynomial degrees to try
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with results of polynomial regression
        """
        # Auto-detect columns if not provided
        if target_column is None or predictor_column is None:
            cols = self.identify_modeling_columns()
            target_column = cols["target"] if target_column is None else target_column
            predictor_column = cols["predictor"] if predictor_column is None else predictor_column
        
        # Check if columns exist
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        if predictor_column not in self.df.columns:
            raise ValueError(f"Predictor column '{predictor_column}' not found in DataFrame")
        
        # Prepare data
        data = self.df.dropna(subset=[target_column, predictor_column]).copy()
        
        # Split data
        X = data[[predictor_column]]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        results = {}
        
        # Linear model (degree=1) as baseline
        try:
            # Fit linear model using statsmodels for detailed statistics
            X_train_sm = sm.add_constant(X_train)
            model = sm.OLS(y_train, X_train_sm).fit()
            
            # Make predictions
            X_test_sm = sm.add_constant(X_test)
            y_pred = model.predict(X_test_sm)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Add model information
            results[1] = {
                "degree": 1,
                "formula": f"{target_column} = {model.params[0]:.4f} + {model.params[1]:.4f}*{predictor_column}",
                "coefficients": model.params.tolist(),
                "metrics": {
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "aic": model.aic,
                    "bic": model.bic
                },
                "model_type": "polynomial",
                "statsmodels_summary": model.summary().as_text()
            }
            
            print(f"[ADVANCED MODELS] Linear model (degree=1): R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {model.aic:.4f}")
        
        except Exception as e:
            print(f"[ADVANCED MODELS] Error fitting linear model: {str(e)}")
            results[1] = {"error": str(e), "degree": 1}
        
        # Fit polynomial models for each degree
        for degree in degrees:
            try:
                # Create polynomial features
                poly = PolynomialFeatures(degree=degree)
                X_poly_train = poly.fit_transform(X_train)
                X_poly_test = poly.transform(X_test)
                
                # Fit linear regression on polynomial features
                model = LinearRegression()
                model.fit(X_poly_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_poly_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Create formula string
                feature_names = poly.get_feature_names_out(input_features=[predictor_column])
                formula = f"{target_column} = {model.intercept_:.4f}"
                for i, coef in enumerate(model.coef_[1:], 1):  # Skip intercept term
                    formula += f" + {coef:.4f}*{feature_names[i]}"
                
                # Use statsmodels for AIC/BIC
                X_poly_sm = sm.add_constant(X_poly_train[:, 1:])  # Skip the first column (already ones)
                sm_model = sm.OLS(y_train, X_poly_sm).fit()
                
                results[degree] = {
                    "degree": degree,
                    "formula": formula,
                    "coefficients": model.coef_.tolist(),
                    "intercept": float(model.intercept_),
                    "metrics": {
                        "r2": r2,
                        "rmse": rmse,
                        "mae": mae,
                        "aic": sm_model.aic,
                        "bic": sm_model.bic
                    },
                    "model_type": "polynomial",
                    "feature_names": feature_names.tolist()
                }
                
                print(f"[ADVANCED MODELS] Polynomial model (degree={degree}): R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {sm_model.aic:.4f}")
            
            except Exception as e:
                print(f"[ADVANCED MODELS] Error fitting polynomial model with degree {degree}: {str(e)}")
                results[degree] = {"error": str(e), "degree": degree}
        
        # Find best model based on AIC
        valid_models = {k: v for k, v in results.items() if "error" not in v}
        if valid_models:
            best_degree = min(valid_models.items(), 
                              key=lambda x: x[1]['metrics']['aic'] if 'metrics' in x[1] else float('inf'))[0]
            results["best_model"] = {
                "degree": best_degree,
                "selection_criterion": "AIC",
                "metrics": results[best_degree].get("metrics", {})
            }
            print(f"[ADVANCED MODELS] Best polynomial model based on AIC: degree={best_degree}")
        
        # Save plots directory
        os.makedirs("plots/regression", exist_ok=True)
        
        # Plot actual vs predicted for best model
        try:
            if "best_model" in results:
                best_degree = results["best_model"]["degree"]
                
                # Refit model for plotting
                poly = PolynomialFeatures(degree=best_degree)
                X_poly = poly.fit_transform(data[[predictor_column]])
                model = LinearRegression()
                model.fit(X_poly, data[target_column])
                
                # Generate predictions for plotting
                y_pred = model.predict(X_poly)
                
                # Create scatter plot
                plt.figure(figsize=(10, 6))
                plt.scatter(data[predictor_column], data[target_column], alpha=0.5, label='Actual data')
                
                # Sort for line plot
                sorted_indices = np.argsort(data[predictor_column])
                plt.plot(
                    data[predictor_column].iloc[sorted_indices],
                    y_pred[sorted_indices],
                    'r-',
                    linewidth=2,
                    label=f'Polynomial model (degree={best_degree})'
                )
                
                plt.title(f'Polynomial Regression: {predictor_column} vs {target_column}')
                plt.xlabel(predictor_column)
                plt.ylabel(target_column)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = f"plots/regression/polynomial_regression_{predictor_column}_{target_column}.png"
                plt.savefig(plot_path)
                plt.close()
                
                results["plot_path"] = plot_path
                print(f"[ADVANCED MODELS] Saved regression plot to {plot_path}")
        except Exception as e:
            print(f"[ADVANCED MODELS] Error creating regression plot: {str(e)}")
        
        return results
    
    def fit_log_transformation_models(self,
                                    target_column: str = None, 
                                    predictor_column: str = None,
                                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Fit models with various log transformations
        
        Args:
            target_column: Column to predict (if None, will be auto-detected)
            predictor_column: Column to use as predictor (if None, will be auto-detected)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with results of log transformation models
        """
        # Auto-detect columns if not provided
        if target_column is None or predictor_column is None:
            cols = self.identify_modeling_columns()
            target_column = cols["target"] if target_column is None else target_column
            predictor_column = cols["predictor"] if predictor_column is None else predictor_column
        
        # Check if columns exist
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        if predictor_column not in self.df.columns:
            raise ValueError(f"Predictor column '{predictor_column}' not found in DataFrame")
        
        # Prepare data
        data = self.df.dropna(subset=[target_column, predictor_column]).copy()
        
        # Verify data is positive for log transformation
        min_target = data[target_column].min()
        min_predictor = data[predictor_column].min()
        
        # Add small constant if needed to make values positive
        target_offset = abs(min(0, min_target)) + 1 if min_target <= 0 else 0
        predictor_offset = abs(min(0, min_predictor)) + 1 if min_predictor <= 0 else 0
        
        if target_offset > 0:
            print(f"[ADVANCED MODELS] Adding offset of {target_offset} to {target_column} for log transformation")
        if predictor_offset > 0:
            print(f"[ADVANCED MODELS] Adding offset of {predictor_offset} to {predictor_column} for log transformation")
        
        # Create transformations
        data['target'] = data[target_column] + target_offset
        data['predictor'] = data[predictor_column] + predictor_offset
        data['log_target'] = np.log(data['target'])
        data['log_predictor'] = np.log(data['predictor'])
        
        # Define transformation types
        transforms = {
            "linear": {
                "X": data[['predictor']],
                "y": data['target'],
                "name": "Linear",
                "formula": "y = a + b*x"
            },
            "log_x": {
                "X": data[['log_predictor']],
                "y": data['target'],
                "name": "Log-Linear",
                "formula": "y = a + b*log(x)"
            },
            "log_y": {
                "X": data[['predictor']],
                "y": data['log_target'],
                "name": "Linear-Log",
                "formula": "log(y) = a + b*x"
            },
            "log_log": {
                "X": data[['log_predictor']],
                "y": data['log_target'],
                "name": "Log-Log",
                "formula": "log(y) = a + b*log(x)"
            }
        }
        
        results = {}
        
        for transform_name, transform_data in transforms.items():
            try:
                # Split data
                X = transform_data["X"]
                y = transform_data["y"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Fit model
                X_train_sm = sm.add_constant(X_train)
                model = sm.OLS(y_train, X_train_sm).fit()
                
                # Make predictions
                X_test_sm = sm.add_constant(X_test)
                y_pred = model.predict(X_test_sm)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Transform back if needed
                if transform_name in ["log_y", "log_log"]:
                    y_test_orig = np.exp(y_test) - target_offset
                    y_pred_orig = np.exp(y_pred) - target_offset
                    orig_r2 = r2_score(y_test_orig, y_pred_orig)
                    orig_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
                    orig_mae = mean_absolute_error(y_test_orig, y_pred_orig)
                else:
                    y_test_orig = y_test - target_offset
                    y_pred_orig = y_pred - target_offset
                    orig_r2 = r2
                    orig_rmse = rmse
                    orig_mae = mae
                
                # Add model information
                results[transform_name] = {
                    "name": transform_data["name"],
                    "formula": transform_data["formula"],
                    "model_formula": f"{transform_data['formula']} where a={model.params[0]:.4f}, b={model.params[1]:.4f}",
                    "coefficients": model.params.tolist(),
                    "transformed_metrics": {
                        "r2": r2,
                        "rmse": rmse,
                        "mae": mae
                    },
                    "original_scale_metrics": {
                        "r2": orig_r2,
                        "rmse": orig_rmse,
                        "mae": orig_mae
                    },
                    "aic": model.aic,
                    "bic": model.bic,
                    "statsmodels_summary": model.summary().as_text()
                }
                
                print(f"[ADVANCED MODELS] {transform_data['name']} model: R² = {orig_r2:.4f}, RMSE = {orig_rmse:.4f}, AIC = {model.aic:.4f}")
            
            except Exception as e:
                print(f"[ADVANCED MODELS] Error fitting {transform_data['name']} model: {str(e)}")
                results[transform_name] = {"error": str(e), "name": transform_data["name"]}
        
        # Find best model based on original scale R²
        valid_models = {k: v for k, v in results.items() if "error" not in v}
        if valid_models:
            best_transform = max(valid_models.items(), 
                              key=lambda x: x[1]['original_scale_metrics']['r2'])[0]
            results["best_model"] = {
                "transform": best_transform,
                "name": results[best_transform]["name"],
                "selection_criterion": "R² on original scale",
                "metrics": results[best_transform].get("original_scale_metrics", {})
            }
            print(f"[ADVANCED MODELS] Best transformation based on R²: {results[best_transform]['name']}")
        
        # Save plots directory
        os.makedirs("plots/models", exist_ok=True)
        
        # Plot actual vs predicted for all transformations
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot original data points
            plt.scatter(data[predictor_column], data[target_column], alpha=0.5, label='Actual data', color='black', s=20)
            
            colors = ['blue', 'green', 'red', 'purple']
            for i, (transform_name, transform_data) in enumerate(transforms.items()):
                if transform_name in results and "error" not in results[transform_name]:
                    # Get model coefficients
                    coeffs = results[transform_name]["coefficients"]
                    
                    # Create range of x values for plotting
                    x_range = np.linspace(data[predictor_column].min(), data[predictor_column].max(), 100)
                    
                    # Calculate predicted values based on transformation type
                    if transform_name == "linear":
                        x_transform = x_range
                        y_pred = coeffs[0] + coeffs[1] * x_transform
                    elif transform_name == "log_x":
                        x_transform = np.log(x_range + predictor_offset)
                        y_pred = coeffs[0] + coeffs[1] * x_transform
                    elif transform_name == "log_y":
                        x_transform = x_range
                        y_pred = np.exp(coeffs[0] + coeffs[1] * x_transform) - target_offset
                    elif transform_name == "log_log":
                        x_transform = np.log(x_range + predictor_offset)
                        y_pred = np.exp(coeffs[0] + coeffs[1] * x_transform) - target_offset
                    
                    # Plot the model
                    plt.plot(x_range, y_pred, '-', linewidth=2, 
                             label=f"{results[transform_name]['name']} (R²: {results[transform_name]['original_scale_metrics']['r2']:.3f})",
                             color=colors[i % len(colors)])
            
            plt.title(f'Transformation Models: {predictor_column} vs {target_column}')
            plt.xlabel(predictor_column)
            plt.ylabel(target_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = f"plots/models/transformation_models_{predictor_column}_{target_column}.png"
            plt.savefig(plot_path)
            plt.close()
            
            results["plot_path"] = plot_path
            print(f"[ADVANCED MODELS] Saved transformation models plot to {plot_path}")
        except Exception as e:
            print(f"[ADVANCED MODELS] Error creating transformation models plot: {str(e)}")
        
        return results
        
    def run_all_regression_models(self) -> Dict[str, Any]:
        """
        Run all regression modeling techniques and compile results
        
        Returns:
            Dictionary with results of all regression models
        """
        print("[ADVANCED MODELS] Running all regression models")
        
        try:
            # Auto-detect modeling columns
            cols = self.identify_modeling_columns()
            target_column = cols["target"]
            predictor_column = cols["predictor"]
            
            # Run all model types
            polynomial_results = self.fit_polynomial_regression(target_column, predictor_column)
            transformation_results = self.fit_log_transformation_models(target_column, predictor_column)
            
            # Compare best models from each approach
            best_models = {}
            
            if "best_model" in polynomial_results:
                poly_degree = polynomial_results["best_model"]["degree"]
                poly_metrics = polynomial_results[poly_degree]["metrics"]
                best_models["polynomial"] = {
                    "type": "polynomial",
                    "degree": poly_degree,
                    "metrics": poly_metrics,
                    "formula": polynomial_results[poly_degree]["formula"]
                }
            
            if "best_model" in transformation_results:
                trans_type = transformation_results["best_model"]["transform"]
                trans_metrics = transformation_results[trans_type]["original_scale_metrics"]
                best_models["transformation"] = {
                    "type": "transformation",
                    "transform": trans_type,
                    "name": transformation_results[trans_type]["name"],
                    "metrics": trans_metrics,
                    "formula": transformation_results[trans_type]["model_formula"]
                }
            
            # Determine overall best model based on R²
            if best_models:
                r2_values = {}
                if "polynomial" in best_models:
                    r2_values["polynomial"] = best_models["polynomial"]["metrics"]["r2"]
                if "transformation" in best_models:
                    r2_values["transformation"] = best_models["transformation"]["metrics"]["r2"]
                
                overall_best = max(r2_values.items(), key=lambda x: x[1])[0]
                
                # Save overall best model details
                best_models["overall_best"] = {
                    "model_type": overall_best,
                    "details": best_models[overall_best],
                    "selection_criterion": "R²"
                }
                
                print(f"[ADVANCED MODELS] Overall best model: {overall_best} with R² = {r2_values[overall_best]:.4f}")
            
            # Compile all results
            all_results = {
                "target_column": target_column,
                "predictor_column": predictor_column,
                "polynomial_models": polynomial_results,
                "transformation_models": transformation_results,
                "best_models": best_models,
                "plot_paths": {
                    "polynomial": polynomial_results.get("plot_path"),
                    "transformation": transformation_results.get("plot_path")
                }
            }
            
            # Save results as JSON
            os.makedirs("reports", exist_ok=True)
            report_path = "reports/advanced_models.json"
            
            # Make JSON serializable
            json_serializable = UtilsManager.make_json_serializable(all_results)
            with open(report_path, 'w') as f:
                json.dump(json_serializable, f, indent=2)
            
            print(f"[ADVANCED MODELS] Saved regression model results to {report_path}")
            all_results["report_path"] = report_path
            
            return all_results
            
        except Exception as e:
            print(f"[ADVANCED MODELS] Error running regression models: {str(e)}")
            print(traceback.format_exc())
            return {"error": str(e)}


class ModelingManager:
    """Manager class for orchestrating model creation, comparison, and evaluation"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the modeling manager
        
        Args:
            df: The DataFrame to use for modeling
        """
        self.df = df
        self.modeler = RegressionModeler(df)
    
    def perform_advanced_modeling(self,
                                target_column: str = None, 
                                predictor_column: str = None,
                                save_report: bool = True,
                                report_path: str = "reports/advanced_models.json",
                                generate_plots: bool = True,
                                plots_dir: str = "plots/models") -> Dict[str, Any]:
        """
        Implement and evaluate alternative regression models including polynomial and log transformations
        
        Args:
            target_column: The target variable (dependent variable)
            predictor_column: The predictor variable (independent variable)
            save_report: Whether to save the report to a file
            report_path: Path to save the report
            generate_plots: Whether to generate model visualization plots
            plots_dir: Directory to save plots in
            
        Returns:
            Dictionary with advanced modeling results
        """
        print(f"[ADVANCED MODELS] Starting advanced modeling analysis")
        
        try:
            # 1. Run polynomial regression with degrees 2-4
            polynomial_results = self.modeler.fit_polynomial_regression(
                target_column=target_column,
                predictor_column=predictor_column,
                degrees=[2, 3, 4]
            )
            
            # 2. Run log transformation models
            log_results = self.modeler.fit_log_transformation_models(
                target_column=target_column,
                predictor_column=predictor_column
            )
            
            # 3. Compare models and generate visualizations
            comparison = self.modeler.compare_models(
                polynomial_results=polynomial_results,
                log_results=log_results,
                target_column=target_column,
                predictor_column=predictor_column,
                output_dir=plots_dir
            ) if generate_plots else self._get_basic_comparison(polynomial_results, log_results)
            
            # 4. Compile overall results
            overall_results = {
                "polynomial_models": polynomial_results,
                "log_transformation_models": log_results,
                "model_comparison": comparison,
                "status": "success"
            }
            
            # 5. Save results if requested
            if save_report:
                self._save_modeling_report(overall_results, report_path)
                overall_results['report_path'] = report_path
            
            print(f"[ADVANCED MODELS] Advanced modeling analysis completed successfully")
            return overall_results
        
        except Exception as e:
            print(f"[ADVANCED MODELS ERROR] Error in advanced modeling: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _get_basic_comparison(self, polynomial_results: Dict[str, Any], log_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a basic comparison of models without generating plots
        
        Args:
            polynomial_results: Results from polynomial regression
            log_results: Results from log transformation models
            
        Returns:
            Dictionary with basic comparison information
        """
        return {
            "linear_model": {
                "name": "Linear Regression",
                "metrics": polynomial_results.get(1, {}).get('metrics', {})
            },
            "best_polynomial": {
                "name": f"Polynomial (degree={polynomial_results.get('best_model', {}).get('degree')})",
                "metrics": polynomial_results.get(polynomial_results.get('best_model', {}).get('degree', 1), {}).get('metrics', {})
            },
            "best_log_transform": {
                "name": f"Log Transformation ({log_results.get('best_model', {}).get('transformation')})",
                "metrics": log_results.get(log_results.get('best_model', {}).get('transformation', ''), {}).get('metrics', {})
            }
        }
    
    def _save_modeling_report(self, results: Dict[str, Any], report_path: str) -> None:
        """
        Save modeling results to a JSON file
        
        Args:
            results: The modeling results to save
            report_path: Path to save the report to
        """
        # Make values JSON serializable
        serializable_results = UtilsManager.make_json_serializable(results)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Write to file
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"[ADVANCED MODELS] Advanced modeling results saved to {report_path}")


# Function interfaces to maintain backward compatibility

# Utility function for JSON serialization
def make_json_serializable(obj):
    """Backward compatibility function for make_json_serializable"""
    return UtilsManager.make_json_serializable(obj)

# Statistical analysis functions
async def perform_advanced_analysis(df: pd.DataFrame, llm, original_path: str, modification_summary: str = None) -> Dict[str, Any]:
    """Backward compatibility function for perform_advanced_analysis"""
    analyzer = StatisticalAnalyzer(df, llm)
    return await analyzer.perform_analysis(original_path, modification_summary)

async def summarize_statistical_findings(statistical_report: Dict[str, Any]) -> str:
    """Backward compatibility function for summarize_statistical_findings"""
    return StatisticalAnalyzer.summarize_findings(statistical_report)

# Regression modeling functions
def fit_polynomial_regression(df: pd.DataFrame, target_column: str = None, predictor_column: str = None,
                           degrees: List[int] = [2, 3, 4], test_size: float = 0.2) -> Dict[str, Any]:
    """Backward compatibility function for fit_polynomial_regression"""
    modeler = RegressionModeler(df)
    return modeler.fit_polynomial_regression(target_column, predictor_column, degrees, test_size)

def fit_log_transformation_models(df: pd.DataFrame, target_column: str = None, predictor_column: str = None,
                                test_size: float = 0.2) -> Dict[str, Any]:
    """Backward compatibility function for fit_log_transformation_models"""
    modeler = RegressionModeler(df)
    return modeler.fit_log_transformation_models(target_column, predictor_column, test_size)

def compare_alternative_models(df: pd.DataFrame, polynomial_results: Dict[str, Any], log_results: Dict[str, Any],
                            target_column: str = None, predictor_column: str = None,
                            output_dir: str = "plots/models") -> Dict[str, Any]:
    """Backward compatibility function for compare_alternative_models"""
    modeler = RegressionModeler(df)
    return modeler.compare_models(polynomial_results, log_results, target_column, predictor_column, output_dir)

def perform_advanced_modeling(df: pd.DataFrame, target_column: str = None, predictor_column: str = None,
                           save_report: bool = True, report_path: str = "reports/advanced_models.json",
                           generate_plots: bool = True, plots_dir: str = "plots/models") -> Dict[str, Any]:
    """Backward compatibility function for perform_advanced_modeling"""
    manager = ModelingManager(df)
    return manager.perform_advanced_modeling(target_column, predictor_column, save_report, report_path, generate_plots, plots_dir)