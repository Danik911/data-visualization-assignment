"""
Regression Utilities

This module contains helper functions for regression analysis workflow steps.
"""

import pandas as pd
import numpy as np
from llama_index.core.workflow import Context
from typing import Dict, Any, Tuple, Optional

from regression_analysis import perform_regression_analysis, RegressionModel
from model_validation import validate_regression_model
from predictive_application import generate_prediction_examples
from advanced_analysis import perform_advanced_modeling

class RegressionUtils:
    """Helper class for regression analysis operations."""
    
    @staticmethod
    async def check_regression_viability(df: pd.DataFrame) -> Tuple[bool, str, str]:
        """
        Check if regression analysis is viable for the given DataFrame.
        
        Args:
            df: The DataFrame to analyze
            
        Returns:
            Tuple of (is_viable, regression_summary, model_quality)
        """
        # Check if dataset has only one column
        if len(df.columns) == 1:
            print("[REGRESSION] Dataset has only one column. Skipping regression analysis.")
            
            # Create a simple summary report for single-column dataset
            regression_summary = "## Regression Analysis Summary\n\n"
            regression_summary += "The dataset contains only one column, making regression analysis impossible.\n"
            regression_summary += "Regression requires at least two columns: one for the predictor and one for the target.\n\n"
            regression_summary += "### Recommendations\n"
            regression_summary += "- Consider enriching the dataset with additional features\n"
            regression_summary += "- Verify that the data was loaded correctly\n"
            
            # Create a minimal result to continue the workflow
            model_quality = "N/A"
            
            return False, regression_summary, model_quality
        
        # Check for numeric columns
        if df.select_dtypes(include=[np.number]).columns.size < 2:
            print("[REGRESSION] Insufficient numeric columns for regression. Skipping analysis.")
            regression_summary = "## Regression Analysis Summary\n\n"
            regression_summary += "The dataset doesn't contain enough numeric columns for regression analysis.\n"
            regression_summary += "Regression requires at least two numeric columns.\n"
            
            model_quality = "N/A"
            
            return False, regression_summary, model_quality
        
        return True, "", ""
    
    @staticmethod
    async def identify_target_predictor(ctx: Context, df: pd.DataFrame) -> Tuple[str, str]:
        """
        Identify appropriate target and predictor columns for regression.
        
        Args:
            ctx: The workflow context
            df: The DataFrame
            
        Returns:
            Tuple of (target_column, predictor_column)
        """
        # Get the dataset analysis to determine appropriate columns
        dataset_analysis = await ctx.get("dataset_analysis", {})
        potential_targets = dataset_analysis.get("potential_targets", {})
        
        # Get recommended target column from dataset analysis or use fallback
        target_column = potential_targets.get("recommended_target")
        
        # Try to find suitable predictor/target columns from available columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use last numeric column as target if not specified
        if not target_column:
            target_column = numeric_cols[-1]
        
        # Find a predictor that's not the target
        predictor_column = None
        for col in numeric_cols:
            if col != target_column:
                predictor_column = col
                break
        
        print(f"[REGRESSION] Selected columns: target={target_column}, predictor={predictor_column}")
        return target_column, predictor_column
    
    @staticmethod
    async def perform_complete_regression_analysis(
            ctx: Context, 
            df: pd.DataFrame, 
            target_column: str, 
            predictor_column: str
        ) -> Dict[str, Any]:
        """
        Perform complete regression analysis including model validation and advanced modeling.
        
        Args:
            ctx: The workflow context
            df: The DataFrame
            target_column: Target variable column name
            predictor_column: Predictor variable column name
            
        Returns:
            Dictionary with regression analysis results
        """
        results = {}
        
        # 1. Perform basic regression analysis
        print("[REGRESSION] Performing regression analysis")
        regression_results = perform_regression_analysis(
            df=df,
            target_column=target_column, 
            predictor_column=predictor_column,
            save_report=True,
            report_path="reports/regression_models.json",
            generate_plots=True,
            plots_dir="plots/regression"
        )
        
        # Store regression model in context for later use
        regression_model = RegressionModel(df, target_column, predictor_column)
        regression_model.fit_full_dataset_model()
        await ctx.set("regression_model", regression_model)
        
        # Store regression results in context
        await ctx.set("regression_results", regression_results)
        results["regression_results"] = regression_results
        
        # 2. Perform Model Validation
        print("[REGRESSION] Validating regression models")
        full_model = regression_model.models.get('full_dataset')
        if full_model:
            X = df[[predictor_column]]
            y = df[target_column]
            
            validation_results = validate_regression_model(
                model=full_model,
                X=X,
                y=y,
                model_name=f"{target_column}-{predictor_column} Regression",
                feature_names=[predictor_column],
                save_report=True,
                report_path="reports/model_validation.json",
                generate_plots=True,
                plots_dir="plots/validation"
            )
            
            # Store validation results
            await ctx.set("validation_results", validation_results)
            results["validation_results"] = validation_results
            
            # Check if model meets assumptions
            assumptions_met = validation_results.get('assumptions_met', {})
            model_quality = "High" if all(assumptions_met.values()) else "Medium" if any(assumptions_met.values()) else "Low"
            await ctx.set("model_quality", model_quality)
            results["model_quality"] = model_quality
        else:
            validation_results = {"status": "error", "error_message": "Full dataset model not available"}
            model_quality = "Unknown"
            await ctx.set("model_quality", model_quality)
            await ctx.set("validation_results", validation_results)
            results["validation_results"] = validation_results
            results["model_quality"] = model_quality
        
        # 3. Perform Advanced Modeling
        print("[REGRESSION] Running advanced modeling analysis")
        advanced_modeling_results = perform_advanced_modeling(
            df=df,
            target_column=target_column,
            predictor_column=predictor_column,
            save_report=True,
            report_path="reports/advanced_models.json",
            generate_plots=True,
            plots_dir="plots/models"
        )
        
        # Store advanced modeling results
        await ctx.set("advanced_modeling_results", advanced_modeling_results)
        results["advanced_modeling_results"] = advanced_modeling_results
        
        # 4. Generate Prediction Examples
        print("[REGRESSION] Generating prediction examples")
        prediction_results = generate_prediction_examples(
            regression_model=regression_model,
            save_report=True,
            report_path="reports/prediction_results.json",
            generate_plots=True,
            plots_dir="plots/predictions"
        )
        
        # Store prediction results
        await ctx.set("prediction_results", prediction_results)
        results["prediction_results"] = prediction_results
        
        return results
    
    @staticmethod
    def generate_regression_summary(
            regression_results: Dict[str, Any],
            validation_results: Dict[str, Any],
            advanced_modeling_results: Dict[str, Any],
            prediction_results: Dict[str, Any],
            target_column: str,
            predictor_column: str
        ) -> str:
        """
        Generate a comprehensive regression summary report.
        
        Args:
            regression_results: Results from regression analysis
            validation_results: Results from model validation
            advanced_modeling_results: Results from advanced modeling
            prediction_results: Results from prediction examples
            target_column: Target variable column name
            predictor_column: Predictor variable column name
            
        Returns:
            Formatted regression summary string
        """
        regression_summary = "## Regression Analysis Summary\n\n"
        
        # Add linear model summary
        if regression_results.get('status') == 'success':
            full_model_info = regression_results.get('full_model', {})
            regression_summary += "### Linear Model\n"
            regression_summary += f"- Formula: {full_model_info.get('formula', 'N/A')}\n"
            metrics = full_model_info.get('metrics', {})
            regression_summary += f"- R-squared: {metrics.get('r_squared', 'N/A'):.4f}\n"
            regression_summary += f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}\n"
            
            # Add model validation summary
            if validation_results.get('status') == 'success':
                regression_summary += "\n### Model Validation\n"
                assumptions = validation_results.get('assumptions_met', {})
                regression_summary += f"- Normality of residuals: {'✅ Met' if assumptions.get('normality', False) else '❌ Not met'}\n"
                regression_summary += f"- Homoscedasticity: {'✅ Met' if assumptions.get('homoscedasticity', False) else '❌ Not met'}\n"
                regression_summary += f"- Zero mean residuals: {'✅ Met' if assumptions.get('zero_mean_residuals', False) else '❌ Not met'}\n"
                
                cv_metrics = validation_results.get('cross_validation', {}).get('metrics', {})
                regression_summary += f"- Cross-validation R²: {cv_metrics.get('r2_mean', 'N/A'):.4f} (±{cv_metrics.get('r2_std', 'N/A'):.4f})\n"
            
            # Add advanced modeling summary
            if advanced_modeling_results.get('status') == 'success':
                comparison = advanced_modeling_results.get('model_comparison', {})
                best_model_key = comparison.get('overall_best_model')
                
                regression_summary += "\n### Alternative Models\n"
                
                if best_model_key and best_model_key in comparison:
                    best_model = comparison.get(best_model_key, {})
                    regression_summary += f"- Best model: {best_model.get('name', 'Unknown')}\n"
                    
                    best_metrics = best_model.get('metrics', {})
                    if best_metrics:
                        regression_summary += f"- AIC: {best_metrics.get('aic', 'N/A'):.2f}\n"
                        regression_summary += f"- BIC: {best_metrics.get('bic', 'N/A'):.2f}\n"
                        regression_summary += f"- R-squared: {best_metrics.get('r2', 'N/A'):.4f}\n"
            
            # Add prediction example summary
            if prediction_results.get('status') == 'success':
                regression_summary += "\n### Predictions\n"
                regression_summary += f"- Prediction examples generated for {target_column}-{predictor_column} model\n"
                regression_summary += "- See prediction plots in plots/predictions/ directory\n"
        else:
            regression_summary += "Regression analysis could not be completed successfully.\n"
        
        return regression_summary