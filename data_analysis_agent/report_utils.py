"""
Report Utilities

This module contains utility functions for report generation, verification, and regeneration.
"""

import os
import json
import traceback
import pandas as pd
from llama_index.core.workflow import Context
from typing import Dict, Any, Optional, List

from data_quality import assess_data_quality, clean_data
from statistical_analysis import generate_statistical_report
from regression_analysis import perform_regression_analysis
from advanced_analysis import perform_advanced_modeling

class ReportUtils:
    """Utilities for report verification and regeneration."""
    
    @staticmethod
    async def verify_reports(reports_to_verify: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Verify if reports exist and are valid JSON files.
        
        Args:
            reports_to_verify: List of report paths to verify
            
        Returns:
            Dictionary with report status information
        """
        reports_status = {}
        
        for report_path in reports_to_verify:
            print(f"Verifying report: {report_path}")
            status = {"exists": False, "complete": False, "error": None}
            
            try:
                # Check if report exists
                if os.path.exists(report_path):
                    status["exists"] = True
                    
                    # Read report and check if it's a valid JSON
                    with open(report_path, 'r') as f:
                        try:
                            report_data = json.load(f)
                            
                            # Check if the report has content
                            if report_data and isinstance(report_data, dict):
                                status["complete"] = True
                            else:
                                status["error"] = "Report exists but contains no valid data"
                                print(f"Error: {report_path} exists but contains no valid data")
                        
                        except json.JSONDecodeError:
                            status["error"] = "Invalid JSON format"
                            print(f"Error: {report_path} is not a valid JSON file")
                else:
                    status["error"] = "Report file does not exist"
                    print(f"Error: {report_path} does not exist")
                
            except Exception as e:
                status["error"] = str(e)
                print(f"Error verifying report {report_path}: {e}")
            
            reports_status[report_path] = status
        
        return reports_status
    
    @staticmethod
    async def regenerate_report(ctx: Context, report_path: str) -> None:
        """
        Attempt to regenerate a missing or incomplete report.
        
        Args:
            ctx: The workflow context
            report_path: Path to the report to regenerate
        """
        print(f"Attempting to regenerate report: {report_path}")
        
        try:
            df = await ctx.get("dataframe")
            report_name = os.path.basename(report_path)
            
            if "data_quality_report" in report_path:
                # Regenerate data quality report
                print("Regenerating data quality report...")
                assess_data_quality(df, save_report=True, report_path=report_path)
                
            elif "cleaning_report" in report_path:
                # Regenerate cleaning report
                print("Regenerating cleaning report...")
                assessment_report = await ctx.get("assessment_report", None)
                if assessment_report:
                    clean_data(
                        df=df,
                        assessment_report=assessment_report,
                        save_report=True,
                        report_path=report_path,
                        generate_plots=False  # Skip plots during regeneration
                    )
                
            elif "regression_models" in report_path:
                await ReportUtils._regenerate_regression_report(ctx, df, report_path)
                
            elif "statistical_analysis_report" in report_path:
                # Regenerate statistical analysis report
                print("Regenerating statistical analysis report...")
                generate_statistical_report(
                    df=df,
                    save_report=True,
                    report_path=report_path
                )
                
            elif "advanced_models" in report_path:
                await ReportUtils._regenerate_advanced_models_report(ctx, df, report_path)
                
            print(f"Successfully regenerated report: {report_path}")
            
        except Exception as e:
            print(f"Failed to regenerate report {report_path}: {str(e)}")
            traceback.print_exc()
    
    @staticmethod
    async def _regenerate_regression_report(ctx: Context, df: pd.DataFrame, report_path: str) -> None:
        """
        Regenerate regression models report.
        
        Args:
            ctx: The workflow context
            df: The DataFrame
            report_path: Path to the report to regenerate
        """
        print("Regenerating regression models report...")
        regression_model = await ctx.get("regression_model", None)
        
        if regression_model:
            regression_model.save_model_results(file_path=report_path)
        else:
            # Try to rebuild the model if not in context
            target_column, predictor_column = await ReportUtils._get_target_predictor_columns(ctx, df)
            
            print(f"[REGENERATION] Using target={target_column}, predictor={predictor_column}")
            perform_regression_analysis(
                df=df,
                target_column=target_column,
                predictor_column=predictor_column,
                save_report=True,
                report_path=report_path,
                generate_plots=False  # Skip plots during regeneration
            )
    
    @staticmethod
    async def _regenerate_advanced_models_report(ctx: Context, df: pd.DataFrame, report_path: str) -> None:
        """
        Regenerate advanced models report.
        
        Args:
            ctx: The workflow context
            df: The DataFrame
            report_path: Path to the report to regenerate
        """
        print("Regenerating advanced models report...")
        target_column, predictor_column = await ReportUtils._get_target_predictor_columns(ctx, df)
        
        print(f"[REGENERATION] Using target={target_column}, predictor={predictor_column}")
        perform_advanced_modeling(
            df=df,
            target_column=target_column,
            predictor_column=predictor_column,
            save_report=True,
            report_path=report_path,
            generate_plots=False  # Skip plots during regeneration
        )
    
    @staticmethod
    async def _get_target_predictor_columns(ctx: Context, df: pd.DataFrame) -> tuple:
        """
        Get target and predictor columns from context or dataset.
        
        Args:
            ctx: The workflow context
            df: The DataFrame
            
        Returns:
            Tuple of (target_column, predictor_column)
        """
        target_column = None
        predictor_column = None
        
        # Try to get columns from regression model first
        regression_model = await ctx.get("regression_model", None)
        if regression_model:
            target_column = regression_model.target_column
            predictor_column = regression_model.predictor_column
            return target_column, predictor_column
        
        # Try to get from dataset analysis
        dataset_analysis = await ctx.get("dataset_analysis", {})
        potential_targets = dataset_analysis.get("potential_targets", {})
        
        # Get target column
        target_column = potential_targets.get("recommended_target")
        if not target_column and 'Sale_Price' in df.columns:
            target_column = 'Sale_Price'
        elif not target_column:
            target_column = df.columns[-1]
        
        # Get predictor column
        recommended_predictors = potential_targets.get("recommended_predictors", [])
        if recommended_predictors:
            predictor_column = recommended_predictors[0]
        else:
            # Find any numeric column that's not the target
            numeric_cols = df.select_dtypes(include=[pd.np.number]).columns.tolist()
            for col in numeric_cols:
                if col != target_column:
                    predictor_column = col
                    break
        
        return target_column, predictor_column
    
    @staticmethod
    def generate_report_status_summary(reports_status: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a formatted summary of report status.
        
        Args:
            reports_status: Dictionary with report status information
            
        Returns:
            Formatted report status summary string
        """
        report_status_summary = "\n\n## Reports Status\n\n"
        
        for report_path, status in reports_status.items():
            report_name = os.path.basename(report_path)
            if status["complete"]:
                report_status_summary += f"- ✅ {report_name}: Successfully generated\n"
            else:
                error = status["error"] or "Unknown error"
                report_status_summary += f"- ⚠️ {report_name}: Issue detected - {error}\n"
        
        return report_status_summary