"""
Workflow Setup Helper

This module contains helper classes and functions for setting up the data analysis workflow,
including data loading, dataset analysis, and quality assessment.
"""

import os
import pandas as pd
import numpy as np
import traceback
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.workflow import Context
from typing import Dict, Any, Tuple, List, Optional

from dataset_analyzer import analyze_dataset
from data_quality import assess_data_quality
from config import get_config

class WorkflowSetup:
    """Helper class for workflow setup operations."""
    
    @staticmethod
    async def load_and_analyze_data(ctx: Context, dataset_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from CSV and perform initial analysis.
        
        Args:
            ctx: The workflow context
            dataset_path: Path to the dataset
            
        Returns:
            Tuple of (DataFrame, analysis results dictionary)
        """
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Create Pandas Query Engine
        from agents import llm
        query_engine = PandasQueryEngine(df=df, llm=llm, verbose=True)
        
        # Store in context
        await ctx.set("dataframe", df)
        await ctx.set("query_engine", query_engine)
        await ctx.set("original_path", dataset_path)
        
        print(f"Successfully loaded {dataset_path} and created PandasQueryEngine.")
        
        # Analyze dataset structure and content
        print("[SETUP] Analyzing dataset to make processing dataset-agnostic")
        analysis_results = analyze_dataset(
            df=df,
            save_report=True,
            report_path='reports/dataset_analysis.json',
            generate_plots=True,
            plots_dir='plots/dataset_analysis'
        )
        
        # Store analysis results
        await ctx.set("dataset_analysis", analysis_results)
        
        return df, analysis_results
    
    @staticmethod
    async def setup_configuration(ctx: Context, analysis_results: Dict[str, Any]) -> None:
        """
        Set up configuration based on dataset properties.
        
        Args:
            ctx: The workflow context
            analysis_results: Results from dataset analysis
        """
        # Get the domain and detected properties
        detected_domain = analysis_results.get("dataset_domain", {}).get("detected_domain", "generic")
        print(f"[SETUP] Detected dataset domain: {detected_domain}")
        
        # Report target and predictor variables
        recommended_target = None
        if "potential_targets" in analysis_results and "recommended_target" in analysis_results["potential_targets"]:
            recommended_target = analysis_results["potential_targets"]["recommended_target"]
            print(f"[SETUP] Recommended target variable: {recommended_target}")
        
        # Load configuration based on detected dataset type
        config = get_config()
        
        # Update configuration with dataset properties
        config.set("dataset_properties.detected_domain", detected_domain)
        config.set("dataset_properties.recommended_target", recommended_target)
        config.set("dataset_properties.analysis", analysis_results)
        
        # Store configuration in context
        await ctx.set("config", config)
    
    @staticmethod
    async def perform_quality_assessment(ctx: Context, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            ctx: The workflow context
            df: The DataFrame to assess
            
        Returns:
            Assessment report dictionary
        """
        print("Performing comprehensive data quality assessment...")
        assessment_report = assess_data_quality(
            df, 
            save_report=True, 
            report_path='reports/data_quality_report.json'
        )
        
        # Store assessment report in context
        await ctx.set("assessment_report", assessment_report)
        
        quality_score = assessment_report['dataset_info']['quality_score']
        print(f"Data quality assessment completed with quality score: {quality_score}")
        
        return assessment_report
    
    @staticmethod
    def format_quality_summary(assessment_report: Dict[str, Any]) -> str:
        """
        Format quality assessment summary for display.
        
        Args:
            assessment_report: The assessment report dictionary
            
        Returns:
            Formatted quality summary string
        """
        issue_summary = assessment_report['issue_summary']
        recommendations = assessment_report['recommendations']
        
        quality_summary = (
            f"Data Quality Assessment Summary:\n"
            f"- Total rows: {issue_summary['total_rows']}, Total columns: {issue_summary['total_columns']}\n"
            f"- Missing values: {issue_summary['missing_value_count']}\n"
            f"- Duplicate rows: {issue_summary['duplicate_row_count']}\n"
            f"- Outliers detected: {issue_summary['outlier_count']}\n"
            f"- Impossible values: {issue_summary['impossible_value_count']}\n"
            f"- Quality score: {assessment_report['dataset_info']['quality_score']}/100\n\n"
            f"Recommendations:\n"
        )
        
        for category, recs in recommendations.items():
            if recs:
                quality_summary += f"- {category.replace('_', ' ').title()}:\n"
                for rec in recs:
                    quality_summary += f"  * {rec}\n"
        
        return quality_summary
    
    @staticmethod
    async def gather_initial_stats(ctx: Context, df: pd.DataFrame, query_engine) -> Tuple[str, Dict[str, Any]]:
        """
        Gather initial statistics about the dataset.
        
        Args:
            ctx: The workflow context
            df: The DataFrame
            query_engine: The PandasQueryEngine
            
        Returns:
            Tuple of (stats summary string, column info dictionary)
        """
        initial_info_str = "Could not retrieve initial stats."
        column_info_dict = {}
        
        try:
            # Use query engine to get stats
            if hasattr(query_engine, 'aquery'):
                response = await query_engine.aquery(
                    "Show the shape of the dataframe (number of rows and columns) and the output of df.describe(include='all')"
                )
            else:
                response = query_engine.query(
                    "Show the shape of the dataframe (number of rows and columns) and the output of df.describe(include='all')"
                )
            
            initial_info_str = str(response)
            
            # Get column information
            missing_counts = df.isna().sum().to_dict()
            dtypes = df.dtypes.astype(str).to_dict()
            column_info_dict = {"dtypes": dtypes, "missing_counts": missing_counts}
            
            print(f"--- Initial Info Gathered ---\n{initial_info_str[:100]}...\n-----------------------------")
            
            # Store in context
            await ctx.set("stats_summary", initial_info_str)
            await ctx.set("column_info", column_info_dict)
        
        except Exception as e:
            print(f"Warning: Could not query initial info from engine during setup: {e}")
            initial_info_str = f"Columns: {df.columns.tolist()}"
            column_info_dict = {"columns": df.columns.tolist()}
            
            # Store minimal info in context
            await ctx.set("stats_summary", initial_info_str)
            await ctx.set("column_info", column_info_dict)
        
        return initial_info_str, column_info_dict
    
    @staticmethod
    async def initialize_required_reports(ctx: Context) -> List[str]:
        """
        Initialize list of required reports for the workflow.
        
        Args:
            ctx: The workflow context
            
        Returns:
            List of report paths
        """
        required_reports = [
            "reports/dataset_analysis.json",
            "reports/data_quality_report.json",
            "reports/cleaning_report.json",
            "reports/statistical_analysis_report.json",
            "reports/regression_models.json",
            "reports/advanced_models.json"
        ]
        
        await ctx.set("required_reports", required_reports)
        return required_reports