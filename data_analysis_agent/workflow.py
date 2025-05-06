import os
import pandas as pd
import numpy as np
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.workflow import Workflow, Context, StopEvent, step
from llama_index.core.workflow import StartEvent
from llama_index.core.agent import FunctionCallingAgent

# Import events from events.py and new workflow_events.py
from events import *
from workflow_events import (
    InitialAssessmentEvent, DataAnalysisEvent, ModificationRequestEvent, 
    ModificationCompleteEvent, RegressionModelingEvent, RegressionCompleteEvent,
    VisualizationRequestEvent, FinalizeReportsEvent
)

# Import setup helper
from workflow_setup import WorkflowSetup

# Import report and regression utilities
from report_utils import ReportUtils
from regression_utils import RegressionUtils

# Import existing modules
from agents import create_agents, llm
from data_quality import clean_data
from reporting import generate_report
from consultation import handle_user_consultation
from advanced_analysis import perform_advanced_analysis
from visualization import generate_visualizations


class DataAnalysisFlow(Workflow):
    """
    Main workflow class for the data analysis pipeline.
    Orchestrates the steps from data loading to report generation.
    """
    
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> InitialAssessmentEvent:
        """Initialize the agents and setup the workflow"""

        try:
            # Load data and perform initial analysis
            df, analysis_results = await WorkflowSetup.load_and_analyze_data(ctx, ev.dataset_path)
            
            # Setup configuration based on dataset properties
            await WorkflowSetup.setup_configuration(ctx, analysis_results)
            
            # Initialize agents
            self.data_prep_agent, self.data_analysis_agent = create_agents()
            
            # Perform data quality assessment
            assessment_report = await WorkflowSetup.perform_quality_assessment(ctx, df)
            
            # Format quality assessment summary
            quality_summary = WorkflowSetup.format_quality_summary(assessment_report)
            
            # Get initial statistics
            query_engine = await ctx.get("query_engine")
            initial_info_str, column_info_dict = await WorkflowSetup.gather_initial_stats(ctx, df, query_engine)
            
            # Combine quality assessment with basic stats
            combined_summary = f"{quality_summary}\n\nAdditional Statistics:\n{initial_info_str}"
            
            # Add analysis summary from dataset analyzer
            if "summary" in analysis_results:
                analysis_summary = "\n\nDataset Analysis Summary:\n"
                for point in analysis_results["summary"]:
                    analysis_summary += f"- {point}\n"
                combined_summary += analysis_summary
            
            # Initialize list of required reports
            await WorkflowSetup.initialize_required_reports(ctx)
            
            return InitialAssessmentEvent(
                stats_summary=combined_summary,
                column_info=column_info_dict,
                original_path=ev.dataset_path,
            )
        except Exception as e:
            print(f"Error during setup: Failed to load {ev.dataset_path} or create engine. Error: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Setup failed: {e}")
        
    @step
    async def data_preparation(self, ctx: Context, ev: InitialAssessmentEvent) -> DataAnalysisEvent:
        """Use the data prep agent to suggest cleaning/preparation based on schema and quality assessment."""

        initial_info = ev.stats_summary
        assessment_report = await ctx.get("assessment_report", None)

        # Enhanced prompt with quality assessment insights
        prep_prompt = (
            f"The dataset (from {ev.original_path}) has been analyzed with our enhanced data quality assessment tool. Here's the comprehensive summary:\n\n{initial_info}\n\n"
            f"Based on these statistics and quality assessment, describe the necessary data preparation steps. "
            f"Pay special attention to the recommendations from our data quality assessment tool, which has already identified issues using Tukey's method for outliers, systematic data type verification, and uniqueness verification. "
            f"For each issue category (missing values, outliers, duplicates, impossible values, data types), suggest specific actions with statistical justification. "
            f"Focus on describing *what* needs to be done and *why* based on the provided assessment and stats. If the assessment shows a high quality score with minimal issues, acknowledge that minimal cleaning is needed."
        )
        result = self.data_prep_agent.chat(prep_prompt)

        # Extract prepared data description from agent response
        prepared_data_description = None
        if hasattr(result, 'response'):
            prepared_data_description = result.response
            if not prepared_data_description:
                prepared_data_description = "Agent returned an empty description despite the prompt."
                print("Warning: Agent response attribute was empty.")
        else:
            prepared_data_description = "Could not extract data preparation description from agent response."
            print(f"Warning: Agent response does not have expected 'response' attribute. Full result: {result}")

        print(f"--- Prep Agent Description Output ---\n{prepared_data_description[:100]}...\n------------------------------------")

        # Store the agent's suggested description
        await ctx.set("agent_prepared_data_description", prepared_data_description)

        return DataAnalysisEvent(
            prepared_data_description=prepared_data_description,
            original_path=ev.original_path
        )

    @step
    async def human_consultation(self, ctx: Context, ev: DataAnalysisEvent) -> ModificationRequestEvent:
        """Analyzes initial assessment, asks user for cleaning decisions using numbered options."""
        print("--- Running Human Consultation Step ---")
        agent_suggestion = ev.prepared_data_description
        original_path = ev.original_path
        stats_summary = await ctx.get("stats_summary", "Stats not available.")
        column_info = await ctx.get("column_info", {})
        
        # Use the refactored consultation module
        consultation_result = await handle_user_consultation(
            ctx, 
            llm,
            agent_suggestion,
            stats_summary,
            column_info,
            original_path
        )
        
        return ModificationRequestEvent(
            user_approved_description=consultation_result["user_approved_description"],
            original_path=original_path
        )

    @step
    async def data_modification(self, ctx: Context, ev: ModificationRequestEvent) -> ModificationCompleteEvent:
        """Applies the data modifications using the DataCleaner class based on user input."""
        print("--- Running Enhanced Data Modification Step ---")
        df = await ctx.get("dataframe")
        assessment_report = await ctx.get("assessment_report")
        original_path = ev.original_path
        
        # Apply data cleaning
        print("Applying data cleaning using DataCleaner with quality assessment report...")
        cleaned_df, cleaning_report = clean_data(
            df=df,
            assessment_report=assessment_report,
            save_report=True,
            report_path='reports/cleaning_report.json',
            generate_plots=True,
            plots_dir='plots/cleaning_comparisons'
        )
        
        print(f"Data cleaning completed with {len(cleaning_report['cleaning_log'])} steps")
        
        # Update the context with cleaned DataFrame
        await ctx.set("dataframe", cleaned_df)
        await ctx.set("cleaning_report", cleaning_report)
        
        # Update the query engine with the cleaned DataFrame
        query_engine = PandasQueryEngine(df=cleaned_df, llm=llm, verbose=True)
        await ctx.set("query_engine", query_engine)
        
        # Generate a summary of the cleaning performed
        cleaning_summary = "Data cleaning was performed with the following steps:\n"
        for i, step in enumerate(cleaning_report['cleaning_log'], 1):
            cleaning_summary += f"{i}. {step['action']}: "
            if step['action'] == 'standardize_mode_values' and 'changes' in step['details']:
                cleaning_summary += f"Standardized {step['details']['changes']} Mode values\n"
            elif step['action'] == 'handle_missing_values':
                cleaning_summary += f"Addressed missing values in {', '.join(step['details']['strategies'].keys())}\n"
            elif step['action'] == 'handle_outliers':
                cleaning_summary += f"Handled outliers in {', '.join(step['details']['columns'])} using {step['details']['method']} method\n"
            elif step['action'] == 'handle_duplicates':
                cleaning_summary += f"Removed {step['details']['duplicates_removed']} duplicate rows\n"
            elif step['action'] == 'handle_impossible_values':
                cleaning_summary += f"Fixed impossible values in {', '.join(step['details']['constraints'].keys())}\n"
            else:
                cleaning_summary += f"Completed\n"
                
        cleaning_summary += "\nBefore/After Metrics:\n"
        metrics = cleaning_report['metrics_comparison']
        cleaning_summary += f"- Rows: {metrics['row_count']['before']} → {metrics['row_count']['after']} ({metrics['row_count']['change']} change)\n"
        cleaning_summary += f"- Missing values: {metrics['missing_values']['before']} → {metrics['missing_values']['after']} ({metrics['missing_values']['change']} change)\n"
        
        if 'numeric_stats' in metrics:
            for col, stats in metrics['numeric_stats'].items():
                if 'mean' in stats:
                    cleaning_summary += f"- {col} mean: {stats['mean']['before']:.2f} → {stats['mean']['after']:.2f}\n"
        
        await ctx.set("modification_summary", cleaning_summary)
        print(f"--- Cleaning Summary ---\n{cleaning_summary[:100]}...\n-------------------------")

        return ModificationCompleteEvent(
            original_path=original_path,
            modification_summary=cleaning_summary
        )

    @step
    async def advanced_statistical_analysis(self, ctx: Context, ev: ModificationCompleteEvent) -> RegressionModelingEvent:
        """Performs advanced statistical analysis on the cleaned data."""
        print("--- Running Advanced Statistical Analysis Step ---")
        df = await ctx.get("dataframe")
        original_path = ev.original_path
        modification_summary = ev.modification_summary
        
        # Perform advanced analysis
        analysis_results = await perform_advanced_analysis(
            df=df,
            llm=llm,
            original_path=original_path,
            modification_summary=modification_summary
        )
        
        # Store results in context
        await ctx.set("statistical_report", analysis_results["statistical_report"])
        await ctx.set("statistical_summary", analysis_results["summary"])
        await ctx.set("advanced_plot_info", analysis_results["plot_info"])
        
        # Continue to regression modeling step
        return RegressionModelingEvent(
            modified_data_path=analysis_results["modified_data_path"],
            statistical_report_path=analysis_results["statistical_report_path"]
        )

    @step
    async def regression_modeling(self, ctx: Context, ev: RegressionModelingEvent) -> RegressionCompleteEvent:
        """Performs regression analysis including linear regression and advanced models."""
        print("--- Running Regression Modeling Step (Phase 3) ---")
        df = await ctx.get("dataframe")
        
        # Check if regression is viable for this dataset
        is_viable, regression_summary, model_quality = await RegressionUtils.check_regression_viability(df)
        
        if not is_viable:
            # Skip regression and return with summary
            await ctx.set("regression_results", {"status": "skipped", "reason": "not_viable"})
            await ctx.set("model_quality", model_quality)
            
            return RegressionCompleteEvent(
                modified_data_path=ev.modified_data_path,
                regression_summary=regression_summary,
                model_quality=model_quality
            )
        
        # Identify target and predictor columns
        target_column, predictor_column = await RegressionUtils.identify_target_predictor(ctx, df)
        
        # Perform complete regression analysis
        regression_analysis = await RegressionUtils.perform_complete_regression_analysis(
            ctx, df, target_column, predictor_column
        )
        
        # Generate regression summary
        regression_summary = RegressionUtils.generate_regression_summary(
            regression_analysis["regression_results"],
            regression_analysis["validation_results"],
            regression_analysis["advanced_modeling_results"],
            regression_analysis["prediction_results"],
            target_column,
            predictor_column
        )
        
        return RegressionCompleteEvent(
            modified_data_path=ev.modified_data_path,
            regression_summary=regression_summary,
            model_quality=regression_analysis["model_quality"]
        )

    @step
    async def analysis_reporting(self, ctx: Context, ev: RegressionCompleteEvent) -> VisualizationRequestEvent:
        """Generates a report based on the analysis results."""
        print("--- Running Analysis & Reporting Step ---")
        df = await ctx.get("dataframe")
        original_path = ev.modified_data_path
        
        # Get the modification summary from context
        modification_summary = await ctx.get("modification_summary", "Modification summary not available.")
        
        # Get the statistical summary from context
        statistical_summary = await ctx.get("statistical_summary", "Statistical summary not available.")
        
        # Add regression summary to the report
        combined_summary = statistical_summary + "\n\n" + ev.regression_summary
        
        # Use the refactored reporting module
        reporting_results = await generate_report(
            df=df,
            llm=llm,
            original_path=original_path,
            modification_summary=modification_summary,
            statistical_summary=combined_summary
        )
        
        # Update context with final dataframe
        await ctx.set("dataframe", reporting_results["final_df"])
        await ctx.set("final_report", reporting_results["final_report"])
        
        # Combine the report with information about the advanced plots and regression quality
        advanced_plot_info = await ctx.get("advanced_plot_info", "Advanced plot information not available.")
        
        # Add model quality information
        model_quality_info = f"\n\n## Model Quality Assessment\n\nRegression Model Quality: {ev.model_quality}\n"
        
        enhanced_report = reporting_results["final_report"] + "\n\n## Advanced Visualizations\n\n" + advanced_plot_info + model_quality_info
        
        return VisualizationRequestEvent(
            modified_data_path=reporting_results["modified_file_path"],
            report=enhanced_report
        )

    @step
    async def create_visualizations(self, ctx: Context, ev: VisualizationRequestEvent) -> FinalizeReportsEvent:
        """Generates standard and advanced visualizations for the cleaned data."""
        print("--- Running Enhanced Visualization Step ---")
        df = await ctx.get("dataframe")
        modified_data_path = ev.modified_data_path
        final_report = ev.report
        
        if df is None:
            print("Error: DataFrame not found in context for visualization.")
            # Return with error but continue to report finalization
            return FinalizeReportsEvent(
                final_report=final_report,
                visualization_info="Error: DataFrame missing for visualization.",
                plot_paths=[],
                reports_to_verify=await ctx.get("required_reports", [])
            )
            
        # Generate visualizations
        visualization_results = await generate_visualizations(df, llm, modified_data_path)
        
        # Get the list of required reports that need to be verified
        required_reports = await ctx.get("required_reports", [
            "reports/data_quality_report.json",
            "reports/cleaning_report.json",
            "reports/statistical_analysis_report.json",
            "reports/regression_models.json",
            "reports/advanced_models.json"
        ])
        
        # Continue to report finalization step
        return FinalizeReportsEvent(
            final_report=final_report,
            visualization_info=visualization_results["visualization_info"],
            plot_paths=visualization_results["plot_paths"],
            reports_to_verify=required_reports
        )

    @step
    async def finalize_reports(self, ctx: Context, ev: FinalizeReportsEvent) -> StopEvent:
        """Verifies and finalizes all reports as the last step in the workflow."""
        print("--- Running Report Finalization Step ---")
        final_report = ev.final_report
        visualization_info = ev.visualization_info
        plot_paths = ev.plot_paths
        reports_to_verify = ev.reports_to_verify
        
        # Verify reports and generate status
        reports_status = await ReportUtils.verify_reports(reports_to_verify)
        
        # Attempt to regenerate any missing or incomplete reports
        for report_path, status in reports_status.items():
            if not status["complete"]:
                await ReportUtils.regenerate_report(ctx, report_path)
        
        # Update the final report with the status of all reports
        report_status_summary = ReportUtils.generate_report_status_summary(reports_status)
        final_report_with_status = final_report + report_status_summary
        
        # Create a condensed final result
        final_result = {
            "final_report": final_report_with_status,
            "visualization_info": visualization_info,
            "plot_paths": plot_paths,
            "reports_status": reports_status
        }
        
        print("Report finalization complete. Workflow finished.")
        return StopEvent(result=final_result)