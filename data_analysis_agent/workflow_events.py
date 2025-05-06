"""
Workflow Events

This module defines all event classes used in the data analysis workflow.
Events are used to pass data between workflow steps and trigger actions.
"""

from llama_index.core.workflow import Event
from typing import List, Dict, Any, Optional

# Basic workflow events
class InitialAssessmentEvent(Event):
    """Carries initial stats summary after loading data."""
    stats_summary: str
    column_info: dict 
    original_path: str

class DataPrepEvent(Event):
    """Event for data preparation details."""
    original_path: str
    column_names: list[str]
    stats_summary: str
    column_info: dict 

class DataAnalysisEvent(Event):
    """Event for conveying prepared data description to next step."""
    prepared_data_description: str
    original_path: str

class ModificationRequestEvent(Event):
    """Carries the user-approved modification description."""
    user_approved_description: str
    original_path: str

class ModificationCompleteEvent(Event):
    """Event indicating that data modifications have been completed."""
    original_path: str
    modification_summary: str | None = None

class CleaningInputRequiredEvent(Event):
    """Event indicating user input is needed for cleaning decisions."""
    issues: dict 
    prompt_message: str

class CleaningResponseEvent(Event):
    """Event carrying the user's cleaning decisions."""
    user_choices: dict

class VisualizationRequestEvent(Event):
    """Event indicating that data visualization is requested."""
    modified_data_path: str 
    report: str

# Advanced analytics events
class AdvancedAnalysisCompleteEvent(Event):
    """Event indicating that advanced statistical analysis is complete."""
    statistical_report: dict
    summary: str
    plot_info: str
    statistical_report_path: str
    modified_data_path: str

class RegressionModelingEvent(Event):
    """Event triggered after advanced analysis to perform regression modeling."""
    modified_data_path: str
    statistical_report_path: str

class RegressionCompleteEvent(Event):
    """Event triggered when regression modeling is complete."""
    modified_data_path: str
    regression_summary: str
    model_quality: str

# Report finalization events
class VisualizationCompleteEvent(Event):
    """Event triggered when visualization is complete."""
    final_report: str
    visualization_info: str
    plot_paths: list

class FinalizeReportsEvent(Event):
    """Event triggered to finalize all reports."""
    final_report: str
    visualization_info: str
    plot_paths: list
    reports_to_verify: list