import os
import asyncio
import traceback
import logging
import datetime
import argparse
from pathlib import Path
from events import CleaningInputRequiredEvent, CleaningResponseEvent
from workflow import DataAnalysisFlow

def setup_logging(log_level=logging.INFO):
    """Set up logging to both console and file"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create a timestamped log filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"data_analysis_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels of logs
    
    # Clear existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with specified log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Create file handler which logs everything (debug level)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

async def run_workflow(dataset_path):
    """Run the data analysis workflow on the given dataset"""

    workflow = DataAnalysisFlow(timeout=300, verbose=True)

    try:
        handler = workflow.run(
            dataset_path=dataset_path,
        )

       
        async for event in handler.stream_events():
            logging.info(f"Run Workflow Loop: Received event: {type(event).__name__}")

            if isinstance(event, CleaningInputRequiredEvent):
                logging.info("Run Workflow Loop: Handling CleaningInputRequiredEvent.")
                user_input_numbers = input(event.prompt_message) 

                logging.info(f"Run Workflow Loop: User entered numbers: {user_input_numbers}")
                logging.info("Run Workflow Loop: Sending CleaningResponseEvent...")
               
                handler.ctx.send_event(
                    CleaningResponseEvent(user_choices={"numbers": user_input_numbers.strip()})
                )
                logging.info("Run Workflow Loop: Sent CleaningResponseEvent.")

        final_result_dict = await handler

        logging.info("\n==== Final Report ====")
        final_report = final_result_dict.get('final_report', 'N/A')
        logging.info(final_report)

        # Add visualization info
        viz_info = final_result_dict.get('visualization_info', 'No visualization info generated.')
        logging.info("\n==== Visualization Info ====")
        logging.info(viz_info)

        # Print plot paths if available
        plot_paths = final_result_dict.get('plot_paths', [])
        if plot_paths and isinstance(plot_paths, list):
            logging.info("\nGenerated Plots:")
            for path in plot_paths:
                # Check if it's an error message from the tool
                if "Error:" not in path:
                    logging.info(f"- {os.path.abspath(path)}") # Show absolute path
                else:
                    logging.info(f"- {path}") # Print error message
        elif isinstance(plot_paths, str): # Handle case where string message is returned
             logging.info(f"\nPlot Generation Note: {plot_paths}")

        return final_result_dict
    except Exception as e:
         logging.error(f"Workflow failed: {e}")
         logging.error(traceback.format_exc())
         return None

def parse_args():
    parser = argparse.ArgumentParser(description='Run data analysis workflow with logging')
    parser.add_argument('dataset_path', nargs='?', help='Path to the dataset CSV file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Set the console logging level')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set up logging with specified level
    log_level = getattr(logging, args.log_level)
    log_file = setup_logging(log_level)
    logging.info(f"Starting data analysis workflow. Log file: {log_file}")
    
    # Get dataset path from arguments or prompt
    dataset_path = args.dataset_path
    if not dataset_path:
        dataset_path = input("Enter the path to the dataset CSV file: ")
    
    # Ensure path uses correct path separators for the OS
    dataset_path = os.path.normpath(dataset_path)
    logging.info(f"Using dataset: {dataset_path}")
    
    # Run the workflow
    asyncio.run(run_workflow(dataset_path))
    logging.info("Workflow execution completed")