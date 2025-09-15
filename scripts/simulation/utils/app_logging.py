import logging
from pathlib import Path
import time
import traceback
from contextlib import contextmanager

def setup_logging(name="PDA_Chain", level=logging.INFO):
    """Configure logging for the chain
    
    Args:
        name: Logger name
        level: Logging level (logging.INFO, logging.WARNING, etc.)
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s %(name)-12s %(message)s'
        ))
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger

class TimingRecorder:
    def __init__(self, output_dir):
        self.timings = {}
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.errors = []
        self.error_occurred = False  # Flag to track if any error occurred
        self.logger = logging.getLogger("TimingRecorder")

    @contextmanager
    def record(self, name):
        self.logger.info(f"Starting stage: {name}")
        start = time.time()
        try:
            yield
        except Exception as e:
            self.errors.append(f"Error in {name}: {str(e)}")
            self.error_occurred = True  # Set the flag when an error occurs
            raise  # Re-raise the exception after logging
        finally:
            end = time.time()
            duration = end - start
            self.timings[name] = duration
            self.logger.info(f"Completed stage: {name} in {duration:.2f} seconds")

    def write_report(self):
        try:
            total_time = time.time() - self.start_time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Create report content
            report = [f"Timing Report ({timestamp})", "============="]
            
            # Indicate if errors occurred
            if self.error_occurred:
                report.append("*** Errors occurred during execution ***")
            
            # Add timing entries
            for name, duration in sorted(self.timings.items()):
                report.append(f"{name:<30} : {duration:>.2f} seconds")
            
            report.append("-" * 50)
            report.append(f"{'Total time':<30} : {total_time:>.2f} seconds")
            
            # Add error section if there were any errors
            if self.errors:
                report.append("\nErrors encountered:")
                report.append("===================")
                for error in self.errors:
                    report.append(error)
            
            # Append to summary file
            summary_path = self.output_dir / "timing_summary.txt"
            with open(summary_path, "a") as f:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("\n".join(report))
            
            # Print to console
            print("\n".join(report))
            
        except Exception as e:
            print(f"Error writing timing report: {str(e)}")
            print(traceback.format_exc())
