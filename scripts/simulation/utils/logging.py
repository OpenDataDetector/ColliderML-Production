import logging
from pathlib import Path
import time
import traceback
from contextlib import contextmanager

def setup_logging(name="PDA_Chain"):
    """Configure logging for the chain"""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-8s %(name)-12s %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

class TimingRecorder:
    def __init__(self, output_dir):
        self.timings = {}
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.errors = []
        self.logger = logging.getLogger("TimingRecorder")

    @contextmanager
    def record(self, name):
        self.logger.info(f"Starting stage: {name}")
        start = time.time()
        try:
            yield
        except Exception as e:
            self.errors.append(f"Error in {name}: {str(e)}")
            raise  # Re-raise the exception after logging
        finally:
            end = time.time()
            duration = end - start
            self.timings[name] = duration
            self.logger.info(f"Completed stage: {name} in {duration:.2f} seconds")

    def write_report(self):
        try:
            total_time = time.time() - self.start_time
            report = ["Timing Report", "============="]
            
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
            
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            report_path = self.output_dir / "timing_report.txt"
            with open(report_path, "w") as f:
                f.write("\n".join(report))
            
            # Print to console
            print("\n".join(report))
            
        except Exception as e:
            print(f"Error writing timing report: {str(e)}")
            print(traceback.format_exc())