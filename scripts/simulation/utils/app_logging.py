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


# ------------------------------
# Stderr filtering utilities
# ------------------------------

@contextmanager
def filter_stderr(patterns):
    """Context manager to filter sys.stderr lines matching any regex patterns.

    Args:
        patterns: Iterable of compiled regex patterns or pattern strings.
    """
    import sys
    import re

    compiled_patterns = [re.compile(p) if isinstance(p, str) else p for p in patterns]

    def should_filter(line: str) -> bool:
        for pat in compiled_patterns:
            if pat.search(line):
                return True
        return False

    class _FilteredStderr:
        def __init__(self, original_stderr):
            self._original = original_stderr
            self._buffer = ""

        def write(self, text):
            self._buffer += text
            lines = self._buffer.split('\n')
            self._buffer = lines[-1]
            for line in lines[:-1]:
                if not should_filter(line):
                    self._original.write(line + '\n')

        def flush(self):
            if self._buffer and not should_filter(self._buffer):
                self._original.write(self._buffer)
                self._buffer = ""
            self._original.flush()

        def __getattr__(self, name):
            return getattr(self._original, name)

    original_stderr = sys.stderr
    try:
        sys.stderr = _FilteredStderr(original_stderr)
        yield
    finally:
        sys.stderr = original_stderr


def geant4_warning_filter_patterns():
    """Return regex patterns that match noisy Geant4 G4Exception warnings to suppress."""
    return [
        r'G4WT\d+ > \s*-------- WWWW ------- G4Exception-START',
        r'\*\*\* G4Exception : part\d+',
        r'Primary particle PDG=\d+ deltaMass\(MeV\)=[\d.]+',
        r'Specified mass\(MeV\)=[\d.]+.*PDG mass',
        r'To change the tolerance or the exception severity',
        r'\*\*\* This is just a warning message\.',
        r'-------- WWWW -------- G4Exception-END'
    ]


@contextmanager
def suppress_geant4_warning_exceptions():
    """Context manager to suppress common Geant4 G4Exception warning blocks.

    Usage:
        with suppress_geant4_warning_exceptions():
            ddsim.run()
    """
    import sys
    import re

    # Match either with or without G4WT prefix; tolerate variable dashes/spaces
    start_re = re.compile(r"(?:G4WT\d+\s*>\s*)?-+\s*WWWW\s*-+\s*G4Exception-START", re.IGNORECASE)
    end_re = re.compile(r"-+\s*WWWW\s*-+\s*G4Exception-END", re.IGNORECASE)
    header_re = re.compile(r"\*\*\*\s*G4Exception\s*:\s*", re.IGNORECASE)

    class _G4FilteredStderr:
        def __init__(self, original_stderr):
            self._original = original_stderr
            self._buffer = ""
            self._in_block = False

        def write(self, text):
            self._buffer += text
            lines = self._buffer.split('\n')
            self._buffer = lines[-1]
            for line in lines[:-1]:
                if self._in_block:
                    # If inside a block, suppress lines until END
                    if end_re.search(line):
                        self._in_block = False
                    # Do not write block lines
                    continue
                else:
                    # Detect start of a block either at banner or header line
                    if start_re.search(line) or header_re.search(line):
                        self._in_block = True
                        continue
                    self._original.write(line + '\n')

        def flush(self):
            if self._buffer and not self._in_block:
                self._original.write(self._buffer)
                self._buffer = ""
            # If in block, drop buffer silently
            self._original.flush()

        def __getattr__(self, name):
            return getattr(self._original, name)

    original_stderr = sys.stderr
    try:
        sys.stderr = _G4FilteredStderr(original_stderr)
        yield
    finally:
        sys.stderr = original_stderr