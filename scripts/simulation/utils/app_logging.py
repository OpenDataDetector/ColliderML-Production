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


@contextmanager
def suppress_geant4_warning_exceptions_fd():
    """FD-level variant to suppress Geant4 G4Exception warning blocks (captures C++ writes).

    This redirects file descriptor 2 (stderr) to a pipe and filters lines in a reader thread.
    """
    import os
    import sys
    import threading
    import re

    start_re = re.compile(r"(?:G4WT\d+\s*>\s*)?-+\s*WWWW\s*-+\s*G4Exception-START", re.IGNORECASE)
    end_re = re.compile(r"-+\s*WWWW\s*-+\s*G4Exception-END", re.IGNORECASE)
    header_re = re.compile(r"\*\*\*\s*G4Exception\s*:\s*", re.IGNORECASE)

    # Duplicate original stderr FD to restore later
    orig_fd = os.dup(2)
    r_fd, w_fd = os.pipe()

    # Make FD 2 point to write end of our pipe
    os.dup2(w_fd, 2)
    os.close(w_fd)

    # Reader thread: read lines from r_fd, filter, and write to orig_fd
    stop_evt = threading.Event()

    def reader():
        with os.fdopen(r_fd, 'rb', buffering=0) as rfp, os.fdopen(orig_fd, 'wb', buffering=0) as ofp:
            buf = bytearray()
            in_block = False
            while not stop_evt.is_set():
                chunk = rfp.read(1024)
                if not chunk:
                    break
                buf.extend(chunk)
                while True:
                    nl = buf.find(b'\n')
                    if nl < 0:
                        break
                    line = bytes(buf[:nl]).decode(errors='replace')
                    del buf[:nl+1]
                    if in_block:
                        if end_re.search(line):
                            in_block = False
                        continue
                    else:
                        if start_re.search(line) or header_re.search(line):
                            in_block = True
                            continue
                        ofp.write(line.encode('utf-8', errors='replace') + b'\n')
            # Flush any remaining non-block buffered content
            if buf and not in_block:
                try:
                    ofp.write(bytes(buf))
                except Exception:
                    pass

    t = threading.Thread(target=reader, name="stderr-g4-filter", daemon=True)
    t.start()
    try:
        yield
    finally:
        # Restore FD 2
        os.dup2(orig_fd, 2)
        # Signal and join reader
        stop_evt.set()
        try:
            t.join(timeout=1.0)
        except Exception:
            pass