import uproot
import glob
from typing import Optional, List, Dict, Generator
import os
import time

# Only import Event class
from .event import EDM4hepEvent


class EDM4hepDataset:
    """
    Iterates through EDM4hep files, yielding Event objects initialized
    one by one using their file path and event index.

    Note: This approach loads each event individually, which can be
    less performant than batch loading for large datasets.
    """

    def __init__(self,
                 dataset_path: str,
                 name: Optional[str] = None,
                 file_pattern: str = "**/edm4hep.root", # Recursive search by default
                 max_files: Optional[int] = None,
                 max_events_per_file: Optional[int] = None,
                 detector_params: Optional[Dict[str, float]] = None):
                 # batch_size and features_to_load are no longer used here
        """
        Initializes the dataset iterator configuration.

        Args:
            dataset_path (str): Path to the root directory of the dataset.
            name (Optional[str]): A name for this dataset. Defaults to dataset_path basename.
            file_pattern (str): Glob pattern to find ROOT files within dataset_path.
            max_files (Optional[int]): Maximum number of files to process.
            max_events_per_file (Optional[int]): Maximum number of events to process from each file.
            detector_params (Optional[Dict[str, float]]): Optional detector geometry parameters
                 to pass to each Event object during initialization.
        """
        self.dataset_path = os.path.abspath(dataset_path)
        self.name = name if name else os.path.basename(self.dataset_path)
        self.file_pattern = file_pattern
        self.max_files = max_files
        self.max_events_per_file = max_events_per_file
        self.detector_params = detector_params # Store detector params

        self.files: List[str] = self._find_files()

        if not self.files:
            print(f"Warning: No files found for pattern '{os.path.join(self.dataset_path, self.file_pattern)}'")

    def _find_files(self) -> List[str]:
        """Finds ROOT files matching the pattern within the dataset path."""
        search_path = os.path.join(self.dataset_path, self.file_pattern)
        files = sorted(glob.glob(search_path, recursive=True))
        if not files:
            # Warning printed in __init__
            return []
        if self.max_files is not None and self.max_files < len(files):
            print(f"Limiting to first {self.max_files} files out of {len(files)} found.")
            return files[:self.max_files]
        print(f"Found {len(files)} files.")
        return files

    # Removed _determine_branches, _get_uproot_key, _filter_branches_for_file

    def __iter__(self) -> Generator[EDM4hepEvent, None, None]:
        """
        Iterates through files and event indices, yielding initialized Event objects.
        The Event object itself handles loading data for its specific index.
        """
        total_events_yielded = 0
        files_processed_count = 0
        start_time = time.time()

        for file_idx, file_path in enumerate(self.files):
            try:
                # Open file briefly just to get number of events
                with uproot.open(file_path) as f:
                    if "events" not in f:
                        print(f"  Warning: 'events' tree not found in {file_path}. Skipping.")
                        continue
                    events_tree = f["events"]
                    num_entries = events_tree.num_entries

                if num_entries == 0:
                    # print(f"  Skipping empty file: {os.path.basename(file_path)}")
                    continue

                num_events_to_process = num_entries
                if self.max_events_per_file is not None:
                    num_events_to_process = min(num_entries, self.max_events_per_file)

                files_processed_count += 1
                print(f"Processing file {files_processed_count}/{len(self.files)}: {os.path.basename(file_path)} ({num_events_to_process} events)..." if len(self.files) > 1 else f"Processing file: {os.path.basename(file_path)} ({num_events_to_process} events)...")

                # Loop through event indices for this file
                for i in range(num_events_to_process):
                    try:
                        # Instantiate Event - it will load its own data via file_path and index i
                        event = EDM4hepEvent(file_path=file_path,
                                           event_index=i,
                                           detector_params=self.detector_params)
                        yield event
                        total_events_yielded += 1
                    except FileNotFoundError:
                         # If the file disappears between finding it and Event trying to load it
                         print(f"  Error: File not found when trying to load event {i} from {file_path}. Skipping rest of file.")
                         break # Stop processing this file
                    except ValueError as e:
                         # Catch errors from Event._load_and_assign_data (e.g., bad index, load fail)
                         print(f"  Error loading event {i} from {os.path.basename(file_path)}: {e}. Skipping event.")
                         continue # Skip to next event index
                    except Exception as e:
                         # Catch any other unexpected errors during Event instantiation or loading
                         print(f"  Unexpected error processing event {i} from {os.path.basename(file_path)}: {e}. Skipping event.")
                         continue # Skip to next event index

            except FileNotFoundError:
                 print(f"  Error: File not found {file_path} when attempting to open. Skipping.")
                 continue
            except Exception as e:
                # Catch errors during file opening or getting num_entries
                print(f"  Error processing file {file_path}: {e}. Skipping file.")
                continue # Skip to next file

        end_time = time.time()
        print(f"\nFinished iterating. Yielded {total_events_yielded} events from {files_processed_count} files in {end_time - start_time:.2f} seconds.")

    # Removed __len__

    def __repr__(self) -> str:
         # Simplified repr, removed features
         return f"<EDM4hepDatasetIterator name='{self.name}' path='{self.dataset_path}' files={len(self.files)} max_events_per_file={self.max_events_per_file}>"
