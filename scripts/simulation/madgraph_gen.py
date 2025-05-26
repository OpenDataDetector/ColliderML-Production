import os
import sys
import subprocess
import shutil
import argparse
import yaml
import re # Import regex module
import logging
from pathlib import Path
from utils.config import create_base_parser, load_config

logger = logging.getLogger(__name__)

def run_command(command, cwd=None, env=None, shell=False):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=cwd,
        env=env,
        shell=shell
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {command}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        sys.exit(1)
    return stdout, stderr

def customize_placeholder_card(template_path, output_path, customizations):
    """Customizes a card by replacing {{PLACEHOLDERS}}."""
    with open(template_path, 'r') as f:
        content = f.read()
    for placeholder, value in customizations.items():
        content = content.replace(str(placeholder), str(value))
    with open(output_path, 'w') as f:
        f.write(content)

def customize_run_card_with_regex(card_path, run_card_settings, global_nevents, global_seed):
    """Modifies the MadGraph-generated run_card.dat using regex for specific parameters."""
    with open(card_path, 'r') as f:
        content_lines = f.readlines()

    # Prepare a dictionary of settings to apply, including global ones
    settings_to_apply = run_card_settings.copy()
    settings_to_apply['nevents'] = global_nevents
    settings_to_apply['iseed'] = global_seed

    modified_lines = []
    for line in content_lines:
        modified_line = line
        for param_name, param_value in settings_to_apply.items():
            # Regex to find lines like: '  10000 = nevents    ! Number of even
            # This pattern looks for: <spaces><value_to_replace><spaces>=<spaces><param_name><spaces><comment_or_nothing>
            # It assumes the value is a single "word" (can include numbers, dots, minus)
            # and the parameter name is exact.
            strict_pattern = rf"^(\s*)(\S+)(\s*=\s*{re.escape(param_name)})(\s*!.*|\s*)$"

            match = re.match(strict_pattern, line)
            if match:
                # Construct the new line with the new value
                # Spaces (group 1), new value, rest of the matched line (group 3 and 4)
                modified_line = f"{match.group(1)}{str(param_value)}{match.group(3)}{match.group(4)}\n"
                # Since a parameter should only be set once, break from inner loop
                break 
        modified_lines.append(modified_line)

    with open(card_path, 'w') as f:
        f.writelines(modified_lines)
    print(f"Updated {card_path} with custom settings for: {list(settings_to_apply.keys())}")

def split_hepmc_file(input_hepmc_path: Path,
                     final_output_base_dir: Path,
                     events_per_file: int,
                     output_filename: str = "events.hepmc"):
    """
    Splits a single (potentially gzipped) HEPMC file into multiple smaller HEPMC files,
    each in its own subdirectory (0, 1, 2, etc.) under final_output_base_dir.
    """
    try:
        import pyhepmc as hep
        from pyhepmc.io import WriterAscii
    except ImportError:
        print("Error: pyhepmc library not found. Please install it to use HEPMC splitting (e.g., pip install pyhepmc).")
        print(f"Skipping splitting of {input_hepmc_path}.")
        return [] # Indicate no files created

    try:
        from tqdm import tqdm
    except ImportError:
        print("Warning: tqdm library not found. Progress bar for splitting will not be shown (e.g., pip install tqdm).")
        def tqdm_dummy(iterable, *args, **kwargs): # Renamed to avoid conflict if tqdm is later imported globally
            return iterable
        tqdm_actual = tqdm_dummy # Use the dummy
    else:
        tqdm_actual = tqdm # Use the real tqdm

    print(f"--- Splitting HEPMC file: {input_hepmc_path} ---")
    print(f"--- Output base directory for splits: {final_output_base_dir} ---")
    print(f"--- Events per split file: {events_per_file} ---")

    current_f_out = None
    files_created = []
    event_count_total = 0
    processed_successfully = False

    try:
        with hep.open(str(input_hepmc_path)) as f_in:
            desc = f"Splitting {input_hepmc_path.name}"
            iterator = tqdm_actual(enumerate(f_in), desc=desc)

            for i, event in iterator:                
                if i % events_per_file == 0:
                    if current_f_out:
                        current_f_out.close()
                    
                    file_idx = i // events_per_file
                    # Use just the run number as the directory name
                    current_split_output_dir = final_output_base_dir / str(file_idx)
                    os.makedirs(current_split_output_dir, exist_ok=True)
                    
                    split_file_path = current_split_output_dir / output_filename
                    current_f_out = WriterAscii(str(split_file_path)) # Ensure path is string for WriterAscii
                    files_created.append(split_file_path)
                
                if current_f_out:
                    event.event_number = i % events_per_file
                    current_f_out.write_event(event)
                event_count_total = i + 1
        
        processed_successfully = True # If loop completes without error

    except Exception as e:
        print(f"Error during HEPMC splitting of {input_hepmc_path}: {e}")
    finally:
        if current_f_out:
            try:
                current_f_out.close()
            except Exception as e_close:
                print(f"Error closing output file during HEPMC splitting: {e_close}")
        
    if processed_successfully and event_count_total > 0:
        print(f"--- Splitting complete. Processed {event_count_total} events from {input_hepmc_path.name} into {len(files_created)} files. ---")
        return files_created
    elif processed_successfully and event_count_total == 0:
        print(f"--- No events found or processed in {input_hepmc_path.name}. ---")
        return []
    else: # Not processed_successfully
        print(f"--- Splitting failed for {input_hepmc_path.name}. ---")
        # Clean up any partially created files from this attempt if needed, though current logic doesn't require it.
        return []

def main():
    parser = create_base_parser("MadGraph event generation for ColliderML")
    args = parser.parse_args()
    config = load_config(args)

    # Debug the config structure
    print("=== Config Structure Debugging ===")
    print(f"Config type: {type(config)}")
    print(f"Config dir contents: {dir(config)}")
    if hasattr(config, 'splitting_config'):
        print(f"splitting_config: {config.splitting_config}")
        print(f"splitting_config.enable: {config.splitting_config.get('enable')}")
    else:
        print("No 'splitting_config' attribute found in config!")
        try:
            # Try accessing as dict (old behavior)
            print(f"Trying dict access - splitting_config: {config.get('splitting_config')}")
        except:
            print("Dict access also failed")
    print("=================================")

    process_name = f"{config.dataset}_{config.version}"
    mg_base_path = Path(config.mg_base_path)
    scratch_dir = Path(config.generation_scratch_dir)

    # Determine the effective output directory (matches pythia_gen.py logic)
    effective_output_dir = Path(config.output)
    
    # When splitting is enabled, we create run_X directories directly under config.output
    # When splitting is disabled, we use config.output/config.output_subdir as in pythia_gen.py
    try:
        # First try attribute access (for utils.config style)
        if hasattr(config, 'splitting_config'):
            splitting_config = config.splitting_config
            print(f"Got splitting_config via attribute: {splitting_config}")
        else:
            # Fall back to dict access
            splitting_config = getattr(config, 'splitting_config', {})
            print(f"Got splitting_config via getattr: {splitting_config}")
        
        if isinstance(splitting_config, dict):
            splitting_enabled = splitting_config.get('enable', False)
            print(f"splitting_enabled = {splitting_enabled} (via dict access)")
        else:
            # Handle case where it might be an object with attributes
            splitting_enabled = getattr(splitting_config, 'enable', False)
            print(f"splitting_enabled = {splitting_enabled} (via attr access)")
    except Exception as e:
        print(f"Error accessing splitting config: {e}")
        print("Defaulting to splitting_enabled = False")
        splitting_enabled = False
    
    if splitting_enabled:
        # For splitting, use the base output dir without subdir
        # This ensures run X dirs are created directly at the base level
        print(f"--- Splitting enabled: Files will be placed in {effective_output_dir}/[0,1,2,...]/ ---")
    else:
        # For non-splitting case, use normal output/subdir structure
        if config.output_subdir:
            effective_output_dir = effective_output_dir / config.output_subdir
            print(f"--- Splitting disabled: Files will be placed in {effective_output_dir}/ ---")
    
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get other splitting config after determining the enabled state
    try:
        split_events_per_file = splitting_config.get('events_per_file', 1000)
        split_output_filename = splitting_config.get('output_filename', 'events.hepmc')
        print(f"Split events per file: {split_events_per_file}")
        print(f"Split output filename: {split_output_filename}")
    except Exception as e:
        print(f"Error accessing splitting config details: {e}")
        print("Using default values")
        split_events_per_file = 1000
        split_output_filename = 'events.hepmc'

    mg5_exe = mg_base_path / "bin" / "mg5_aMC"

    mg_model_cmd = f"import model {config.mg_model}"
    mg_define_cmds = config.mg_definitions
    mg_generate_cmd = config.mg_generate_command

    temp_run_dir = scratch_dir / f"mg5_run_{process_name}"
    temp_run_dir.mkdir(parents=True, exist_ok=True)
    
    madgraph_proc_output_dirname = "proc_output_mg"
    
    # --- Step 1: Generate Process Directory (includes default cards) ---
    temp_proc_script_path = temp_run_dir / "process_script.mg5"
    with open(temp_proc_script_path, 'w') as f_out:
        f_out.write(f"{mg_model_cmd}\n")
        for define_cmd in mg_define_cmds:
            f_out.write(f"{define_cmd}\n")
        f_out.write(f"{mg_generate_cmd}\n")
        f_out.write(f"output {madgraph_proc_output_dirname} -f\n")
        f_out.write("exit\n")

    print(f"--- Running MadGraph for process generation (script: {temp_proc_script_path}) ---")
    stdout_proc, stderr_proc = run_command([str(mg5_exe), str(temp_proc_script_path)], cwd=temp_run_dir)
    print("--- MadGraph process generation STDOUT: ---")
    print(stdout_proc)
    if stderr_proc:
        print("--- MadGraph process generation STDERR: ---")
        print(stderr_proc)
    print("--- MadGraph process generation complete. ---")

    # Process directory path
    generated_process_dir = temp_run_dir / madgraph_proc_output_dirname
    cards_dir = generated_process_dir / "Cards"
    if not cards_dir.is_dir():
        print(f"Error: Cards directory not found at {cards_dir} after process generation.")
        sys.exit(1)

    # --- Step 2a: Customize run_card.dat using Regex ---
    run_card_path = cards_dir / "run_card.dat"
    run_card_settings_from_config = config.card_customizations.get('run_card', {})
    print(f"--- Customizing run_card.dat at {run_card_path} using regex... ---")
    customize_run_card_with_regex(run_card_path, 
                                  run_card_settings_from_config, 
                                  config.events, 
                                  config.seed)

    # --- Step 2b: Customize pythia8_card_default.dat using Placeholders ---
    if 'pythia8_card' in config.card_templates:
        pythia8_template_name = config.card_templates['pythia8_card']
        pythia8_template_path = Path(__file__).parent / "card_templates" / pythia8_template_name
        pythia8_output_path = cards_dir / "pythia8_card_default.dat"
        pythia8_customizations = config.card_customizations.get('pythia8_card', {})
        
        # Add global settings to pythia8 customizations if they use generic placeholders
        # For example, if pythia8_card template uses {{NEVENTS}} or {{ISEED}}
        # Note: current yaml has {{PYTHIA_SEED}} not {{ISEED}}
        # pythia8_customizations['{{NEVENTS}}'] = config.events 

        if pythia8_template_path.exists():
            print(f"--- Customizing pythia8_card_default.dat from template {pythia8_template_name}... ---")
            customize_placeholder_card(pythia8_template_path, pythia8_output_path, pythia8_customizations)
        else:
            print(f"Warning: Pythia8 card template {pythia8_template_name} not found. Skipping customization.")
    else:
        print("No pythia8_card template specified in config. Assuming MG default is okay or not used.")

    # --- Step 3: Launch MadGraph event generation ---
    temp_launch_script_path = temp_run_dir / "launch_script.mg5"
    with open(temp_launch_script_path, 'w') as f:
        f.write(f"launch {generated_process_dir} -f\n")
        f.write("set automatic_html_opening False\n")
        f.write("exit\n") 
    
    print(f"--- Running MadGraph for event generation (script: {temp_launch_script_path}) ---")
    stdout_event, stderr_event = run_command([str(mg5_exe), str(temp_launch_script_path)], cwd=generated_process_dir)
    print("--- MadGraph event generation STDOUT: ---")
    print(stdout_event)
    if stderr_event:
        print("--- MadGraph event generation STDERR: ---")
        print(stderr_event)
    print("--- MadGraph event generation complete. ---")

    # --- Step 4: Process and Move/Split output files ---
    print("--- Processing and Moving/Splitting output event files... ---")
    events_dir_in_process = generated_process_dir / "Events"
    actual_events_subdirs = list(events_dir_in_process.glob("run_*"))
    if not actual_events_subdirs:
        if events_dir_in_process.is_dir():
             actual_events_subdirs = [events_dir_in_process]
        else:
            print(f"Warning: Events directory {events_dir_in_process} not found.")
            actual_events_subdirs = []

    files_processed_count = 0
    
    for events_subdir_path in actual_events_subdirs:
        if events_subdir_path.is_dir():
            # Process LHE files: move them directly
            for pattern in ["*.lhe", "*.lhe.gz"]:
                for event_file_path in events_subdir_path.glob(pattern):
                    try:
                        destination_path = effective_output_dir / event_file_path.name
                        shutil.move(str(event_file_path), str(destination_path))
                        print(f"Moved LHE file {event_file_path.name} to {destination_path}")
                        files_processed_count += 1
                    except Exception as e:
                        print(f"Error moving LHE file {event_file_path.name}: {e}")

            # Process HepMC files: split if enabled, otherwise move
            for pattern in ["*.hepmc", "*.hepmc.gz"]:
                for event_file_path in events_subdir_path.glob(pattern):
                    if splitting_enabled:
                        print(f"Processing HEPMC file for splitting: {event_file_path}")
                        # Create run_X subdirs inside effective_output_dir
                        created_split_files = split_hepmc_file(
                            input_hepmc_path=event_file_path,
                            final_output_base_dir=effective_output_dir, # run_X subdirs will be created here
                            events_per_file=split_events_per_file,
                            output_filename=split_output_filename
                        )
                        if created_split_files:
                            files_processed_count += len(created_split_files)
                            try:
                                event_file_path.unlink() # Remove original temp file
                                print(f"Removed original temporary HEPMC file {event_file_path} after successful splitting.")
                            except OSError as e:
                                print(f"Warning: Could not remove original temporary HEPMC file {event_file_path}: {e}")
                        else:
                            print(f"HEPMC splitting produced no files for {event_file_path} or was skipped. Attempting to move original.")
                            try:
                                destination_path = effective_output_dir / event_file_path.name
                                shutil.move(str(event_file_path), str(destination_path))
                                print(f"Moved original HEPMC file {event_file_path.name} to {destination_path} (splitting failed/skipped).")
                                files_processed_count += 1
                            except Exception as e:
                                print(f"Error moving original HEPMC file {event_file_path.name} after failed/skipped split: {e}")
                    else:
                        # Splitting not enabled, move the HEPMC file directly
                        try:
                            destination_path = effective_output_dir / event_file_path.name
                            shutil.move(str(event_file_path), str(destination_path))
                            print(f"Moved HEPMC file {event_file_path.name} to {destination_path} (splitting disabled).")
                            files_processed_count += 1
                        except Exception as e:
                            print(f"Error moving HEPMC file {event_file_path.name}: {e}")
    
    if files_processed_count == 0:
        print("Warning: No event files were found, moved, or split from the MadGraph run.")

    # --- Step 5: Cleanup ---
    print(f"--- Cleaning up temporary directory: {temp_run_dir}... ---")
    shutil.rmtree(temp_run_dir)
    print(f"--- Cleaned up temporary directory: {temp_run_dir} ---")

    print("--- MadGraph generation pipeline finished. ---")

if __name__ == "__main__":
    main() 