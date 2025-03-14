import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union


class EcalDigitizer:
    """
    Silicon ECAL digitization class based on DDCaloDigi implementation.
    
    This class implements the digitization effects for a silicon-based 
    electromagnetic calorimeter, including:
    
    1. Conversion of energy to electron-hole pairs
    2. Poisson fluctuations of the number of pairs
    3. Electronics noise
    4. Limited dynamic range
    5. Dead channels
    6. Miscalibration (correlated and uncorrelated)
    7. Energy threshold effects
    8. Timing resolution
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the digitizer with configuration parameters.
        
        Args:
            config: Dictionary with configuration parameters
        """
        # Default configuration
        self.config = {
            # Basic parameters
            "ehEnergy": 3.6e-9,  # Energy to create e-h pair in silicon (GeV)
            "calibEcalMip": 1.0e-4,  # MIP calibration factor
            "ecalMaxDynMip": 2500,  # Max dynamic range in MIP units
            "ecal_elec_noise": 0.0,  # Electronics noise as fraction of MIP
            
            # Threshold and timing
            "thresholdEcal": 5.0e-5,  # Energy threshold
            "ecalTimeResolution": 10.0,  # Time resolution in ns
            
            # Miscalibration
            "misCalibEcal_uncorrel": 0.0,  # Uncorrelated miscalibration
            "misCalibEcal_correl": 0.0,  # Correlated miscalibration
            "misCalibEcal_uncorrel_keep": False,  # Keep same miscalib between events
            
            # Dead channels
            "deadCellFractionEcal": 0.0,  # Fraction of dead channels
            "deadCellEcal_keep": False,  # Keep same dead cells between events
            
            # Random seed
            "random_seed": 42  # Seed for reproducibility
        }
        
        # Update with user config if provided
        if config is not None:
            self.config.update(config)
        
        # Initialize random number generator
        self.rng = np.random.RandomState(self.config["random_seed"])
        
        # Storage for persistent miscalibration and dead cells
        self.cell_miscalibs = {}
        self.cell_dead = {}
        
        # Event-correlated miscalibration (changes per event)
        self.event_correl_miscalib = 1.0
        
    def new_event(self):
        """
        Reset event-specific variables for a new event.
        """
        # Generate new event-correlated miscalibration
        if self.config["misCalibEcal_correl"] > 0:
            self.event_correl_miscalib = self.rng.normal(
                1.0, self.config["misCalibEcal_correl"]
            )
    
    def digitise_hit(self, energy: float, time: float, cell_id: Tuple[int, int]) -> Tuple[float, float]:
        """
        Digitize a single hit with all effects.
        
        Args:
            energy: Energy deposit in GeV
            time: Hit time in ns
            cell_id: Tuple of (id0, id1) for cell identification
            
        Returns:
            Tuple of (digitized_energy, digitized_time)
        """
        # Apply threshold
        if energy <= self.config["thresholdEcal"]:
            return 0.0, 0.0
        
        # Apply silicon digitization effects
        energy = self.silicon_digi(energy)
        
        # Apply miscalibration
        energy = self.apply_miscalibration(energy, cell_id)
        
        # Apply dead cell effect
        if self.is_dead_cell(cell_id):
            energy = 0.0
        
        # Apply timing resolution
        time = self.apply_timing_resolution(time)
        
        return energy, time
    
    def silicon_digi(self, energy: float) -> float:
        """
        Apply silicon-specific digitization effects.
        
        Args:
            energy: Energy deposit in GeV
            
        Returns:
            Digitized energy in GeV
        """
        # Calculate number of electron-hole pairs
        # energy in GeV, ehEnergy in GeV (3.6e-9 GeV = 3.6 eV)
        n_eh_pairs = energy / self.config["ehEnergy"]
        
        # Apply Poisson fluctuations
        if n_eh_pairs > 0:
            smeared_energy = energy * self.rng.poisson(n_eh_pairs) / n_eh_pairs
        else:
            smeared_energy = energy
        
        # Apply electronics dynamic range limit
        max_energy = self.config["ecalMaxDynMip"] * self.config["calibEcalMip"]
        if self.config["ecalMaxDynMip"] > 0:
            smeared_energy = min(smeared_energy, max_energy)
        
        # Add electronics noise
        if self.config["ecal_elec_noise"] > 0:
            noise = self.rng.normal(
                0, 
                self.config["ecal_elec_noise"] * self.config["calibEcalMip"]
            )
            smeared_energy += noise
            
        return max(0, smeared_energy)  # Ensure energy is not negative
    
    def apply_miscalibration(self, energy: float, cell_id: Tuple[int, int]) -> float:
        """
        Apply miscalibration effects to the energy.
        
        Args:
            energy: Energy deposit in GeV
            cell_id: Tuple of (id0, id1) for cell identification
            
        Returns:
            Miscalibrated energy in GeV
        """
        # Apply uncorrelated miscalibration
        if self.config["misCalibEcal_uncorrel"] > 0:
            if self.config["misCalibEcal_uncorrel_keep"]:
                # Use persistent miscalibration
                if cell_id not in self.cell_miscalibs:
                    # Generate new miscalibration for this cell
                    miscal = self.rng.normal(
                        1.0, self.config["misCalibEcal_uncorrel"]
                    )
                    self.cell_miscalibs[cell_id] = miscal
                else:
                    # Use existing miscalibration
                    miscal = self.cell_miscalibs[cell_id]
            else:
                # Generate new miscalibration each time
                miscal = self.rng.normal(
                    1.0, self.config["misCalibEcal_uncorrel"]
                )
            
            energy *= miscal
        
        # Apply correlated miscalibration
        if self.config["misCalibEcal_correl"] > 0:
            energy *= self.event_correl_miscalib
            
        return energy
    
    def is_dead_cell(self, cell_id: Tuple[int, int]) -> bool:
        """
        Check if a cell is dead.
        
        Args:
            cell_id: Tuple of (id0, id1) for cell identification
            
        Returns:
            True if the cell is dead, False otherwise
        """
        if self.config["deadCellFractionEcal"] <= 0:
            return False
            
        if self.config["deadCellEcal_keep"]:
            # Use persistent dead cell map
            if cell_id not in self.cell_dead:
                # Determine if this cell is dead
                is_dead = self.rng.uniform(0, 1) < self.config["deadCellFractionEcal"]
                self.cell_dead[cell_id] = is_dead
            else:
                # Use existing dead cell status
                is_dead = self.cell_dead[cell_id]
        else:
            # Determine randomly each time
            is_dead = self.rng.uniform(0, 1) < self.config["deadCellFractionEcal"]
            
        return is_dead
    
    def apply_timing_resolution(self, time: float) -> float:
        """
        Apply timing resolution effects.
        
        Args:
            time: Hit time in ns
            
        Returns:
            Smeared time in ns
        """
        if self.config["ecalTimeResolution"] > 0:
            time = self.rng.normal(time, self.config["ecalTimeResolution"])
        return time
    
    def digitise_dataframe(self, hits_df: pd.DataFrame, id_columns: Tuple[str, str] = ('x', 'y')) -> pd.DataFrame:
        """
        Digitize a dataframe of hits.
        
        Args:
            hits_df: DataFrame with hits
            id_columns: Column names to use as cell ID
            
        Returns:
            DataFrame with digitized hits
        """
        # Create a copy to avoid modifying the original
        result_df = hits_df.copy()
        
        # Reset for new event
        self.new_event()
        
        # Apply digitization to each hit
        for idx, row in result_df.iterrows():
            cell_id = (int(row[id_columns[0]]), int(row[id_columns[1]]))
            energy, time = self.digitise_hit(row['energy'], row['time'], cell_id)
            
            result_df.at[idx, 'energy'] = energy
            result_df.at[idx, 'time'] = time
            
        # Remove hits below threshold
        result_df = result_df[result_df['energy'] > self.config["thresholdEcal"]]
        
        return result_df


def digitise_ecal_hits(hits: pd.DataFrame, contribs: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Digitise the ECal hits and contributions.
    
    This function implements the digitization steps from DDCaloDigi for silicon ECAL.
    
    Args:
        hits: pd.DataFrame
            The hits dataframe with columns for energy, time, position, etc.
        contribs: pd.DataFrame
            The contributions dataframe (not used in current implementation).
        config: Optional[Dict]
            Configuration parameters for the digitization.
            
    Returns:
        pd.DataFrame
            The digitised hits dataframe.
    """
    # Create digitizer with provided or default config
    digitizer = EcalDigitizer(config)
    
    # Apply digitization to the hits dataframe
    digitized_hits = digitizer.digitise_dataframe(hits)
    
    return digitized_hits 