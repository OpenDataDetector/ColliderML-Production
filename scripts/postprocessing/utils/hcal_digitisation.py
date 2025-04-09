import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union


class HcalDigitizer:
    """
    Scintillator HCAL digitization class based on DDCaloDigi implementation.
    
    This class implements the digitization effects for a scintillator-based 
    hadronic calorimeter with SiPM readout, including:
    
    1. Conversion of energy to photoelectrons
    2. SiPM saturation modeling
    3. Binomial fluctuations for pixel occupancy
    4. Pixel spread and electronics noise
    5. Saturation unfolding for energy reconstruction
    6. Dead channels
    7. Miscalibration (correlated and uncorrelated)
    8. Energy threshold effects
    9. Timing resolution and window cuts
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
            "hcal_PPD_pe_per_mip": 10,  # Photoelectrons per MIP
            "hcal_PPD_n_pixels": 400,   # Number of pixels in SiPM
            "calibHcalMip": 1.0e-4,     # MIP calibration factor
            "hcalMaxDynMip": 200,       # Max dynamic range in MIP units
            "hcal_pixSpread": 0.05,     # Relative spread of SiPM pixel signal
            "hcal_elec_noise": 0.0,     # Electronics noise as fraction of MIP
            
            # Threshold and timing
            "thresholdHcal": 2.0e-4,    # Energy threshold
            "hcalTimeResolution": 10.0,  # Time resolution in ns
            
            # Time window parameters
            "useHcalTiming": True,       # Whether to use timing information
            "hcalTimeWindowMin": -10.0,  # Minimum time window in ns
            "hcalBarrelTimeWindowMax": 100.0,  # Maximum time window for barrel in ns
            "hcalEndcapTimeWindowMax": 100.0,  # Maximum time window for endcap in ns
            "hcalDeltaTimeHitResolution": 10.0,  # Time resolution for hit merging
            "hcalCorrectTimesForPropagation": False,  # Correct for propagation time
            
            # Miscalibration
            "misCalibHcal_uncorrel": 0.0,  # Uncorrelated miscalibration
            "misCalibHcal_correl": 0.0,    # Correlated miscalibration
            "misCalibHcal_uncorrel_keep": False,  # Keep same miscalib between events
            "hcal_misCalibNpix": 0.05,     # Miscalibration of # SiPM pixels
            
            # Dead channels
            "deadCellFractionHcal": 0.0,   # Fraction of dead channels
            "deadCellHcal_keep": False,    # Keep same dead cells between events
            
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
        if self.config["misCalibHcal_correl"] > 0:
            self.event_correl_miscalib = self.rng.normal(
                1.0, self.config["misCalibHcal_correl"]
            )
    
    def digitise_hit(self, energy: float, time: float, cell_id: Tuple[int, int], 
                     is_barrel: bool = True, position: Optional[Tuple[float, float, float]] = None) -> Tuple[float, float]:
        """
        Digitize a single hit with all effects.
        
        Args:
            energy: Energy deposit in GeV
            time: Hit time in ns
            cell_id: Tuple of (id0, id1) for cell identification
            is_barrel: Whether the hit is in the barrel (True) or endcap (False)
            position: (x, y, z) position of the hit, used for time propagation correction
            
        Returns:
            Tuple of (digitized_energy, digitized_time)
        """
        # Apply threshold
        if energy <= self.config["thresholdHcal"]:
            return 0.0, 0.0
        
        # Apply time window cut
        if self.config["useHcalTiming"]:
            # Determine time window max based on detector region
            time_window_max = (self.config["hcalBarrelTimeWindowMax"] if is_barrel 
                              else self.config["hcalEndcapTimeWindowMax"])
            
            # Calculate propagation time correction if needed
            dt = 0.0
            if self.config["hcalCorrectTimesForPropagation"] and position is not None:
                x, y, z = position
                r = np.sqrt(x**2 + y**2 + z**2)
                dt = r / 300.0 - 0.1  # Light propagation time (r/c - offset)
            
            # Apply time window cut
            if time - dt <= self.config["hcalTimeWindowMin"] or time - dt >= time_window_max:
                return 0.0, 0.0
            
            # Apply timing resolution
            time = self.apply_timing_resolution(time)
        
        # Apply scintillator+SiPM digitization effects
        energy = self.scintillator_digi(energy)
        
        # Apply miscalibration
        energy = self.apply_miscalibration(energy, cell_id)
        
        # Apply dead cell effect
        if self.is_dead_cell(cell_id):
            energy = 0.0
        
        return energy, time
    
    def scintillator_digi(self, energy: float) -> float:
        """
        Apply scintillator+SiPM specific digitization effects.
        
        Args:
            energy: Energy deposit in GeV
            
        Returns:
            Digitized energy in GeV
        """
        # Convert energy to expected photoelectrons (via MIP calibration)
        npe = self.config["hcal_PPD_pe_per_mip"] * energy / self.config["calibHcalMip"]
        
        # Apply SiPM saturation
        npix = self.config["hcal_PPD_n_pixels"]
        if npix > 0:
            # Apply average SiPM saturation behavior
            npe = npix * (1.0 - np.exp(-npe / npix))
            
            # Apply binomial fluctuations
            p = npe / npix  # Fraction of hit pixels on SiPM
            npe = self.rng.binomial(npix, p)  # npe now quantized to integer pixels
        
        # Apply pixel spread (variations in pixel capacitance)
        if self.config["hcal_pixSpread"] > 0 and npe > 0:
            npe *= self.rng.normal(1.0, self.config["hcal_pixSpread"] / np.sqrt(npe))
        
        # Apply electronics dynamic range limit
        if self.config["hcalMaxDynMip"] > 0:
            max_npe = self.config["hcalMaxDynMip"] * self.config["hcal_PPD_pe_per_mip"]
            npe = min(npe, max_npe)
        
        # Add electronics noise
        if self.config["hcal_elec_noise"] > 0:
            noise = self.rng.normal(
                0, 
                self.config["hcal_elec_noise"] * self.config["hcal_PPD_pe_per_mip"]
            )
            npe += noise
        
        # Unfold the saturation
        if npix > 0:
            # Apply miscalibration to number of pixels
            smearedNpix = npix
            if self.config["hcal_misCalibNpix"] > 0:
                smearedNpix = npix * self.rng.normal(1.0, self.config["hcal_misCalibNpix"])
            
            # Threshold for linear continuation (95% of pixels)
            r = 0.95
            
            if npe < r * smearedNpix:
                # Standard unfolding for normal range
                npe = -smearedNpix * np.log(1.0 - (npe / smearedNpix))
            else:
                # Linear continuation for very high amplitudes
                npe = (1/(1-r) * (npe - r*smearedNpix) - 
                       smearedNpix * np.log(1-r))
        
        # Convert back to energy
        energy = self.config["calibHcalMip"] * npe / self.config["hcal_PPD_pe_per_mip"]
        
        return max(0, energy)  # Ensure energy is not negative
    
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
        if self.config["misCalibHcal_uncorrel"] > 0:
            if self.config["misCalibHcal_uncorrel_keep"]:
                # Use persistent miscalibration
                if cell_id not in self.cell_miscalibs:
                    # Generate new miscalibration for this cell
                    miscal = self.rng.normal(
                        1.0, self.config["misCalibHcal_uncorrel"]
                    )
                    self.cell_miscalibs[cell_id] = miscal
                else:
                    # Use existing miscalibration
                    miscal = self.cell_miscalibs[cell_id]
            else:
                # Generate new miscalibration each time
                miscal = self.rng.normal(
                    1.0, self.config["misCalibHcal_uncorrel"]
                )
            
            energy *= miscal
        
        # Apply correlated miscalibration
        if self.config["misCalibHcal_correl"] > 0:
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
        if self.config["deadCellFractionHcal"] <= 0:
            return False
            
        if self.config["deadCellHcal_keep"]:
            # Use persistent dead cell map
            if cell_id not in self.cell_dead:
                # Determine if this cell is dead
                is_dead = self.rng.uniform(0, 1) < self.config["deadCellFractionHcal"]
                self.cell_dead[cell_id] = is_dead
            else:
                # Use existing dead cell status
                is_dead = self.cell_dead[cell_id]
        else:
            # Determine randomly each time
            is_dead = self.rng.uniform(0, 1) < self.config["deadCellFractionHcal"]
            
        return is_dead
    
    def apply_timing_resolution(self, time: float) -> float:
        """
        Apply timing resolution effects.
        
        Args:
            time: Hit time in ns
            
        Returns:
            Smeared time in ns
        """
        if self.config["hcalTimeResolution"] > 0:
            time = self.rng.normal(time, self.config["hcalTimeResolution"])
        return time
    
    def get_min_time(self, hits_row, contribs_df):
        """
        Get the minimum time from contributions for a hit.
        
        Args:
            hits_row: Row from the hits DataFrame
            contribs_df: DataFrame with contribution information
            
        Returns:
            Minimum time from contributions
        """
        if 'time' not in contribs_df.columns:
            return 0.0
            
        if 'contribution_begin' not in hits_row or 'contribution_end' not in hits_row:
            return 0.0
            
        contributions = contribs_df.iloc[hits_row.contribution_begin:hits_row.contribution_end]
        if contributions.empty:
            return 0.0
            
        return contributions.time.min()
    
    def digitise_dataframe(self, hits_df: pd.DataFrame, contribs_df: pd.DataFrame = None, 
                          id_columns: Tuple[str, str] = ('x', 'y')) -> pd.DataFrame:
        """
        Digitize a dataframe of hits.
        
        Args:
            hits_df: DataFrame with hits
            contribs_df: DataFrame with contribution information (for timing)
            id_columns: Column names to use as cell ID
            
        Returns:
            DataFrame with digitized hits
        """
        # Create a copy to avoid modifying the original
        result_df = hits_df.copy()
        
        # Reset for new event
        self.new_event()
        
        # Extract timing information if available
        if contribs_df is not None and 'time' in contribs_df.columns:
            # Use apply to get minimum time for each hit
            result_df['time'] = result_df.apply(
                lambda row: self.get_min_time(row, contribs_df), axis=1
            )
        else:
            # No timing information available
            result_df['time'] = 0.0
        
        # Apply digitization to each hit
        digitized_hits = []
        for idx, row in result_df.iterrows():
            # Get cell ID
            cell_id = (int(row[id_columns[0]]), int(row[id_columns[1]]))
            
            # Get position for time propagation correction
            position = (row['x'], row['y'], row['z']) if all(c in row for c in ['x', 'y', 'z']) else None
            
            # Determine if hit is in barrel or endcap based on detector name
            is_barrel = True
            if 'detector' in row:
                detector_name = row['detector'].lower()
                is_barrel = 'barrel' in detector_name
            
            # Apply digitization
            energy, time = self.digitise_hit(row['energy'], row['time'], cell_id, is_barrel, position)
            
            # Update the result
            if energy > self.config["thresholdHcal"]:
                row_copy = row.copy()
                row_copy['energy'] = energy
                row_copy['time'] = time
                digitized_hits.append(row_copy)
        
        # Create new DataFrame with only hits that passed threshold and time window
        if digitized_hits:
            return pd.DataFrame(digitized_hits)
        else:
            # Return empty DataFrame with same columns
            return pd.DataFrame(columns=result_df.columns)


def digitise_hcal_hits(hits: pd.DataFrame, contribs: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Digitise the HCal hits and contributions.
    
    This function implements the digitization steps from DDCaloDigi for scintillator HCAL with SiPM readout.
    
    Args:
        hits: pd.DataFrame
            The hits dataframe with columns for energy, time, position, etc.
        contribs: pd.DataFrame
            The contributions dataframe with timing information.
        config: Optional[Dict]
            Configuration parameters for the digitization.
            
    Returns:
        pd.DataFrame
            The digitised hits dataframe.
    """
    # Create digitizer with provided or default config
    digitizer = HcalDigitizer(config)
    
    # Apply digitization to the hits dataframe
    digitized_hits = digitizer.digitise_dataframe(hits, contribs)
    
    return digitized_hits 