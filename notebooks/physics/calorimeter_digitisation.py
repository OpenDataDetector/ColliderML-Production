import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List

# Import the individual digitizers
from ecal_digitisation import EcalDigitizer, digitise_ecal_hits
from hcal_digitisation import HcalDigitizer, digitise_hcal_hits

# Hardcoded calorimeter collection names from edm4hep_utils.py
ECAL_COLLECTIONS = [
    'ECalBarrelCollection',
    'ECalEndcapCollection',
]

HCAL_COLLECTIONS = [
    'HCalBarrelCollection',
    'HCalEndcapCollection',
]


def digitise_calorimeter_hits(hits_df: pd.DataFrame, 
                             contribs_df: pd.DataFrame,
                             detector_type: str = 'auto',
                             config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Digitize calorimeter hits, automatically selecting the appropriate digitizer.
    
    This function serves as a unified interface for both ECAL and HCAL digitization.
    It can automatically determine the detector type from the hits dataframe or
    use the specified detector type.
    
    Args:
        hits_df: DataFrame with hits
        contribs_df: DataFrame with contribution information (for timing)
        detector_type: 'ecal', 'hcal', or 'auto' to determine from the data
        config: Configuration parameters for the digitization
        
    Returns:
        DataFrame with digitized hits
    """
    # Determine detector type if set to auto
    if detector_type.lower() == 'auto':
        detector_type = _determine_detector_type(hits_df)
    
    # Apply the appropriate digitization
    if detector_type.lower() == 'ecal':
        return digitise_ecal_hits(hits_df, contribs_df, config)
    elif detector_type.lower() == 'hcal':
        return digitise_hcal_hits(hits_df, contribs_df, config)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}. Must be 'ecal', 'hcal', or 'auto'.")


def _determine_detector_type(hits_df: pd.DataFrame) -> str:
    """
    Determine the detector type from the hits dataframe.
    
    This function tries to infer whether the hits are from ECAL or HCAL
    based on the detector name in the dataframe.
    
    Args:
        hits_df: DataFrame with hits
        
    Returns:
        'ecal' or 'hcal'
    """
    if 'detector' not in hits_df.columns:
        # Default to ECAL if we can't determine
        return 'ecal'
    
    # Check if any hits are from ECAL or HCAL collections
    is_ecal = hits_df['detector'].isin(ECAL_COLLECTIONS).any()
    is_hcal = hits_df['detector'].isin(HCAL_COLLECTIONS).any()
    
    if is_ecal and not is_hcal:
        return 'ecal'
    elif is_hcal and not is_ecal:
        return 'hcal'
    elif is_ecal and is_hcal:
        # If both are present, count which has more hits
        ecal_count = hits_df[hits_df['detector'].isin(ECAL_COLLECTIONS)].shape[0]
        hcal_count = hits_df[hits_df['detector'].isin(HCAL_COLLECTIONS)].shape[0]
        return 'ecal' if ecal_count >= hcal_count else 'hcal'
    else:
        # If neither is present, default to ECAL
        return 'ecal'


def filter_ecal_hits(hits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter hits dataframe to only include ECAL hits.
    
    Args:
        hits_df: DataFrame with hits
        
    Returns:
        DataFrame with only ECAL hits
    """
    if 'detector' not in hits_df.columns:
        return pd.DataFrame()
    
    return hits_df[hits_df['detector'].isin(ECAL_COLLECTIONS)]


def filter_hcal_hits(hits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter hits dataframe to only include HCAL hits.
    
    Args:
        hits_df: DataFrame with hits
        
    Returns:
        DataFrame with only HCAL hits
    """
    if 'detector' not in hits_df.columns:
        return pd.DataFrame()
    
    return hits_df[hits_df['detector'].isin(HCAL_COLLECTIONS)]


def digitise_event(hits_df: pd.DataFrame,
                  contribs_df: pd.DataFrame = None,
                  ecal_config: Optional[Dict] = None,
                  hcal_config: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Digitize both ECAL and HCAL hits for a complete event.
    
    This function processes both calorimeter systems in one call,
    automatically filtering the hits by detector type.
    
    Args:
        hits_df: DataFrame with all calorimeter hits
        contribs_df: DataFrame with contribution information
        ecal_config: Configuration parameters for ECAL digitization
        hcal_config: Configuration parameters for HCAL digitization
        
    Returns:
        Tuple of (digitized_ecal_hits, digitized_hcal_hits)
    """
    # Filter hits by detector type
    ecal_hits = filter_ecal_hits(hits_df)
    hcal_hits = filter_hcal_hits(hits_df)
    
    # Filter contributions if provided
    ecal_contribs = None
    hcal_contribs = None
    
    if contribs_df is not None and 'detector' in contribs_df.columns:
        ecal_contribs = contribs_df[contribs_df['detector'].isin(ECAL_COLLECTIONS)]
        hcal_contribs = contribs_df[contribs_df['detector'].isin(HCAL_COLLECTIONS)]
    else:
        ecal_contribs = contribs_df
        hcal_contribs = contribs_df
    
    # Process ECAL hits
    digitized_ecal = digitise_ecal_hits(ecal_hits, ecal_contribs, ecal_config) if not ecal_hits.empty else pd.DataFrame()
    
    # Process HCAL hits
    digitized_hcal = digitise_hcal_hits(hcal_hits, hcal_contribs, hcal_config) if not hcal_hits.empty else pd.DataFrame()
    
    return digitized_ecal, digitized_hcal


class CalorimeterDigitizer:
    """
    Combined digitizer for both ECAL and HCAL.
    
    This class provides a unified interface for digitizing both
    electromagnetic and hadronic calorimeter hits.
    """
    
    def __init__(self, ecal_config: Optional[Dict] = None, hcal_config: Optional[Dict] = None):
        """
        Initialize the digitizer with configuration parameters.
        
        Args:
            ecal_config: Configuration parameters for ECAL digitization
            hcal_config: Configuration parameters for HCAL digitization
        """
        # Create individual digitizers
        self.ecal_digitizer = EcalDigitizer(ecal_config)
        self.hcal_digitizer = HcalDigitizer(hcal_config)
    
    def new_event(self):
        """
        Reset event-specific variables for a new event.
        """
        self.ecal_digitizer.new_event()
        self.hcal_digitizer.new_event()
    
    def digitise_ecal(self, hits_df: pd.DataFrame, contribs_df: pd.DataFrame = None,
                     id_columns: Tuple[str, str] = ('x', 'y')) -> pd.DataFrame:
        """
        Digitize ECAL hits.
        
        Args:
            hits_df: DataFrame with ECAL hits
            contribs_df: DataFrame with contribution information
            id_columns: Column names to use as cell ID
            
        Returns:
            DataFrame with digitized ECAL hits
        """
        # Filter to ensure only ECAL hits are processed
        ecal_hits = filter_ecal_hits(hits_df) if 'detector' in hits_df.columns else hits_df
        
        if ecal_hits.empty:
            return pd.DataFrame()
            
        return self.ecal_digitizer.digitise_dataframe(ecal_hits, contribs_df, id_columns)
    
    def digitise_hcal(self, hits_df: pd.DataFrame, contribs_df: pd.DataFrame = None,
                     id_columns: Tuple[str, str] = ('x', 'y')) -> pd.DataFrame:
        """
        Digitize HCAL hits.
        
        Args:
            hits_df: DataFrame with HCAL hits
            contribs_df: DataFrame with contribution information
            id_columns: Column names to use as cell ID
            
        Returns:
            DataFrame with digitized HCAL hits
        """
        # Filter to ensure only HCAL hits are processed
        hcal_hits = filter_hcal_hits(hits_df) if 'detector' in hits_df.columns else hits_df
        
        if hcal_hits.empty:
            return pd.DataFrame()
            
        return self.hcal_digitizer.digitise_dataframe(hcal_hits, contribs_df, id_columns)
    
    def digitise_event(self, hits_df: pd.DataFrame,
                      contribs_df: pd.DataFrame = None,
                      id_columns: Tuple[str, str] = ('x', 'y')) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Digitize both ECAL and HCAL hits for a complete event.
        
        Args:
            hits_df: DataFrame with all calorimeter hits
            contribs_df: DataFrame with contribution information
            id_columns: Column names to use as cell ID
            
        Returns:
            Tuple of (digitized_ecal_hits, digitized_hcal_hits)
        """
        # Reset for new event
        self.new_event()
        
        # Filter hits by detector type
        ecal_hits = filter_ecal_hits(hits_df)
        hcal_hits = filter_hcal_hits(hits_df)
        
        # Filter contributions if provided
        ecal_contribs = None
        hcal_contribs = None
        
        if contribs_df is not None and 'detector' in contribs_df.columns:
            ecal_contribs = contribs_df[contribs_df['detector'].isin(ECAL_COLLECTIONS)]
            hcal_contribs = contribs_df[contribs_df['detector'].isin(HCAL_COLLECTIONS)]
        else:
            ecal_contribs = contribs_df
            hcal_contribs = contribs_df
        
        # Process ECAL hits
        digitized_ecal = self.digitise_ecal(ecal_hits, ecal_contribs, id_columns) if not ecal_hits.empty else pd.DataFrame()
        
        # Process HCAL hits
        digitized_hcal = self.digitise_hcal(hcal_hits, hcal_contribs, id_columns) if not hcal_hits.empty else pd.DataFrame()
        
        return digitized_ecal, digitized_hcal 