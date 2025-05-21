"""
PyEDM4hep - Python library for working with EDM4hep data format

This library provides an object-oriented interface for working with EDM4hep data,
focusing on HEP detector data from the ACTS simulation framework. It adapts and enhances
the functionality from utilities like edm4hep_utils.py, viz_utils.py, and particle_decay_tree.py
into a cohesive package.

Main components:
- Event: Wrapper around EDM4hep event data with accessor methods
- Particle: Class representing a particle with properties and relations
- Hits: Classes for different kinds of detector hits and their properties
- Decay: Tools for working with particle decay trees and graphs
- Plotting: Visualization tools for event, particle, and decay tree display
- Utils: Data loading and processing utilities

Example usage:
```python
from pyedm4hep import EDM4hepEvent
from pyedm4hep.plotting import PlottingHandler

# Get a particle and explore its properties
particle = event.get_particle(42)
print(f"PDG code: {particle.pdg}, pT: {particle.pt} GeV")

# Get and visualize decay tree
decay_tree = event.get_decay_tree()

# Visualize the event
plotter = PlottingHandler(event)
plotter.event_overview(detector_params={'tracking_radius': 1000, 'tracking_z_max': 2000})
```
"""

# Export main classes and functions for direct import
from .event import EDM4hepEvent
from .particle import Particle
from .hits import TrackerHit, CaloHit, CaloContribution
from .decay import DecayGraphHandler
from .plotting import PlottingHandler
# Add new imports for dataset handling and comparison
from .dataset import EDM4hepDataset

# Version info
__version__ = '0.1.1'
