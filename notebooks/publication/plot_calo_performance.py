#!/usr/bin/env python
"""
Plot calorimeter digitization performance.

This script loads truth and digitized calorimeter data,
computes residuals and pulls, and generates publication-quality plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import yaml
import argparse
from pathlib import Path

# Add pyedm4hep to path
sys.path.append("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/OtherLibraries/pyedm4hep")
from pyedm4hep import EDM4hepEventBatch

import atlasify as atl
atl.ATLAS = "ColliderML"


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config):
    """Load truth and digitized data."""
    detector_params = config['detector_params']
    events = tuple(config['events_range'])
    
    # Load digitized file (contains both truth and digi)
    print(f"Loading data from {config['digi_file']}")
    batch = EDM4hepEventBatch(
        config['digi_file'], 
        events=events, 
        full_load=False, 
        detector_params=detector_params
    )
    
    # Get dataframes
    particles = batch.get_particles_df()
    digi_hits = batch.get_digi_calo_hits_df()
    
    return particles, digi_hits


def process_data(particles, digi_hits, pdg_code=11):
    """Extract generator particles and compute residuals.
    
    Args:
        particles: DataFrame of MCParticles
        digi_hits: DataFrame of digitized calo hits
        pdg_code: PDG code to filter (default 11 for electrons)
    
    Returns:
        combined_df: DataFrame with generator particles and calo energy
        residual, relative_residual, pull: Series with computed quantities
    """
    # Extract generator particles (not created in simulation)
    generator_particles = particles[
        (particles["PDG"] == pdg_code) & 
        (particles["created_in_simulation"] == False)
    ].copy()
    
    # Sum digitized energy per event
    digi_sum = digi_hits.groupby("event_id")["energy"].sum().reset_index(name="energy_sum")
    
    # Merge generator particles with calo energy
    combined = generator_particles.merge(
        digi_sum, 
        left_on="event_id", 
        right_on="event_id", 
        how="inner"
    )
    
    # Compute pt if not present
    if 'pt' not in combined.columns:
        combined['pt'] = np.sqrt(combined['px']**2 + combined['py']**2)
    
    # Compute residuals
    residual = (combined["energy"] - combined["energy_sum"]).abs()
    relative_residual = residual / combined["energy"]
    pull = (combined["energy"] - combined["energy_sum"]) / np.sqrt(combined["energy_sum"])
    
    # Handle cases where energy_sum is 0
    pull[combined["energy_sum"] == 0] = 0
    
    return combined, residual, relative_residual, pull


def plot_energy_distribution(generator_particles, output_dir, config):
    """Plot generator energy distribution."""
    fig, ax = plt.subplots()
    
    energy_data = generator_particles["energy"]
    log_bins = np.logspace(
        np.log10(energy_data.min()), 
        np.log10(energy_data.max()), 
        config['plot_params']['energy_nbins']
    )
    counts, bins = np.histogram(energy_data, bins=log_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = bins[1:] - bins[:-1]
    errors = np.sqrt(counts) / bin_widths
    
    ax.errorbar(bin_centers, counts / bin_widths, yerr=errors, 
                fmt='o', color='royalblue', capsize=2, markersize=3)
    ax.set_xlabel('Energy [GeV]')
    ax.set_ylabel('Events / GeV')
    ax.set_xscale('log')
    atl.atlasify("Simulation", config['plot_params']['energy_label'], enlarge=1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / "energy_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "energy_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved energy_distribution.pdf")


def plot_residuals_pulls(residual, relative_residual, pull, output_dir, config):
    """Plot residual, relative residual, and pull distributions."""
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot 1: Absolute residual
    bins_residual = np.logspace(
        np.log10(residual[residual > 0].min()), 
        np.log10(residual.max()), 
        50
    )
    counts, bins = np.histogram(residual, bins=bins_residual)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = bins[1:] - bins[:-1]
    errors = np.sqrt(counts)
    
    ax[0].errorbar(bin_centers, counts, xerr=bin_widths/2, yerr=errors, 
                   fmt='o', color='royalblue', capsize=2, markersize=3, 
                   elinewidth=1, markeredgewidth=1)
    ax[0].set_xlabel('Energy Residual [GeV]')
    ax[0].set_ylabel('Events')
    ax[0].set_xscale('log')
    ax[0].grid(True, alpha=0.3)
    atl.atlasify("Simulation", r"$|E_{\mathrm{gen}} - E_{\mathrm{calo}}|$", 
                 axes=ax[0], enlarge=1.0)
    
    # Plot 2: Relative residual
    bins_relative = np.logspace(
        np.log10(relative_residual[relative_residual > 0].min()), 
        np.log10(relative_residual.max()), 
        50
    )
    counts, bins = np.histogram(relative_residual, bins=bins_relative)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = bins[1:] - bins[:-1]
    errors = np.sqrt(counts)
    
    ax[1].errorbar(bin_centers, counts, xerr=bin_widths/2, yerr=errors, 
                   fmt='o', color='royalblue', capsize=2, markersize=3, 
                   elinewidth=1, markeredgewidth=1)
    ax[1].set_xlabel('Relative Energy Residual')
    ax[1].set_ylabel('Events')
    ax[1].set_xscale('log')
    ax[1].grid(True, alpha=0.3)
    atl.atlasify("Simulation", r"$|E_{\mathrm{gen}} - E_{\mathrm{calo}}| / E_{\mathrm{gen}}$", 
                 axes=ax[1], enlarge=1.0)
    
    # Plot 3: Pull distribution
    bins_pull = np.linspace(-3, 3, 50)
    counts, bins = np.histogram(pull, bins=bins_pull)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = bins[1:] - bins[:-1]
    errors = np.sqrt(counts)
    
    ax[2].errorbar(bin_centers, counts, xerr=bin_widths/2, yerr=errors, 
                   fmt='o', color='royalblue', capsize=2, markersize=3, 
                   elinewidth=1, markeredgewidth=1)
    ax[2].set_xlabel('Pull')
    ax[2].set_ylabel('Events')
    ax[2].set_xlim(-3, 3)
    ax[2].grid(True, alpha=0.3)
    atl.atlasify("Simulation", r"$(E_{\mathrm{gen}} - E_{\mathrm{calo}}) / \sqrt{E_{\mathrm{calo}}}$", 
                 axes=ax[2], enlarge=1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_pulls.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "residuals_pulls.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved residuals_pulls.pdf")


def plot_profiles(combined, relative_residual, output_dir, config):
    """Plot profile plots: mean relative residual vs eta and pt."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Profile 1: Mean relative residual vs eta
    eta_bins = np.linspace(
        combined['eta'].min(), 
        combined['eta'].max(), 
        config['plot_params']['profile_nbins']
    )
    eta_bin_centers = []
    eta_mean_residuals = []
    eta_errors = []
    
    for i in range(len(eta_bins)-1):
        mask = (combined['eta'] >= eta_bins[i]) & (combined['eta'] < eta_bins[i+1])
        if mask.sum() > 0:
            eta_bin_centers.append((eta_bins[i] + eta_bins[i+1]) / 2)
            residuals_in_bin = relative_residual[mask]
            eta_mean_residuals.append(residuals_in_bin.mean())
            eta_errors.append(residuals_in_bin.std() / np.sqrt(len(residuals_in_bin)))
    
    ax[0].errorbar(eta_bin_centers, eta_mean_residuals, yerr=eta_errors,
                   fmt='o', color='royalblue', capsize=3, markersize=5,
                   elinewidth=1.5, markeredgewidth=1.5)
    ax[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax[0].set_xlabel(r'$\eta$')
    ax[0].set_ylabel(r'Mean $|E_{\mathrm{gen}} - E_{\mathrm{calo}}| / E_{\mathrm{gen}}$')
    ax[0].grid(True, alpha=0.3)
    atl.atlasify("Simulation", config['plot_params']['profile_label'], 
                 axes=ax[0], enlarge=1.0)
    
    # Profile 2: Mean relative residual vs pt
    pt_bins = np.logspace(
        np.log10(combined['pt'].min()), 
        np.log10(combined['pt'].max()), 
        config['plot_params']['profile_nbins']
    )
    pt_bin_centers = []
    pt_mean_residuals = []
    pt_errors = []
    
    for i in range(len(pt_bins)-1):
        mask = (combined['pt'] >= pt_bins[i]) & (combined['pt'] < pt_bins[i+1])
        if mask.sum() > 0:
            # Use geometric mean for bin center in log space
            pt_bin_centers.append(np.sqrt(pt_bins[i] * pt_bins[i+1]))
            residuals_in_bin = relative_residual[mask]
            pt_mean_residuals.append(residuals_in_bin.mean())
            pt_errors.append(residuals_in_bin.std() / np.sqrt(len(residuals_in_bin)))
    
    ax[1].errorbar(pt_bin_centers, pt_mean_residuals, yerr=pt_errors,
                   fmt='o', color='royalblue', capsize=3, markersize=5,
                   elinewidth=1.5, markeredgewidth=1.5)
    ax[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax[1].set_xlabel(r'$p_T$ [GeV]')
    ax[1].set_ylabel(r'Mean $|E_{\mathrm{gen}} - E_{\mathrm{calo}}| / E_{\mathrm{gen}}$')
    ax[1].set_xscale('log')
    ax[1].grid(True, alpha=0.3)
    atl.atlasify("Simulation", config['plot_params']['profile_label'], 
                 axes=ax[1], enlarge=1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / "profiles_eta_pt.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "profiles_eta_pt.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved profiles_eta_pt.pdf")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Plot calorimeter performance')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    particles, digi_hits = load_data(config)
    print(f"  Loaded {len(particles)} particles")
    print(f"  Loaded {len(digi_hits)} digitized calo hits")
    
    # Process data
    print("\nProcessing data...")
    combined, residual, relative_residual, pull = process_data(
        particles, digi_hits, pdg_code=config['pdg_code']
    )
    print(f"  Found {len(combined)} generator particles with calo hits")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Energy distribution
    generator_particles = particles[
        (particles["PDG"] == config['pdg_code']) & 
        (particles["created_in_simulation"] == False)
    ]
    plot_energy_distribution(generator_particles, output_dir, config)
    
    # Residuals and pulls
    plot_residuals_pulls(residual, relative_residual, pull, output_dir, config)
    
    # Profile plots
    plot_profiles(combined, relative_residual, output_dir, config)
    
    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()

