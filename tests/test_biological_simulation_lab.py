#!/usr/bin/env python3
"""
ICEBURG Biological Simulation Lab - Testing Computational Simulations
Demonstrates that biology is metrics, and metrics can be simulated computationally
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, 'src')

from iceburg.lab.biological_simulation_lab import BiologicalSimulationLab

print("=" * 80)
print("ICEBURG BIOLOGICAL SIMULATION LAB - COMPUTATIONAL SIMULATIONS")
print("=" * 80)
print()
print("Philosophy: Biology is metrics. Metrics can be simulated computationally.")
print("We can test biological hypotheses using physics/biology models.")
print()

# Create lab instance
lab = BiologicalSimulationLab()

print("Lab Status:")
status = lab.get_lab_status()
print(f"  Status: {status['status']}")
print(f"  Philosophy: {status['philosophy']}")
print(f"  Available Simulations: {status['available_simulations']}")
print()

# Test 1: Quantum Coherence in Photosynthesis
print("=" * 80)
print("TEST 1: Quantum Coherence in Photosynthesis")
print("=" * 80)
print("Model: Quantum mechanics simulation of electron transfer chains")
print("Based on: Quantum coherence theory, energy transfer efficiency")
print()

result = lab.run_simulation(
    'quantum_coherence_photosynthesis',
    {
        'num_sites': 10,
        'temperature': 77.0,  # Kelvin (cryogenic)
        'coupling_strength': 0.1,  # eV
        'dephasing_rate': 0.01  # ps^-1
    }
)

print(f"Success: {result.success}")
print(f"Execution Time: {result.execution_time:.4f}s")
print(f"Simulation Metrics:")
for key, value in result.simulation_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")
print()

# Test 2: Bioelectric Signaling
print("=" * 80)
print("TEST 2: Bioelectric Signaling")
print("=" * 80)
print("Model: Electrical circuit simulation of membrane potential")
print("Based on: Hodgkin-Huxley model, electrical circuit theory")
print()

result = lab.run_simulation(
    'bioelectric_signaling',
    {
        'time_steps': 1000,
        'dt': 0.01,  # ms
        'membrane_capacitance': 1.0,  # uF/cm^2
        'membrane_resistance': 10.0,  # kOhm·cm^2
        'resting_potential': -70.0  # mV
    }
)

print(f"Success: {result.success}")
print(f"Execution Time: {result.execution_time:.4f}s")
print(f"Simulation Metrics:")
for key, value in result.simulation_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")
print()

# Test 3: Quantum Entanglement in Biological Systems
print("=" * 80)
print("TEST 3: Quantum Entanglement in Biological Systems")
print("=" * 80)
print("Model: Quantum mechanics simulation of entangled molecular states")
print("Based on: Quantum entanglement theory, molecular physics")
print()

result = lab.run_simulation(
    'quantum_entanglement_biological',
    {
        'num_molecules': 100,
        'interaction_strength': 0.05,  # eV
        'temperature': 300.0  # Kelvin
    }
)

print(f"Success: {result.success}")
print(f"Execution Time: {result.execution_time:.4f}s")
print(f"Simulation Metrics:")
for key, value in result.simulation_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")
print()

# Test 4: Pancreatic Bioelectric-Brain Synchronization
print("=" * 80)
print("TEST 4: Pancreatic Bioelectric-Brain Synchronization")
print("=" * 80)
print("Model: Synchronized oscillators (pancreas and brain)")
print("Based on: Electrical circuit models, synchronization theory")
print()

result = lab.run_simulation(
    'pancreatic_bioelectric',
    {
        'time_steps': 2000,
        'dt': 0.01,  # ms
        'glucose_level': 100.0,  # mg/dL
        'coupling_strength': 0.1
    }
)

print(f"Success: {result.success}")
print(f"Execution Time: {result.execution_time:.4f}s")
print(f"Simulation Metrics:")
for key, value in result.simulation_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")
print()

# Statistical Analysis
print("=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)
print("Running multiple simulations to test statistical significance...")
print()

# Run multiple simulations for statistical analysis
num_runs = 10
results = []

for i in range(num_runs):
    result = lab.run_simulation(
        'quantum_coherence_photosynthesis',
        {
            'num_sites': 10,
            'temperature': 77.0,
            'coupling_strength': 0.1,
            'dephasing_rate': 0.01
        }
    )
    if result.success:
        results.append(result.simulation_metrics['measured_efficiency'])

if results:
    mean_efficiency = np.mean(results)
    std_efficiency = np.std(results)
    sem_efficiency = std_efficiency / np.sqrt(len(results))
    
    print(f"Statistical Results (n={len(results)}):")
    print(f"  Mean Efficiency: {mean_efficiency:.6f}")
    print(f"  Standard Deviation: {std_efficiency:.6f}")
    print(f"  Standard Error: {sem_efficiency:.6f}")
    print(f"  95% Confidence Interval: [{mean_efficiency - 1.96*sem_efficiency:.6f}, {mean_efficiency + 1.96*sem_efficiency:.6f}]")
    print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total Simulations Run: {lab.get_lab_status()['total_simulations_run']}")
print()
print("Key Points:")
print("1. Biology IS metrics - gene expression, protein concentrations, electrical signals")
print("2. Metrics CAN be simulated computationally using physics/biology models")
print("3. These are ACTUAL COMPUTATIONS, not LLM-generated text")
print("4. Results are based on physics/biology models, not training data")
print("5. Statistical analysis can be performed on simulation results")
print()
print("This demonstrates that:")
print("- Biology can be modeled as computational simulations")
print("- Hypotheses can be tested using physics/biology models")
print("- Results are computed from actual models, not generated text")
print("- ICEBURG's philosophy supports computational simulation of biological systems")

