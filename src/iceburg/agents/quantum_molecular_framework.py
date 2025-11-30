"""
Quantum Molecular Chemistry Framework
=====================================

Enhanced framework capturing:
- Periodic table patterns and electron configurations
- Bond energies and molecular orbitals
- Quantum chemistry (wavefunctions, electron density)
- Vibrational frequencies and energetic signatures
- Emergent properties and cooperative effects
- Resonance patterns and phase transitions
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class BondType(Enum):
    """Types of chemical bonds"""
    COVALENT = "covalent"
    POLAR_COVALENT = "polar_covalent"
    IONIC = "ionic"
    HYDROGEN = "hydrogen"
    VAN_DER_WAALS = "van_der_waals"
    METALLIC = "metallic"
    COORDINATE = "coordinate"

@dataclass
class ElementSignature:
    """Energetic signature of an element from periodic table"""
    symbol: str
    atomic_number: int
    period: int
    group: int
    electron_configuration: str
    ionization_energy: float  # eV
    electron_affinity: float  # eV
    atomic_radius: float  # pm
    electronegativity: float  # Pauling scale
    valence_electrons: int
    
    # Quantum properties
    orbital_energies: Dict[str, float]  # s, p, d, f orbital energies (eV)
    spin_orbit_coupling: float  # eV
    magnetic_moment: float  # Bohr magnetons
    
    # Energetic signature
    characteristic_frequencies: List[float]  # Hz - vibrational/electronic transitions
    resonance_frequencies: List[float]  # Hz - resonant modes
    quantum_coherence_time: float  # seconds

@dataclass
class BondEnergetics:
    """Energetic properties of chemical bonds"""
    bond_type: BondType
    bond_length: float  # Angstroms
    bond_energy: float  # kJ/mol
    bond_order: float  # 1.0 = single, 2.0 = double, etc.
    
    # Quantum properties
    overlap_integral: float  # Orbital overlap
    bond_dipole_moment: float  # Debye
    polarizability: float  # Å³
    
    # Vibrational properties
    stretching_frequency: float  # cm⁻¹ (IR spectroscopy)
    bending_frequency: float  # cm⁻¹
    force_constant: float  # N/m
    
    # Energetic signature
    resonance_frequency: float  # Hz
    quantum_coherence: float  # coherence factor 0-1

@dataclass
class MolecularOrbital:
    """Molecular orbital properties"""
    orbital_type: str  # bonding, antibonding, nonbonding
    energy: float  # eV
    symmetry: str  # σ, π, δ, etc.
    occupation: int  # number of electrons
    
    # Quantum properties
    wavefunction_coefficients: Dict[str, float]  # atomic orbital contributions
    electron_density: float  # e/Å³
    spin_density: float  # e/Å³
    
    # Energetic signature
    transition_energy: float  # eV (HOMO-LUMO gap, etc.)
    oscillator_strength: float  # transition probability

@dataclass
class MolecularEnergeticSignature:
    """Complete energetic signature of a molecule"""
    molecular_formula: str
    elements: List[ElementSignature]
    bonds: List[BondEnergetics]
    orbitals: List[MolecularOrbital]
    
    # Quantum properties
    total_energy: float  # eV
    homo_energy: float  # eV (Highest Occupied Molecular Orbital)
    lumo_energy: float  # eV (Lowest Unoccupied Molecular Orbital)
    band_gap: float  # eV (HOMO-LUMO gap)
    
    # Vibrational spectrum
    normal_modes: List[Dict[str, float]]  # frequency, intensity, symmetry
    ir_active_modes: List[float]  # cm⁻¹
    raman_active_modes: List[float]  # cm⁻¹
    
    # Electronic transitions
    electronic_transitions: List[Dict[str, float]]  # energy, oscillator_strength, symmetry
    
    # Energetic signature
    characteristic_frequencies: List[float]  # Hz
    resonance_frequencies: List[float]  # Hz
    quantum_coherence_time: float  # seconds
    coherence_factor: float  # 0-1
    
    # Emergent properties
    cooperativity_factor: float  # cooperative binding effects
    phase_transition_temperature: Optional[float]  # K
    critical_points: List[Dict[str, float]]  # phase transitions

class QuantumMolecularFramework:
    """
    Framework for quantum molecular chemistry with energetic signatures.
    Captures periodic table patterns, bond energies, molecular orbitals,
    and emergent properties.
    """
    
    def __init__(self):
        # Periodic table data (simplified - would use full database in production)
        self.elements = self._initialize_periodic_table()
        self.bond_energies = self._initialize_bond_energies()
        self.molecular_orbital_patterns = self._initialize_orbital_patterns()
    
    def _initialize_periodic_table(self) -> Dict[str, ElementSignature]:
        """Initialize periodic table with quantum properties"""
        elements = {}
        
        # Key biological elements with quantum properties
        biological_elements = {
            "H": {
                "atomic_number": 1, "period": 1, "group": 1,
                "electron_configuration": "1s¹",
                "ionization_energy": 13.598, "electron_affinity": 0.754,
                "atomic_radius": 53, "electronegativity": 2.20,
                "valence_electrons": 1,
                "orbital_energies": {"1s": -13.6},
                "spin_orbit_coupling": 0.0,
                "magnetic_moment": 1.0,
                "characteristic_frequencies": [4.4e14, 1.2e15],  # UV transitions
                "resonance_frequencies": [2.4e15],  # Lyman series
                "quantum_coherence_time": 1e-12
            },
            "C": {
                "atomic_number": 6, "period": 2, "group": 14,
                "electron_configuration": "1s²2s²2p²",
                "ionization_energy": 11.260, "electron_affinity": 1.262,
                "atomic_radius": 67, "electronegativity": 2.55,
                "valence_electrons": 4,
                "orbital_energies": {"1s": -296.2, "2s": -19.4, "2p": -10.7},
                "spin_orbit_coupling": 0.0001,
                "magnetic_moment": 0.0,
                "characteristic_frequencies": [1.0e14, 2.5e14],  # IR/visible
                "resonance_frequencies": [1.5e14],
                "quantum_coherence_time": 1e-11
            },
            "N": {
                "atomic_number": 7, "period": 2, "group": 15,
                "electron_configuration": "1s²2s²2p³",
                "ionization_energy": 14.534, "electron_affinity": -0.07,
                "atomic_radius": 56, "electronegativity": 3.04,
                "valence_electrons": 5,
                "orbital_energies": {"1s": -409.9, "2s": -25.6, "2p": -13.2},
                "spin_orbit_coupling": 0.0002,
                "magnetic_moment": 3.0,
                "characteristic_frequencies": [1.2e14, 3.0e14],
                "resonance_frequencies": [1.8e14],
                "quantum_coherence_time": 1e-11
            },
            "O": {
                "atomic_number": 8, "period": 2, "group": 16,
                "electron_configuration": "1s²2s²2p⁴",
                "ionization_energy": 13.618, "electron_affinity": 1.461,
                "atomic_radius": 48, "electronegativity": 3.44,
                "valence_electrons": 6,
                "orbital_energies": {"1s": -543.1, "2s": -32.4, "2p": -15.9},
                "spin_orbit_coupling": 0.0003,
                "magnetic_moment": 2.0,
                "characteristic_frequencies": [1.5e14, 3.5e14],
                "resonance_frequencies": [2.0e14],
                "quantum_coherence_time": 1e-11
            },
            "Na": {
                "atomic_number": 11, "period": 3, "group": 1,
                "electron_configuration": "1s²2s²2p⁶3s¹",
                "ionization_energy": 5.139, "electron_affinity": 0.548,
                "atomic_radius": 190, "electronegativity": 0.93,
                "valence_electrons": 1,
                "orbital_energies": {"1s": -1045.0, "2s": -63.4, "2p": -30.8, "3s": -5.14},
                "spin_orbit_coupling": 0.0001,
                "magnetic_moment": 1.0,
                "characteristic_frequencies": [5.9e14],  # Visible (yellow)
                "resonance_frequencies": [5.1e14],
                "quantum_coherence_time": 1e-12
            },
            "K": {
                "atomic_number": 19, "period": 4, "group": 1,
                "electron_configuration": "1s²2s²2p⁶3s²3p⁶4s¹",
                "ionization_energy": 4.341, "electron_affinity": 0.501,
                "atomic_radius": 243, "electronegativity": 0.82,
                "valence_electrons": 1,
                "orbital_energies": {"4s": -4.34},
                "spin_orbit_coupling": 0.0001,
                "magnetic_moment": 1.0,
                "characteristic_frequencies": [4.3e14],  # Visible (violet)
                "resonance_frequencies": [3.8e14],
                "quantum_coherence_time": 1e-12
            },
            "Ca": {
                "atomic_number": 20, "period": 4, "group": 2,
                "electron_configuration": "1s²2s²2p⁶3s²3p⁶4s²",
                "ionization_energy": 6.113, "electron_affinity": 0.024,
                "atomic_radius": 197, "electronegativity": 1.00,
                "valence_electrons": 2,
                "orbital_energies": {"4s": -6.11},
                "spin_orbit_coupling": 0.0001,
                "magnetic_moment": 0.0,
                "characteristic_frequencies": [4.2e14],
                "resonance_frequencies": [3.9e14],
                "quantum_coherence_time": 1e-12
            },
            "Cl": {
                "atomic_number": 17, "period": 3, "group": 17,
                "electron_configuration": "1s²2s²2p⁶3s²3p⁵",
                "ionization_energy": 12.968, "electron_affinity": 3.613,
                "atomic_radius": 99, "electronegativity": 3.16,
                "valence_electrons": 7,
                "orbital_energies": {"3p": -13.0},
                "spin_orbit_coupling": 0.0005,
                "magnetic_moment": 1.0,
                "characteristic_frequencies": [1.8e14],
                "resonance_frequencies": [1.5e14],
                "quantum_coherence_time": 1e-11
            }
        }
        
        for symbol, props in biological_elements.items():
            elements[symbol] = ElementSignature(
                symbol=symbol,
                atomic_number=props["atomic_number"],
                period=props["period"],
                group=props["group"],
                electron_configuration=props["electron_configuration"],
                ionization_energy=props["ionization_energy"],
                electron_affinity=props["electron_affinity"],
                atomic_radius=props["atomic_radius"],
                electronegativity=props["electronegativity"],
                valence_electrons=props["valence_electrons"],
                orbital_energies=props["orbital_energies"],
                spin_orbit_coupling=props["spin_orbit_coupling"],
                magnetic_moment=props["magnetic_moment"],
                characteristic_frequencies=props["characteristic_frequencies"],
                resonance_frequencies=props["resonance_frequencies"],
                quantum_coherence_time=props["quantum_coherence_time"]
            )
        
        return elements
    
    def _initialize_bond_energies(self) -> Dict[str, float]:
        """Initialize bond energy database (kJ/mol)"""
        return {
            # Single bonds
            "H-H": 436,
            "C-C": 347,
            "C-H": 413,
            "C-N": 305,
            "C-O": 358,
            "C-Cl": 339,
            "N-H": 391,
            "O-H": 463,
            "O-O": 146,
            "N-N": 163,
            # Double bonds
            "C=C": 614,
            "C=O": 799,
            "C=N": 615,
            "N=O": 607,
            # Triple bonds
            "C≡C": 839,
            "C≡N": 891,
            "N≡N": 945,
            # Ionic/coordinate
            "Na-Cl": 411,  # Ionic
            "K-Cl": 423,   # Ionic
            "Ca-O": 464,   # Ionic
            # Hydrogen bonds
            "H...O": 20,   # Hydrogen bond (weaker)
            "H...N": 18,
        }
    
    def _initialize_orbital_patterns(self) -> Dict[str, Dict[str, float]]:
        """Initialize molecular orbital patterns"""
        return {
            "sigma_bonding": {
                "overlap_integral": 0.7,
                "bond_order": 1.0,
                "symmetry": "σ"
            },
            "pi_bonding": {
                "overlap_integral": 0.3,
                "bond_order": 1.0,
                "symmetry": "π"
            },
            "sigma_antibonding": {
                "overlap_integral": -0.3,
                "bond_order": -1.0,
                "symmetry": "σ*"
            },
            "pi_antibonding": {
                "overlap_integral": -0.1,
                "bond_order": -1.0,
                "symmetry": "π*"
            }
        }
    
    def calculate_molecular_energetic_signature(
        self,
        molecular_formula: str,
        celestial_em_environment: Dict[str, float]
    ) -> MolecularEnergeticSignature:
        """
        Calculate complete energetic signature of a molecule,
        including quantum properties and emergent patterns.
        """
        # Parse molecular formula (simplified - would use proper parser)
        elements_in_molecule = self._parse_molecular_formula(molecular_formula)
        
        # Get element signatures
        element_signatures = [self.elements[symbol] for symbol in elements_in_molecule if symbol in self.elements]
        
        # Calculate bonds
        bonds = self._calculate_bonds(element_signatures, molecular_formula)
        
        # Calculate molecular orbitals
        orbitals = self._calculate_molecular_orbitals(element_signatures, bonds)
        
        # Calculate vibrational spectrum
        normal_modes = self._calculate_vibrational_modes(bonds, element_signatures)
        
        # Calculate electronic transitions
        electronic_transitions = self._calculate_electronic_transitions(orbitals)
        
        # Calculate energetic signature (affected by celestial EM)
        characteristic_frequencies = self._calculate_characteristic_frequencies(
            element_signatures, bonds, celestial_em_environment
        )
        
        resonance_frequencies = self._calculate_resonance_frequencies(
            characteristic_frequencies, celestial_em_environment
        )
        
        # Calculate quantum coherence
        quantum_coherence_time = self._calculate_quantum_coherence(
            element_signatures, bonds, celestial_em_environment
        )
        
        # Calculate emergent properties
        cooperativity_factor = self._calculate_cooperativity(bonds, celestial_em_environment)
        
        # Calculate HOMO-LUMO gap
        homo_energy = max([orb.energy for orb in orbitals if orb.occupation > 0], default=0.0)
        lumo_energy = min([orb.energy for orb in orbitals if orb.occupation == 0], default=0.0)
        band_gap = lumo_energy - homo_energy if lumo_energy > homo_energy else 0.0
        
        return MolecularEnergeticSignature(
            molecular_formula=molecular_formula,
            elements=element_signatures,
            bonds=bonds,
            orbitals=orbitals,
            total_energy=sum([orb.energy * orb.occupation for orb in orbitals]),
            homo_energy=homo_energy,
            lumo_energy=lumo_energy,
            band_gap=band_gap,
            normal_modes=normal_modes,
            ir_active_modes=[mode["frequency"] for mode in normal_modes if mode.get("ir_active", False)],
            raman_active_modes=[mode["frequency"] for mode in normal_modes if mode.get("raman_active", False)],
            electronic_transitions=electronic_transitions,
            characteristic_frequencies=characteristic_frequencies,
            resonance_frequencies=resonance_frequencies,
            quantum_coherence_time=quantum_coherence_time,
            coherence_factor=min(1.0, quantum_coherence_time / 1e-10),  # Normalized
            cooperativity_factor=cooperativity_factor,
            phase_transition_temperature=None,  # Would calculate for phase transitions
            critical_points=[]
        )
    
    def _parse_molecular_formula(self, formula: str) -> List[str]:
        """Parse molecular formula into element symbols (simplified)"""
        # Very simplified - would use proper chemical formula parser
        elements = []
        i = 0
        while i < len(formula):
            if formula[i].isupper():
                symbol = formula[i]
                if i + 1 < len(formula) and formula[i+1].islower():
                    symbol += formula[i+1]
                    i += 2
                else:
                    i += 1
                elements.append(symbol)
            else:
                i += 1
        return elements
    
    def _calculate_bonds(
        self,
        elements: List[ElementSignature],
        formula: str
    ) -> List[BondEnergetics]:
        """Calculate bond energetics from elements"""
        bonds = []
        
        # Simplified bond calculation
        # In production, would use proper molecular structure
        for i, elem1 in enumerate(elements):
            for elem2 in elements[i+1:]:
                # Calculate bond type based on electronegativity difference
                delta_en = abs(elem1.electronegativity - elem2.electronegativity)
                
                if delta_en < 0.5:
                    bond_type = BondType.COVALENT
                elif delta_en < 1.7:
                    bond_type = BondType.POLAR_COVALENT
                else:
                    bond_type = BondType.IONIC
                
                # Bond length (sum of atomic radii with correction)
                bond_length = (elem1.atomic_radius + elem2.atomic_radius) / 100.0  # Convert to Angstroms
                
                # Bond energy (lookup or calculate)
                bond_key = f"{elem1.symbol}-{elem2.symbol}"
                bond_energy = self.bond_energies.get(bond_key, 300.0)  # Default 300 kJ/mol
                
                # Vibrational frequency (from bond energy and reduced mass)
                reduced_mass = (elem1.atomic_number * elem2.atomic_number) / (elem1.atomic_number + elem2.atomic_number)
                force_constant = bond_energy * 1000 / (bond_length ** 2)  # Approximate
                stretching_frequency = (1 / (2 * math.pi)) * math.sqrt(force_constant / (reduced_mass * 1.66e-27))  # Hz
                stretching_frequency_cm = stretching_frequency / (3e10)  # Convert to cm⁻¹
                
                bonds.append(BondEnergetics(
                    bond_type=bond_type,
                    bond_length=bond_length,
                    bond_energy=bond_energy,
                    bond_order=1.0,
                    overlap_integral=0.7 if bond_type == BondType.COVALENT else 0.3,
                    bond_dipole_moment=delta_en * 1.0,  # Debye (approximate)
                    polarizability=(elem1.atomic_radius + elem2.atomic_radius) ** 3 / 1000.0,
                    stretching_frequency=stretching_frequency_cm,
                    bending_frequency=stretching_frequency_cm * 0.5,
                    force_constant=force_constant,
                    resonance_frequency=stretching_frequency,
                    quantum_coherence=0.8 if bond_type == BondType.COVALENT else 0.5
                ))
        
        return bonds
    
    def _calculate_molecular_orbitals(
        self,
        elements: List[ElementSignature],
        bonds: List[BondEnergetics]
    ) -> List[MolecularOrbital]:
        """Calculate molecular orbitals from atomic orbitals"""
        orbitals = []
        
        # Simplified MO calculation
        # In production, would use proper quantum chemistry (Hartree-Fock, DFT, etc.)
        for i, elem in enumerate(elements):
            for orbital_type, energy in elem.orbital_energies.items():
                # Create bonding orbital
                orbitals.append(MolecularOrbital(
                    orbital_type="bonding",
                    energy=energy - 2.0,  # Stabilized by bonding
                    symmetry="σ" if "s" in orbital_type else "π",
                    occupation=2 if energy < -10 else 0,  # Simplified
                    wavefunction_coefficients={elem.symbol: 1.0},
                    electron_density=1.0,
                    spin_density=0.0,
                    transition_energy=abs(energy) + 2.0,
                    oscillator_strength=0.1
                ))
                
                # Create antibonding orbital
                orbitals.append(MolecularOrbital(
                    orbital_type="antibonding",
                    energy=energy + 2.0,  # Destabilized
                    symmetry="σ*" if "s" in orbital_type else "π*",
                    occupation=0,
                    wavefunction_coefficients={elem.symbol: 1.0},
                    electron_density=0.1,
                    spin_density=0.0,
                    transition_energy=abs(energy) + 4.0,
                    oscillator_strength=0.05
                ))
        
        return sorted(orbitals, key=lambda x: x.energy)
    
    def _calculate_vibrational_modes(
        self,
        bonds: List[BondEnergetics],
        elements: List[ElementSignature]
    ) -> List[Dict[str, float]]:
        """Calculate normal modes of vibration"""
        modes = []
        
        for bond in bonds:
            # Stretching mode
            modes.append({
                "frequency": bond.stretching_frequency,
                "intensity": 1.0,
                "symmetry": "A1",
                "ir_active": True,
                "raman_active": True,
                "mode_type": "stretching"
            })
            
            # Bending mode
            modes.append({
                "frequency": bond.bending_frequency,
                "intensity": 0.5,
                "symmetry": "E",
                "ir_active": True,
                "raman_active": False,
                "mode_type": "bending"
            })
        
        return modes
    
    def _calculate_electronic_transitions(
        self,
        orbitals: List[MolecularOrbital]
    ) -> List[Dict[str, float]]:
        """Calculate electronic transitions"""
        transitions = []
        
        occupied = [orb for orb in orbitals if orb.occupation > 0]
        unoccupied = [orb for orb in orbitals if orb.occupation == 0]
        
        for occ in occupied:
            for unocc in unoccupied:
                transition_energy = unocc.energy - occ.energy
                if transition_energy > 0:
                    transitions.append({
                        "energy": transition_energy,
                        "oscillator_strength": occ.oscillator_strength * unocc.oscillator_strength,
                        "symmetry": f"{occ.symmetry} → {unocc.symmetry}",
                        "from_orbital": occ.orbital_type,
                        "to_orbital": unocc.orbital_type
                    })
        
        return sorted(transitions, key=lambda x: x["energy"])
    
    def _calculate_characteristic_frequencies(
        self,
        elements: List[ElementSignature],
        bonds: List[BondEnergetics],
        em_environment: Dict[str, float]
    ) -> List[float]:
        """Calculate characteristic frequencies (affected by EM environment)"""
        frequencies = []
        
        # Element characteristic frequencies
        for elem in elements:
            for freq in elem.characteristic_frequencies:
                # Modulate by EM environment
                em_modulation = 1.0 + em_environment.get("geomagnetic_field_strength", 0.3) * 0.01
                frequencies.append(freq * em_modulation)
        
        # Bond vibrational frequencies
        for bond in bonds:
            # Convert cm⁻¹ to Hz
            freq_hz = bond.stretching_frequency * 3e10
            em_modulation = 1.0 + em_environment.get("schumann_resonance_fundamental", 7.83) / 1e6
            frequencies.append(freq_hz * em_modulation)
        
        return sorted(frequencies)
    
    def _calculate_resonance_frequencies(
        self,
        characteristic_frequencies: List[float],
        em_environment: Dict[str, float]
    ) -> List[float]:
        """Calculate resonance frequencies (where EM fields couple)"""
        resonances = []
        
        schumann_fundamental = em_environment.get("schumann_resonance_fundamental", 7.83)
        schumann_harmonics = [schumann_fundamental * n for n in [1, 2, 3, 4, 5]]
        
        for char_freq in characteristic_frequencies:
            for schumann in schumann_harmonics:
                # Find resonances (when frequencies match or are harmonics)
                if abs(char_freq - schumann) / schumann < 0.1:  # Within 10%
                    resonances.append((char_freq + schumann) / 2)  # Average
                # Check for harmonic relationships
                for n in [2, 3, 4]:
                    if abs(char_freq - schumann * n) / (schumann * n) < 0.1:
                        resonances.append(char_freq)
        
        return sorted(set(resonances))
    
    def _calculate_quantum_coherence(
        self,
        elements: List[ElementSignature],
        bonds: List[BondEnergetics],
        em_environment: Dict[str, float]
    ) -> float:
        """Calculate quantum coherence time (affected by EM environment)"""
        # Base coherence from elements
        base_coherence = min([elem.quantum_coherence_time for elem in elements], default=1e-12)
        
        # Enhance by covalent bonds
        covalent_bonds = [b for b in bonds if b.bond_type == BondType.COVALENT]
        if covalent_bonds:
            bond_coherence = max([b.quantum_coherence for b in covalent_bonds])
            base_coherence *= (1.0 + bond_coherence)
        
        # Modulate by EM environment (can enhance or destroy coherence)
        em_strength = em_environment.get("geomagnetic_field_strength", 0.3)
        em_modulation = 1.0 + em_strength * 0.1  # Can enhance coherence
        
        # Schumann resonance can enhance coherence
        schumann = em_environment.get("schumann_resonance_fundamental", 7.83)
        schumann_enhancement = 1.0 + (schumann / 100.0) * 0.01
        
        return base_coherence * em_modulation * schumann_enhancement
    
    def _calculate_cooperativity(
        self,
        bonds: List[BondEnergetics],
        em_environment: Dict[str, float]
    ) -> float:
        """Calculate cooperativity factor (emergent property)"""
        # Cooperativity increases with number of bonds
        n_bonds = len(bonds)
        base_cooperativity = 1.0 + (n_bonds - 1) * 0.1
        
        # EM environment can enhance cooperativity
        em_strength = sum([v for k, v in em_environment.items() if "electromagnetic_influence" in k])
        em_enhancement = 1.0 + em_strength * 0.05
        
        return base_cooperativity * em_enhancement

