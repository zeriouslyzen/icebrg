"""
Equipment Database
Comprehensive lab equipment database
"""

from typing import Any, Dict, Optional, List
import json
from pathlib import Path


class EquipmentDatabase:
    """Lab equipment database"""
    
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = Path(database_path or "data/lab/equipment_catalog.json")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.equipment: Dict[str, Dict[str, Any]] = self._load_database()
        self._initialize_default_equipment()
    
    def _load_database(self) -> Dict[str, Dict[str, Any]]:
        """Load equipment database"""
        if self.database_path.exists():
            try:
                with open(self.database_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_database(self) -> bool:
        """Save equipment database"""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.equipment, f, indent=2)
            return True
        except Exception:
            return False
    
    def _initialize_default_equipment(self):
        """Initialize default equipment catalog"""
        if self.equipment:
            return
        
        default_equipment = {
            "quantum": [
                {
                    "id": "qpu_001",
                    "name": "Quantum Processing Unit",
                    "type": "quantum_computer",
                    "qubits": 100,
                    "fidelity": 0.99,
                    "available": True
                },
                {
                    "id": "qsim_001",
                    "name": "Quantum Simulator",
                    "type": "quantum_simulator",
                    "max_qubits": 50,
                    "available": True
                }
            ],
            "particle_physics": [
                {
                    "id": "accel_001",
                    "name": "Particle Accelerator",
                    "type": "accelerator",
                    "energy": "10 TeV",
                    "available": False
                },
                {
                    "id": "det_001",
                    "name": "Particle Detector",
                    "type": "detector",
                    "sensitivity": "high",
                    "available": True
                }
            ],
            "materials_science": [
                {
                    "id": "micro_001",
                    "name": "Electron Microscope",
                    "type": "microscope",
                    "resolution": "1 nm",
                    "available": True
                },
                {
                    "id": "xray_001",
                    "name": "X-ray Diffractometer",
                    "type": "diffractometer",
                    "wavelength": "Cu K-alpha",
                    "available": True
                }
            ],
            "computational": [
                {
                    "id": "hpc_001",
                    "name": "HPC Cluster",
                    "type": "hpc_cluster",
                    "nodes": 100,
                    "cores_per_node": 64,
                    "available": True
                },
                {
                    "id": "gpu_001",
                    "name": "GPU Cluster",
                    "type": "gpu_cluster",
                    "gpus": 50,
                    "available": True
                }
            ],
            "biological": [
                {
                    "id": "seq_001",
                    "name": "DNA Sequencer",
                    "type": "sequencer",
                    "throughput": "high",
                    "available": True
                },
                {
                    "id": "cell_001",
                    "name": "Cell Culture System",
                    "type": "culture_system",
                    "capacity": 100,
                    "available": True
                }
            ]
        }
        
        self.equipment = default_equipment
        self._save_database()
    
    def get_equipment(
        self,
        category: Optional[str] = None,
        equipment_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get equipment by category and type"""
        if category:
            equipment_list = self.equipment.get(category, [])
        else:
            equipment_list = []
            for cat_equipment in self.equipment.values():
                equipment_list.extend(cat_equipment)
        
        if equipment_type:
            equipment_list = [
                eq for eq in equipment_list
                if eq.get("type") == equipment_type
            ]
        
        return equipment_list
    
    def get_available_equipment(
        self,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get available equipment"""
        equipment = self.get_equipment(category)
        return [eq for eq in equipment if eq.get("available", False)]
    
    def reserve_equipment(self, equipment_id: str) -> bool:
        """Reserve equipment"""
        for category_equipment in self.equipment.values():
            for eq in category_equipment:
                if eq.get("id") == equipment_id:
                    if eq.get("available", False):
                        eq["available"] = False
                        self._save_database()
                        return True
        return False
    
    def release_equipment(self, equipment_id: str) -> bool:
        """Release equipment"""
        for category_equipment in self.equipment.values():
            for eq in category_equipment:
                if eq.get("id") == equipment_id:
                    eq["available"] = True
                    self._save_database()
                    return True
        return False
    
    def add_equipment(
        self,
        category: str,
        equipment: Dict[str, Any]
    ) -> bool:
        """Add equipment to database"""
        if category not in self.equipment:
            self.equipment[category] = []
        
        self.equipment[category].append(equipment)
        return self._save_database()
    
    def get_categories(self) -> List[str]:
        """Get equipment categories"""
        return list(self.equipment.keys())

