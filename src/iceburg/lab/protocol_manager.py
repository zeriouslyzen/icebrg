"""
Protocol Manager
Scientific protocol management and experiment design
"""

from typing import Any, Dict, Optional, List
import json
from pathlib import Path
from datetime import datetime


class ProtocolManager:
    """Manages scientific protocols and experiments"""
    
    def __init__(self, protocols_dir: Optional[str] = None):
        self.protocols_dir = Path(protocols_dir or "data/lab/protocols")
        self.protocols_dir.mkdir(parents=True, exist_ok=True)
        self.protocols: Dict[str, Dict[str, Any]] = {}
        self._load_protocols()
        self._initialize_default_protocols()
    
    def _load_protocols(self):
        """Load protocols from directory"""
        for protocol_file in self.protocols_dir.glob("*.json"):
            try:
                with open(protocol_file, 'r') as f:
                    protocol = json.load(f)
                    protocol_id = protocol.get("id")
                    if protocol_id:
                        self.protocols[protocol_id] = protocol
            except Exception:
                pass
    
    def _initialize_default_protocols(self):
        """Initialize default protocols"""
        if self.protocols:
            return
        
        default_protocols = {
            "pcr": {
                "id": "pcr",
                "name": "Polymerase Chain Reaction",
                "type": "molecular_biology",
                "steps": [
                    {"step": 1, "name": "Denaturation", "temperature": 95, "time": 30},
                    {"step": 2, "name": "Annealing", "temperature": 55, "time": 30},
                    {"step": 3, "name": "Extension", "temperature": 72, "time": 60}
                ],
                "cycles": 30
            },
            "western_blot": {
                "id": "western_blot",
                "name": "Western Blot",
                "type": "protein_analysis",
                "steps": [
                    {"step": 1, "name": "Gel Electrophoresis", "time": 60},
                    {"step": 2, "name": "Transfer", "time": 90},
                    {"step": 3, "name": "Blocking", "time": 60},
                    {"step": 4, "name": "Primary Antibody", "time": 120},
                    {"step": 5, "name": "Secondary Antibody", "time": 60},
                    {"step": 6, "name": "Detection", "time": 30}
                ]
            },
            "quantum_measurement": {
                "id": "quantum_measurement",
                "name": "Quantum State Measurement",
                "type": "quantum_physics",
                "steps": [
                    {"step": 1, "name": "State Preparation", "time": 10},
                    {"step": 2, "name": "Quantum Operation", "time": 5},
                    {"step": 3, "name": "Measurement", "time": 1},
                    {"step": 4, "name": "Data Collection", "time": 5}
                ],
                "shots": 1000
            }
        }
        
        for protocol_id, protocol in default_protocols.items():
            self.protocols[protocol_id] = protocol
            self._save_protocol(protocol_id, protocol)
    
    def _save_protocol(self, protocol_id: str, protocol: Dict[str, Any]):
        """Save protocol to file"""
        protocol_file = self.protocols_dir / f"{protocol_id}.json"
        try:
            with open(protocol_file, 'w') as f:
                json.dump(protocol, f, indent=2)
        except Exception:
            pass
    
    def get_protocol(self, protocol_id: str) -> Optional[Dict[str, Any]]:
        """Get protocol by ID"""
        return self.protocols.get(protocol_id)
    
    def list_protocols(self, protocol_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List protocols"""
        if protocol_type:
            return [
                p for p in self.protocols.values()
                if p.get("type") == protocol_type
            ]
        return list(self.protocols.values())
    
    def create_experiment(
        self,
        protocol_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create experiment from protocol"""
        protocol = self.get_protocol(protocol_id)
        if not protocol:
            return {"error": f"Protocol {protocol_id} not found"}
        
        experiment = {
            "id": f"exp_{int(datetime.now().timestamp())}",
            "protocol_id": protocol_id,
            "protocol_name": protocol.get("name"),
            "parameters": parameters or {},
            "steps": protocol.get("steps", []).copy(),
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
        
        return experiment
    
    def execute_experiment(
        self,
        experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute experiment"""
        experiment["status"] = "running"
        experiment["started_at"] = datetime.now().isoformat()
        
        # Execute steps
        results = []
        for step in experiment.get("steps", []):
            step_result = self._execute_step(step, experiment.get("parameters", {}))
            results.append(step_result)
        
        experiment["status"] = "completed"
        experiment["completed_at"] = datetime.now().isoformat()
        experiment["results"] = results
        
        return experiment
    
    def _execute_step(
        self,
        step: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single step"""
        return {
            "step": step.get("step"),
            "name": step.get("name"),
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_protocol(self, protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Validate protocol"""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = ["id", "name", "type", "steps"]
        for field in required_fields:
            if field not in protocol:
                validation["valid"] = False
                validation["errors"].append(f"Missing required field: {field}")
        
        # Validate steps
        if "steps" in protocol:
            for i, step in enumerate(protocol["steps"]):
                if "name" not in step:
                    validation["valid"] = False
                    validation["errors"].append(f"Step {i+1} missing name")
        
        return validation
    
    def add_protocol(self, protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Add new protocol"""
        validation = self.validate_protocol(protocol)
        
        if not validation["valid"]:
            return {
                "success": False,
                "validation": validation
            }
        
        protocol_id = protocol.get("id")
        if protocol_id:
            self.protocols[protocol_id] = protocol
            self._save_protocol(protocol_id, protocol)
            return {
                "success": True,
                "protocol_id": protocol_id
            }
        
        return {
            "success": False,
            "error": "Protocol missing ID"
        }

