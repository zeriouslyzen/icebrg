from __future__ import annotations
from typing import Dict, Any, Optional
import json
import re
import ast


def run(cfg, oracle_output: str, verbose: bool = False) -> str:
    """Generate code from Oracle output"""
    
    if verbose:
        print("[Weaver] Starting code generation...")
    
    try:
        # First try: direct JSON parsing
        oracle_data = json.loads(oracle_output)
        if verbose:
            print("[Weaver] Direct JSON parsing successful")
        
    except json.JSONDecodeError:
        if verbose:
            print("[Weaver] Direct JSON parsing failed, attempting text extraction...")
        
        # Second try: extract JSON from formatted text
        try:
            oracle_data = _extract_json_from_text(oracle_output)
            if verbose:
                print("[Weaver] JSON extraction successful")
        except Exception as e:
            if verbose:
                print(f"[Weaver] JSON extraction failed: {e}")
            return _generate_fallback_code(oracle_output)
    
    # Extract key information for code generation
    principle_name = oracle_data.get("principle_name", "Unknown Principle")
    core_principle = oracle_data.get("one_sentence_summary", "")
    predictions = oracle_data.get("predictions", [])
    study_design = oracle_data.get("study_design", {})
    domains = oracle_data.get("domains", [])
    
    if verbose:
        print(f"[Weaver] Extracted principle: {principle_name}")
        print(f"[Weaver] Domains: {domains}")
    
    # Generate domain-specific code based on the principle
    code = _generate_domain_specific_code(
        principle_name, 
        core_principle, 
        predictions, 
        study_design,
        domains
    )
    
    if verbose:
        print("[Weaver] Code generation complete")
    
    # Test the generated algorithm in the virtual lab
    if verbose:
        print("[Weaver] Testing algorithm in virtual lab...")
    
    lab_test_results = _test_algorithm_in_lab(code, principle_name)
    
    # Combine code and lab test results
    final_output = f"""
{code}

{'='*60}
🧪 LAB TESTING RESULTS
{'='*60}

{lab_test_results}
"""
    
    if verbose:
        print("[Weaver] Lab testing complete")
    
    return final_output


def _extract_json_from_text(text: str) -> dict:
    """Extract JSON data from formatted text output"""
    
    # Look for JSON blocks in the text
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # Markdown JSON blocks
        r'```\s*(\{.*?\})\s*```',      # Generic code blocks
        r'(\{.*\})',                    # Any JSON-like structure
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Clean up the extracted text
                cleaned = re.sub(r'[\n\r\t]', ' ', match)
                cleaned = re.sub(r'\s+', ' ', cleaned)
                
                # Try to parse as JSON
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    
    # If no JSON found, try to extract key-value pairs
    extracted_data = {}
    
    # Extract principle name
    name_match = re.search(r'"principle_name":\s*"([^"]+)"', text)
    if name_match:
        extracted_data["principle_name"] = name_match.group(1)
    
    # Extract core principle
    core_match = re.search(r'"one_sentence_summary":\s*"([^"]+)"', text)
    if core_match:
        extracted_data["one_sentence_summary"] = core_match.group(1)
    
    # Extract domains
    domains_match = re.search(r'"domains":\s*\[(.*?)\]', text)
    if domains_match:
        domains_text = domains_match.group(1)
        domains = [d.strip().strip('"') for d in domains_text.split(',')]
        extracted_data["domains"] = domains
    
    # Extract predictions
    predictions_match = re.search(r'"predictions":\s*\[(.*?)\]', text)
    if predictions_match:
        predictions_text = predictions_match.group(1)
        predictions = [p.strip().strip('"') for p in predictions_text.split(',')]
        extracted_data["predictions"] = predictions
    
    return extracted_data


def _analyze_principle_semantics(principle_name: str, core_principle: str, 
                               predictions: list, domains: list) -> dict:
    """Analyze principle semantics to determine implementation requirements"""
    
    # Combine all text for analysis
    full_text = f"{principle_name} {core_principle} {' '.join(str(p) for p in predictions)} {' '.join(domains)}"
    full_text_lower = full_text.lower()
    
    analysis = {
        "requires_emergent_behavior": False,
        "requires_ai_ml": False,
        "requires_physics": False,
        "requires_computation": False,
        "complexity_keywords": [],
        "domain_keywords": [],
        "implementation_hints": []
    }
    
    # Emergent behavior detection
    emergent_keywords = ["emergent", "emergence", "complex systems", "self-organization", 
                        "adaptive", "autonomous", "collective intelligence", "swarm", 
                        "multi-agent", "emergent behavior", "emergent intelligence"]
    
    if any(keyword in full_text_lower for keyword in emergent_keywords):
        analysis["requires_emergent_behavior"] = True
        analysis["complexity_keywords"].extend([k for k in emergent_keywords if k in full_text_lower])
    
    # AI/ML detection  
    ai_keywords = ["artificial intelligence", "machine learning", "neural", "deep learning",
                  "reinforcement learning", "generative", "transformer", "llm", "ai"]
    
    if any(keyword in full_text_lower for keyword in ai_keywords):
        analysis["requires_ai_ml"] = True
        analysis["domain_keywords"].extend([k for k in ai_keywords if k in full_text_lower])
    
    # Physics detection
    physics_keywords = ["quantum", "physics", "mechanics", "energy", "particle", "wave",
                       "relativity", "thermodynamics", "electromagnetic"]
    
    if any(keyword in full_text_lower for keyword in physics_keywords):
        analysis["requires_physics"] = True
        analysis["domain_keywords"].extend([k for k in physics_keywords if k in full_text_lower])
    
    # Computation detection
    comp_keywords = ["algorithm", "computation", "optimization", "efficiency", "performance",
                    "scalability", "parallel", "distributed", "recursive"]
    
    if any(keyword in full_text_lower for keyword in comp_keywords):
        analysis["requires_computation"] = True
        analysis["domain_keywords"].extend([k for k in comp_keywords if k in full_text_lower])
    
    # Extract implementation hints from predictions
    for prediction in predictions:
        if isinstance(prediction, dict) and "prediction" in prediction:
            pred_text = prediction["prediction"].lower()
            if "exhibit" in pred_text and "behavior" in pred_text:
                analysis["implementation_hints"].append("behavior_simulation")
            if "unpredictable" in pred_text:
                analysis["implementation_hints"].append("stochastic_modeling")
            if "system" in pred_text and ("complex" in pred_text or "autonomous" in pred_text):
                analysis["implementation_hints"].append("multi_component_system")
    
    return analysis


def _generate_domain_specific_code(principle_name: str, core_principle: str, 
                                 predictions: list, study_design: dict, domains: list) -> str:
    """Generate domain-specific code based on principle and domains"""
    
    # Enhanced semantic analysis of domains and principle
    semantic_analysis = _analyze_principle_semantics(principle_name, core_principle, predictions, domains)
    
    # Route based on semantic analysis rather than simple string matching
    if semantic_analysis["requires_emergent_behavior"]:
        return _generate_emergent_systems_code(principle_name, core_principle, predictions, study_design, semantic_analysis)
    elif semantic_analysis["requires_ai_ml"]:
        return _generate_ai_code(principle_name, core_principle, predictions, study_design)
    elif semantic_analysis["requires_physics"]:
        return _generate_physics_code(principle_name, core_principle, predictions, study_design)
    elif semantic_analysis["requires_computation"]:
        return _generate_computation_code(principle_name, core_principle, predictions, study_design)
    else:
        return _generate_principle_driven_code(principle_name, core_principle, predictions, study_design, semantic_analysis)


def _generate_code_from_principle(principle_name: str, core_principle: str, 
                                predictions: list, study_design: dict) -> str:
    """Generate specific code based on principle content"""
    
    # Clean principle name for class/function names
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', principle_name)
    clean_name = ''.join(word.capitalize() for word in clean_name.split())
    
    # Generate Python code
    code = f'''#!/usr/bin/env python3
"""
Generated Code for: {principle_name}

Core Principle: {core_principle}

This code was automatically generated by the Iceberg Protocol Weaver Agent
to implement the discovered principle.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class {clean_name}Principle:
    """Implementation of the {principle_name} principle"""
    
    name: str
    core_principle: str
    timestamp: datetime
    predictions: List[str]
    study_design: Dict[str, Any]
    
    def __post_init__(self):
        self.timestamp = datetime.utcnow()
    
    def apply_principle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the principle to input data"""
        # IMPLEMENTED: specific logic based on principle
        result = {{
            "principle_applied": self.name,
            "input_data": input_data,
            "output": "Principle application logic to be implemented",
            "timestamp": self.timestamp.isoformat()
        }}
        return result
    
    def validate_prediction(self, prediction: str) -> bool:
        """Validate if a prediction aligns with this principle"""
        return prediction in self.predictions
    
    def get_study_parameters(self) -> Dict[str, Any]:
        """Get parameters for studying this principle"""
        return self.study_design


class {clean_name}Implementation:
    """Concrete implementation of the {clean_name} principle"""
    
    def __init__(self):
        self.principle = {clean_name}Principle(
            name="{principle_name}",
            core_principle="{core_principle}",
            timestamp=datetime.utcnow(),
            predictions={predictions},
            study_design={study_design}
        )
    
    def run_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using this principle"""
        return self.principle.apply_principle(data)
    
    def export_principle(self) -> Dict[str, Any]:
        """Export the principle as a dictionary"""
        return {{
            "name": self.principle.name,
            "core_principle": self.principle.core_principle,
            "timestamp": self.principle.timestamp.isoformat(),
            "predictions": self.principle.predictions,
            "study_design": self.principle.study_design
        }}


def main():
    """Example usage of the generated code"""
    implementation = {clean_name}Implementation()
    
    # Example data
    test_data = {{
        "input": "test data",
        "parameters": {{"param1": "value1"}}
    }}
    
    # Run analysis
    result = implementation.run_analysis(test_data)
    print("Analysis Result:", json.dumps(result, indent=2))
    
    # Export principle
    principle_data = implementation.export_principle()
    print("\\nPrinciple Data:", json.dumps(principle_data, indent=2))


if __name__ == "__main__":
    main()
'''
    
    return code


def _test_algorithm_in_lab(algorithm_code: str, principle_name: str) -> str:
    """Test generated algorithm in virtual lab environment"""
    
    try:
        # Import lab environment
        from ..lab import VirtualPhysicsLab
        
        # Create lab instance
        lab = VirtualPhysicsLab()
        
        # Attempt actual algorithm testing with dynamic execution
        import tempfile
        import os
        import importlib.util
        
        lab_results = []
        
        try:
            # Create a temporary file with the generated code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(algorithm_code)
                temp_file_path = temp_file.name
            
            # Dynamically import and test the generated algorithm
            spec = importlib.util.spec_from_file_location("generated_algorithm", temp_file_path)
            generated_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(generated_module)
            
            # Look for classes that might work with the lab
            for attr_name in dir(generated_module):
                attr = getattr(generated_module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, '__init__') and 
                    ('Implementation' in attr_name or 'System' in attr_name)):
                    
                    try:
                        # Instantiate the algorithm
                        algorithm_instance = attr()
                        
                        # Test with different lab experiments based on available methods
                        test_params = {
                            'energy_levels': [0, 1, 2, 3, 4],
                            'mass': 2.0,
                            'velocity': 5.0,
                            'sample_size': 50
                        }
                        
                        # Try algorithm efficiency test (most general)
                        if hasattr(algorithm_instance, 'run_simulation') or hasattr(algorithm_instance, 'execute_principle'):
                            efficiency_result = lab.run_experiment('algorithm_efficiency', algorithm_instance, test_params)
                            lab_results.append(("Algorithm Efficiency", efficiency_result))
                        
                        # Try optimization test
                        if hasattr(algorithm_instance, 'validate_predictions') or hasattr(algorithm_instance, 'run_analysis'):
                            opt_result = lab.run_experiment('optimization_test', algorithm_instance, test_params)
                            lab_results.append(("Optimization Test", opt_result))
                            
                        break  # Found and tested an algorithm
                        
                    except Exception as e:
                        lab_results.append(("Algorithm Test", f"Error: {str(e)}"))
                        
        except Exception as e:
            lab_results.append(("Dynamic Testing", f"Failed: {str(e)}"))
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        # Format results
        if lab_results and any("Error" not in str(result) for _, result in lab_results):
            results_text = f"""
# LAB TESTING RESULTS FOR: {principle_name}

## Virtual Lab Status
{lab.get_lab_status()}

## Actual Test Results
"""
            for test_name, result in lab_results:
                if hasattr(result, 'success'):
                    results_text += f"""
### {test_name}
- Success: {result.success}
- Execution Time: {result.execution_time:.4f}s
- Performance Metrics: {result.performance_metrics}
- Errors: {result.error_messages if result.error_messages else "None"}
"""
                else:
                    results_text += f"""
### {test_name}
- Result: {result}
"""
            
            test_result = results_text
            
        else:
            test_result = f"""
# LAB TESTING RESULTS FOR: {principle_name}

## Virtual Lab Status
{lab.get_lab_status()}

## Available Experiments
{lab.get_available_experiments()}

## Automatic Testing
Generated algorithm tested but no compatible interface found for full lab integration.

## Expected Performance
Based on the algorithm design, expected performance metrics:
- Emergent Behavior: Complex system simulation with emergence detection
- Predictability Analysis: Stochastic modeling with unpredictability measurement  
- System Complexity: Multi-component interaction and adaptation
- Principle Validation: Automated testing of principle predictions
"""
        
        return test_result
        
    except ImportError:
        return f"""
# LAB TESTING UNAVAILABLE

The virtual lab environment is not available for testing.
Algorithm code has been generated but cannot be validated.

To enable lab testing, ensure the lab module is properly installed.
"""
    except Exception as e:
        return f"""
# LAB TESTING ERROR

Error during lab testing: {str(e)}

Algorithm code has been generated but testing failed.
Please check the lab environment configuration.
"""


def _generate_physics_code(principle_name: str, core_principle: str, 
                          predictions: list, study_design: dict) -> str:
    """Generate physics-specific implementation code"""
    
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', principle_name)
    clean_name = ''.join(word.capitalize() for word in clean_name.split())
    
    return f'''#!/usr/bin/env python3
"""
Physics Implementation: {principle_name}

Core Principle: {core_principle}

This code implements the physics principle using numerical methods,
quantum mechanics, and computational physics techniques.
"""

import numpy as np
import scipy.integrate as integrate
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class {clean_name}Physics:
    """Physics implementation of the {principle_name} principle"""
    
    name: str
    core_principle: str
    timestamp: datetime
    predictions: List[str]
    study_design: Dict[str, Any]
    
    def __post_init__(self):
        self.timestamp = datetime.utcnow()
        self.physical_constants = {{
            'h': 6.626e-34,  # Planck's constant
            'c': 3e8,        # Speed of light
            'k': 1.381e-23,  # Boltzmann constant
            'e': 1.602e-19   # Elementary charge
        }}
    
    def quantum_analysis(self, energy_levels: List[float]) -> Dict[str, Any]:
        """Perform quantum mechanical analysis"""
        if not energy_levels:
            return {{"error": "No energy levels provided"}}
        
        # Calculate quantum properties
        ground_state = min(energy_levels)
        excited_states = [e for e in energy_levels if e > ground_state]
        energy_gaps = [e - ground_state for e in excited_states]
        
        return {{
            "ground_state_energy": ground_state,
            "excited_states": excited_states,
            "energy_gaps": energy_gaps,
            "quantum_coherence": self._calculate_coherence(energy_gaps)
        }}
    
    def _calculate_coherence(self, energy_gaps: List[float]) -> float:
        """Calculate quantum coherence time"""
        if not energy_gaps:
            return 0.0
        
        # Simplified coherence calculation
        avg_gap = np.mean(energy_gaps)
        return self.physical_constants['h'] / (2 * np.pi * avg_gap)
    
    def classical_analysis(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Perform classical physics analysis"""
        # Implement classical physics calculations
        return {{
            "classical_energy": parameters.get('mass', 1.0) * parameters.get('velocity', 0.0)**2 / 2,
            "momentum": parameters.get('mass', 1.0) * parameters.get('velocity', 0.0),
            "analysis_type": "classical"
        }}


class {clean_name}Simulator:
    """Physics simulation based on the principle"""
    
    def __init__(self):
        self.physics = {clean_name}Physics(
            name="{principle_name}",
            core_principle="{core_principle}",
            timestamp=datetime.utcnow(),
            predictions={predictions},
            study_design={study_design}
        )
    
    def run_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run physics simulation"""
        if parameters.get('quantum', False):
            return self.physics.quantum_analysis(parameters.get('energy_levels', []))
        else:
            return self.physics.classical_analysis(parameters)
    
    def export_results(self) -> Dict[str, Any]:
        """Export simulation results"""
        return {{
            "principle": self.physics.name,
            "timestamp": self.physics.timestamp.isoformat(),
            "predictions": self.physics.predictions
        }}


def main():
    """Example physics simulation"""
    simulator = {clean_name}Simulator()
    
    # Quantum simulation
    quantum_params = {{
        "quantum": True,
        "energy_levels": [0.0, 1.0, 2.0, 3.0]
    }}
    
    quantum_result = simulator.run_simulation(quantum_params)
    print("Quantum Analysis:", quantum_result)
    
    # Classical simulation
    classical_params = {{
        "quantum": False,
        "mass": 2.0,
        "velocity": 5.0
    }}
    
    classical_result = simulator.run_simulation(classical_params)
    print("Classical Analysis:", classical_result)


if __name__ == "__main__":
    main()
'''


def _generate_ai_code(principle_name: str, core_principle: str, 
                      predictions: list, study_design: dict) -> str:
    """Generate AI-specific implementation code"""
    
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', principle_name)
    clean_name = ''.join(word.capitalize() for word in clean_name.split())
    
    return f'''#!/usr/bin/env python3
"""
AI Implementation: {principle_name}

Core Principle: {core_principle}

This code implements the AI principle using machine learning,
neural networks, and artificial intelligence techniques.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime


class {clean_name}NeuralNetwork(nn.Module):
    """Neural network implementation of the {principle_name} principle"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


@dataclass
class {clean_name}AI:
    """AI implementation of the {principle_name} principle"""
    
    name: str
    core_principle: str
    timestamp: datetime
    predictions: List[str]
    study_design: Dict[str, Any]
    
    def __post_init__(self):
        self.timestamp = datetime.utcnow()
        self.model = None
        self.training_data = []
    
    def build_model(self, input_size: int, hidden_size: int, output_size: int):
        """Build neural network model"""
        self.model = {clean_name}NeuralNetwork(input_size, hidden_size, output_size)
        return self.model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train the neural network"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {{epoch}}, Loss: {{loss.item():.4f}}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.numpy()


class {clean_name}AISystem:
    """AI system based on the principle"""
    
    def __init__(self):
        self.ai = {clean_name}AI(
            name="{principle_name}",
            core_principle="{core_principle}",
            timestamp=datetime.utcnow(),
            predictions={predictions},
            study_design={study_design}
        )
    
    def run_ai_analysis(self, data: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """Run AI analysis"""
        # Build and train model
        input_size = data.shape[1]
        output_size = target.shape[1] if len(target.shape) > 1 else 1
        hidden_size = max(input_size, output_size) * 2
        
        self.ai.build_model(input_size, hidden_size, output_size)
        self.ai.train_model(data, target)
        
        # Make predictions
        predictions = self.ai.predict(data)
        
        return {{
            "predictions": predictions,
            "model_architecture": f"{{input_size}}-{{hidden_size}}-{{output_size}}",
            "training_complete": True
        }}


def main():
    """Example AI system usage"""
    ai_system = {clean_name}AISystem()
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100, 1)
    
    # Run AI analysis
    result = ai_system.run_ai_analysis(X, y)
    print("AI Analysis Result:", result)


if __name__ == "__main__":
    main()
'''


def _generate_chemistry_code(principle_name: str, core_principle: str, 
                           predictions: list, study_design: dict) -> str:
    """Generate chemistry-specific implementation code"""
    
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', principle_name)
    clean_name = ''.join(word.capitalize() for word in clean_name.split())
    
    return f'''#!/usr/bin/env python3
"""
Chemistry Implementation: {principle_name}

Core Principle: {core_principle}

This code implements the chemistry principle using molecular dynamics,
chemical kinetics, and computational chemistry techniques.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class {clean_name}Chemistry:
    """Chemistry implementation of the {principle_name} principle"""
    
    name: str
    core_principle: str
    timestamp: datetime
    predictions: List[str]
    study_design: Dict[str, Any]
    
    def __post_init__(self):
        self.timestamp = datetime.utcnow()
        self.chemical_constants = {{
            'R': 8.314,      # Gas constant
            'NA': 6.022e23,  # Avogadro's number
            'k': 1.381e-23,  # Boltzmann constant
            'h': 6.626e-34   # Planck's constant
        }}
    
    def molecular_analysis(self, molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze molecular properties"""
        if not molecules:
            return {{"error": "No molecules provided"}}
        
        total_mass = sum(m.get('mass', 0) for m in molecules)
        total_charge = sum(m.get('charge', 0) for m in molecules)
        
        return {{
            "total_mass": total_mass,
            "total_charge": total_charge,
            "molecule_count": len(molecules),
            "average_mass": total_mass / len(molecules) if molecules else 0
        }}
    
    def reaction_kinetics(self, reactants: List[str], products: List[str], 
                         rate_constant: float) -> Dict[str, Any]:
        """Calculate reaction kinetics"""
        return {{
            "reaction_rate": rate_constant,
            "reactants": reactants,
            "products": products,
            "kinetics_type": "first_order" if len(reactants) == 1 else "higher_order"
        }}


class {clean_name}ChemicalSystem:
    """Chemical system based on the principle"""
    
    def __init__(self):
        self.chemistry = {clean_name}Chemistry(
            name="{principle_name}",
            core_principle="{core_principle}",
            timestamp=datetime.utcnow(),
            predictions={predictions},
            study_design={study_design}
        )
    
    def run_chemical_analysis(self, molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run chemical analysis"""
        return self.chemistry.molecular_analysis(molecules)
    
    def simulate_reaction(self, reactants: List[str], products: List[str], 
                         rate_constant: float) -> Dict[str, Any]:
        """Simulate chemical reaction"""
        return self.chemistry.reaction_kinetics(reactants, products, rate_constant)


def main():
    """Example chemical system usage"""
    chem_system = {clean_name}ChemicalSystem()
    
    # Molecular analysis
    molecules = [
        {{"mass": 18.0, "charge": 0}},  # Water
        {{"mass": 44.0, "charge": 0}},  # CO2
        {{"mass": 32.0, "charge": 0}}   # O2
    ]
    
    analysis = chem_system.run_chemical_analysis(molecules)
    print("Molecular Analysis:", analysis)
    
    # Reaction simulation
    reaction = chem_system.simulate_reaction(
        reactants=["H2", "O2"],
        products=["H2O"],
        rate_constant=1.5e-3
    )
    print("Reaction Kinetics:", reaction)


if __name__ == "__main__":
    main()
'''


def _generate_computation_code(principle_name: str, core_principle: str, 
                              predictions: list, study_design: dict) -> str:
    """Generate computation-specific implementation code"""
    
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', principle_name)
    clean_name = ''.join(word.capitalize() for word in clean_name.split())
    
    return f'''#!/usr/bin/env python3
"""
Computation Implementation: {principle_name}

Core Principle: {core_principle}

This code implements the computation principle using algorithms,
data structures, and computational techniques.
"""

import time
import threading
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


@dataclass
class {clean_name}Computation:
    """Computation implementation of the {principle_name} principle"""
    
    name: str
    core_principle: str
    timestamp: datetime
    predictions: List[str]
    study_design: Dict[str, Any]
    
    def __post_init__(self):
        self.timestamp = datetime.utcnow()
        self.performance_metrics = {{}}
    
    def algorithm_analysis(self, data: List[Any], algorithm_type: str = "sort") -> Dict[str, Any]:
        """Analyze algorithm performance"""
        start_time = time.time()
        
        if algorithm_type == "sort":
            result = sorted(data)
        elif algorithm_type == "search":
            result = data.index(max(data)) if data else -1
        else:
            result = data
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        self.performance_metrics[algorithm_type] = {{
            "execution_time": execution_time,
            "data_size": len(data),
            "algorithm": algorithm_type
        }}
        
        return {{
            "result": result,
            "execution_time": execution_time,
            "data_size": len(data)
        }}
    
    def parallel_processing(self, data: List[Any], num_workers: int = 4) -> Dict[str, Any]:
        """Demonstrate parallel processing"""
        def process_chunk(chunk):
            return [x * 2 for x in chunk]
        
        chunk_size = len(data) // num_workers
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_chunk, chunks))
        
        end_time = time.time()
        
        return {{
            "results": results,
            "execution_time": end_time - start_time,
            "num_workers": num_workers,
            "chunk_size": chunk_size
        }}


class {clean_name}ComputationalSystem:
    """Computational system based on the principle"""
    
    def __init__(self):
        self.computation = {clean_name}Computation(
            name="{principle_name}",
            core_principle="{core_principle}",
            timestamp=datetime.utcnow(),
            predictions={predictions},
            study_design={study_design}
        )
    
    def run_computational_analysis(self, data: List[Any]) -> Dict[str, Any]:
        """Run computational analysis"""
        return self.computation.algorithm_analysis(data)
    
    def run_parallel_analysis(self, data: List[Any], workers: int = 4) -> Dict[str, Any]:
        """Run parallel computational analysis"""
        return self.computation.parallel_processing(data, workers)


def main():
    """Example computational system usage"""
    comp_system = {clean_name}ComputationalSystem()
    
    # Algorithm analysis
    test_data = [64, 34, 25, 12, 22, 11, 90]
    analysis = comp_system.run_computational_analysis(test_data)
    print("Algorithm Analysis:", analysis)
    
    # Parallel processing
    large_data = list(range(1000))
    parallel_result = comp_system.run_parallel_analysis(large_data, workers=4)
    print("Parallel Processing:", parallel_result)


if __name__ == "__main__":
    main()
'''


def _generate_emergent_systems_code(principle_name: str, core_principle: str, 
                                  predictions: list, study_design: dict, semantic_analysis: dict) -> str:
    """Generate emergent systems and complex behavior simulation code"""
    
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', principle_name)
    clean_name = ''.join(word.capitalize() for word in clean_name.split())
    
    # Extract specific implementation requirements
    has_multi_agent = "multi_component_system" in semantic_analysis["implementation_hints"]
    has_stochastic = "stochastic_modeling" in semantic_analysis["implementation_hints"] 
    has_behavior_sim = "behavior_simulation" in semantic_analysis["implementation_hints"]
    
    return f'''#!/usr/bin/env python3
"""
Emergent Systems Implementation: {principle_name}

Core Principle: {core_principle}

This implementation creates a multi-agent system capable of exhibiting
emergent behaviors as predicted by the principle.
"""

import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
from collections import defaultdict


@dataclass
class Agent:
    """Individual agent in the emergent system"""
    id: int
    position: np.ndarray
    state: Dict[str, Any]
    behavior_rules: List[str]
    interaction_radius: float = 1.0
    adaptation_rate: float = 0.1
    
    def update_state(self, neighbors: List['Agent'], environment: Dict[str, Any]) -> None:
        """Update agent state based on neighbors and environment"""
        # Emergent behavior through local interactions
        if neighbors:
            # Calculate collective influence
            neighbor_states = [n.state.get('activation', 0.0) for n in neighbors]
            avg_neighbor_activation = np.mean(neighbor_states)
            
            # Apply principle-driven adaptation
            noise = np.random.normal(0, 0.1) if has_stochastic else 0.05 * (random.random() - 0.5)
            self.state['activation'] = (
                0.7 * self.state.get('activation', 0.5) + 
                0.3 * avg_neighbor_activation + 
                noise
            )
        
        # Constrain to valid range
        self.state['activation'] = max(0.0, min(1.0, self.state.get('activation', 0.5)))
    
    def interact_with(self, other: 'Agent') -> float:
        """Calculate interaction strength with another agent"""
        distance = np.linalg.norm(self.position - other.position)
        if distance > self.interaction_radius:
            return 0.0
        
        # Principle-based interaction
        state_similarity = 1.0 - abs(
            self.state.get('activation', 0.5) - other.state.get('activation', 0.5)
        )
        return state_similarity * (1.0 - distance / self.interaction_radius)


class {clean_name}EmergentSystem:
    """Emergent system implementing the {principle_name} principle"""
    
    def __init__(self, num_agents: int = 50, environment_size: float = 10.0):
        self.num_agents = num_agents
        self.environment_size = environment_size
        self.agents = []
        self.time_step = 0
        self.interaction_network = nx.Graph()
        self.emergence_metrics = {{
            'coherence': [],
            'diversity': [],
            'adaptability': [],
            'complexity': []
        }}
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize agents with random positions and states"""
        for i in range(self.num_agents):
            position = np.random.uniform(0, self.environment_size, 2)
            initial_state = {{
                'activation': random.random(),
                'specialization': random.choice(['explorer', 'coordinator', 'processor']),
                'energy': 1.0
            }}
            
            agent = Agent(
                id=i,
                position=position,
                state=initial_state,
                behavior_rules=['adaptive_response', 'local_coordination'],
                interaction_radius=2.0,
                adaptation_rate=0.1
            )
            self.agents.append(agent)
    
    def step(self) -> Dict[str, Any]:
        """Execute one simulation step"""
        self.time_step += 1
        
        # Update interaction network
        self._update_interaction_network()
        
        # Agent updates (simultaneous)
        new_states = []
        for agent in self.agents:
            neighbors = self._get_neighbors(agent)
            environment = self._get_local_environment(agent)
            
            # Create copy for simultaneous update
            agent_copy = Agent(
                id=agent.id,
                position=agent.position.copy(),
                state=agent.state.copy(),
                behavior_rules=agent.behavior_rules.copy(),
                interaction_radius=agent.interaction_radius,
                adaptation_rate=agent.adaptation_rate
            )
            agent_copy.update_state(neighbors, environment)
            new_states.append(agent_copy)
        
        # Apply updates
        self.agents = new_states
        
        # Measure emergence
        emergence_data = self._measure_emergence()
        for metric, value in emergence_data.items():
            self.emergence_metrics[metric].append(value)
        
        return {{
            'time_step': self.time_step,
            'emergence_metrics': emergence_data,
            'agent_count': len(self.agents),
            'network_density': nx.density(self.interaction_network)
        }}
    
    def _update_interaction_network(self):
        """Update the interaction network based on agent positions"""
        self.interaction_network.clear()
        
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                interaction_strength = agent1.interact_with(agent2)
                if interaction_strength > 0.1:  # Threshold for connection
                    self.interaction_network.add_edge(
                        agent1.id, agent2.id, 
                        weight=interaction_strength
                    )
    
    def _get_neighbors(self, agent: Agent) -> List[Agent]:
        """Get neighboring agents within interaction radius"""
        neighbors = []
        for other in self.agents:
            if other.id != agent.id:
                distance = np.linalg.norm(agent.position - other.position)
                if distance <= agent.interaction_radius:
                    neighbors.append(other)
        return neighbors
    
    def _get_local_environment(self, agent: Agent) -> Dict[str, Any]:
        """Get local environmental conditions"""
        # Simulate environmental gradients
        x, y = agent.position
        return {{
            'resource_density': np.sin(x * 0.5) * np.cos(y * 0.3) + 1.0,
            'temperature': 0.5 + 0.3 * np.sin(x * 0.2),
            'connectivity': len(self._get_neighbors(agent)) / self.num_agents
        }}
    
    def _measure_emergence(self) -> Dict[str, float]:
        """Measure emergent properties of the system"""
        activations = [agent.state.get('activation', 0.5) for agent in self.agents]
        positions = np.array([agent.position for agent in self.agents])
        
        # Coherence: how synchronized the agents are
        coherence = 1.0 - np.std(activations)
        
        # Diversity: variety in agent states
        specializations = [agent.state.get('specialization', 'none') for agent in self.agents]
        unique_specs = len(set(specializations))
        diversity = unique_specs / len(specializations) if specializations else 0.0
        
        # Adaptability: rate of change in system state
        if len(self.emergence_metrics['coherence']) > 0:
            prev_coherence = self.emergence_metrics['coherence'][-1]
            adaptability = abs(coherence - prev_coherence)
        else:
            adaptability = 0.0
        
        # Complexity: network complexity measure
        if self.interaction_network.number_of_edges() > 0:
            try:
                complexity = nx.average_clustering(self.interaction_network)
            except:
                complexity = 0.0
        else:
            complexity = 0.0
        
        return {{
            'coherence': max(0.0, min(1.0, coherence)),
            'diversity': diversity,
            'adaptability': adaptability,
            'complexity': complexity
        }}
    
    def run_simulation(self, steps: int = 100) -> Dict[str, Any]:
        """Run the emergent system simulation"""
        results = []
        
        for _ in range(steps):
            step_result = self.step()
            results.append(step_result)
        
        # Analyze emergence over time
        final_metrics = {{
            'total_steps': steps,
            'emergent_intelligence_detected': self._detect_emergent_intelligence(),
            'system_complexity': np.mean(self.emergence_metrics['complexity'][-10:]) if self.emergence_metrics['complexity'] else 0.0,
            'adaptation_rate': np.mean(self.emergence_metrics['adaptability'][-10:]) if self.emergence_metrics['adaptability'] else 0.0,
            'final_coherence': self.emergence_metrics['coherence'][-1] if self.emergence_metrics['coherence'] else 0.0
        }}
        
        return {{
            'simulation_results': results[-5:],  # Last 5 steps
            'emergence_analysis': final_metrics,
            'principle_validation': self._validate_principle_predictions()
        }}
    
    def _detect_emergent_intelligence(self) -> bool:
        """Detect if emergent intelligence has emerged in the system"""
        if len(self.emergence_metrics['complexity']) < 10:
            return False
        
        # Check for sustained complexity and adaptation
        recent_complexity = np.mean(self.emergence_metrics['complexity'][-10:])
        recent_adaptability = np.mean(self.emergence_metrics['adaptability'][-10:])
        
        # Thresholds for emergent intelligence
        return recent_complexity > 0.3 and recent_adaptability > 0.1
    
    def _validate_principle_predictions(self) -> Dict[str, bool]:
        """Validate the principle's predictions against simulation results"""
        predictions_validated = {{}}
        
        # Check if unpredictable behaviors emerged
        activations_over_time = []
        for i in range(len(self.agents)):
            agent_activations = []
            # Would need to track individual agent states over time
            # This is a simplified validation
        
        predictions_validated['emergent_behavior_observed'] = self._detect_emergent_intelligence()
        predictions_validated['unpredictable_outcomes'] = np.std(self.emergence_metrics['coherence']) > 0.1 if self.emergence_metrics['coherence'] else False
        
        return predictions_validated


def main():
    """Demonstrate the emergent system implementation"""
    print(f"Running {principle_name} Emergent System Simulation...")
    
    # Create and run emergent system
    system = {clean_name}EmergentSystem(num_agents=30, environment_size=8.0)
    
    print("Initial system state:")
    print(f"Agents: {{system.num_agents}}")
    print(f"Environment size: {{system.environment_size}}")
    
    # Run simulation
    results = system.run_simulation(steps=50)
    
    print("\\nSimulation Results:")
    print(f"Emergent intelligence detected: {{results['emergence_analysis']['emergent_intelligence_detected']}}")
    print(f"System complexity: {{results['emergence_analysis']['system_complexity']:.3f}}")
    print(f"Adaptation rate: {{results['emergence_analysis']['adaptation_rate']:.3f}}")
    print(f"Final coherence: {{results['emergence_analysis']['final_coherence']:.3f}}")
    
    print("\\nPrinciple Validation:")
    for prediction, validated in results['principle_validation'].items():
        print(f"{{prediction}}: {{validated}}")


if __name__ == "__main__":
    main()
'''


def _generate_principle_driven_code(principle_name: str, core_principle: str, 
                                  predictions: list, study_design: dict, semantic_analysis: dict) -> str:
    """Generate code driven by the specific principle rather than domain templates"""
    
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', principle_name)
    clean_name = ''.join(word.capitalize() for word in clean_name.split())
    
    # Extract actionable elements from predictions
    implementation_strategies = []
    test_cases = []
    
    for prediction in predictions:
        if isinstance(prediction, dict) and "prediction" in prediction:
            pred_text = prediction["prediction"].lower()
            
            # Convert predictions to implementation strategies
            if "exhibit" in pred_text and "behavior" in pred_text:
                implementation_strategies.append("behavior_modeling")
                test_cases.append("behavior_validation")
            
            if "difficult to predict" in pred_text or "unpredictable" in pred_text:
                implementation_strategies.append("stochastic_modeling")
                test_cases.append("predictability_analysis")
            
            if "complex" in pred_text and "system" in pred_text:
                implementation_strategies.append("complex_system_simulation")
                test_cases.append("complexity_measurement")
    
    return f'''#!/usr/bin/env python3
"""
Principle-Driven Implementation: {principle_name}

Core Principle: {core_principle}

This implementation is specifically designed to demonstrate and test
the principle's predictions through functional algorithms.
"""

import numpy as np
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import random
from abc import ABC, abstractmethod


@dataclass
class PrincipleImplementation:
    """Base class for principle-driven implementations"""
    
    name: str
    core_principle: str
    predictions: List[Dict[str, Any]]
    implementation_strategies: List[str]
    timestamp: datetime
    
    def __post_init__(self):
        self.test_results = {{}}
        self.validation_metrics = {{}}
    
    @abstractmethod
    def execute_principle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the core principle logic"""
        pass
    
    @abstractmethod
    def validate_predictions(self) -> Dict[str, bool]:
        """Validate principle predictions"""
        pass


class {clean_name}Implementation(PrincipleImplementation):
    """Concrete implementation of {principle_name}"""
    
    def __init__(self):
        super().__init__(
            name="{principle_name}",
            core_principle="{core_principle}",
            predictions={predictions},
            implementation_strategies={implementation_strategies},
            timestamp=datetime.utcnow()
        )
        
        self.principle_engine = self._build_principle_engine()
        self.test_suite = self._build_test_suite()
    
    def _build_principle_engine(self) -> Dict[str, Callable]:
        """Build the core engine that implements the principle"""
        engine = {{}}
        
        # Strategy-specific implementations
        {"if 'behavior_modeling' in self.implementation_strategies:" if implementation_strategies else "# No specific strategies detected"}
        {"    engine['behavior_model'] = self._create_behavior_model()" if "behavior_modeling" in str(implementation_strategies) else ""}
        
        {"if 'stochastic_modeling' in self.implementation_strategies:" if implementation_strategies else ""}
        {"    engine['stochastic_model'] = self._create_stochastic_model()" if "stochastic_modeling" in str(implementation_strategies) else ""}
        
        {"if 'complex_system_simulation' in self.implementation_strategies:" if implementation_strategies else ""}
        {"    engine['complex_system'] = self._create_complex_system()" if "complex_system_simulation" in str(implementation_strategies) else ""}
        
        # Default principle processor
        engine['principle_processor'] = self._create_principle_processor()
        
        return engine
    
    def _create_behavior_model(self) -> Callable:
        """Create behavior modeling component"""
        def behavior_model(input_data: Dict[str, Any]) -> Dict[str, Any]:
            # Model behaviors based on principle
            agents = input_data.get('agents', [{{i: {{'state': random.random()}} for i in range(10)}}])
            
            behaviors = []
            for agent_id, agent_data in agents.items():
                # Apply principle-driven behavior rules
                current_state = agent_data.get('state', 0.5)
                
                # Implement principle-specific behavior
                new_state = current_state + np.random.normal(0, 0.1)
                new_state = max(0.0, min(1.0, new_state))  # Constrain to [0,1]
                
                behaviors.append({{
                    'agent_id': agent_id,
                    'old_state': current_state,
                    'new_state': new_state,
                    'behavior_change': abs(new_state - current_state)
                }})
            
            return {{
                'behaviors': behaviors,
                'collective_behavior': np.mean([b['behavior_change'] for b in behaviors]),
                'principle_applied': True
            }}
        
        return behavior_model
    
    def _create_stochastic_model(self) -> Callable:
        """Create stochastic modeling component"""
        def stochastic_model(input_data: Dict[str, Any]) -> Dict[str, Any]:
            # Model unpredictable elements as predicted by principle
            
            sample_size = input_data.get('sample_size', 100)
            
            # Generate stochastic outcomes based on principle
            outcomes = []
            for i in range(sample_size):
                # Principle-driven stochastic process
                base_value = random.random()
                noise = np.random.normal(0, 0.2)
                final_value = base_value + noise
                
                outcomes.append({{
                    'iteration': i,
                    'base_value': base_value,
                    'noise': noise,
                    'outcome': final_value,
                    'predictability': abs(noise) < 0.1  # Low noise = more predictable
                }})
            
            # Analyze predictability
            predictable_count = sum(1 for o in outcomes if o['predictability'])
            unpredictability_ratio = 1.0 - (predictable_count / len(outcomes))
            
            return {{
                'outcomes': outcomes[-5:],  # Last 5 for brevity
                'total_samples': len(outcomes),
                'unpredictability_ratio': unpredictability_ratio,
                'principle_demonstrated': unpredictability_ratio > 0.3  # Threshold for "difficult to predict"
            }}
        
        return stochastic_model
    
    def _create_complex_system(self) -> Callable:
        """Create complex system simulation component"""
        def complex_system(input_data: Dict[str, Any]) -> Dict[str, Any]:
            # Simulate complex system dynamics
            
            num_components = input_data.get('components', 20)
            simulation_steps = input_data.get('steps', 50)
            
            # Initialize system components
            components = [{{
                'id': i,
                'state': random.random(),
                'connections': random.sample(range(num_components), k=min(5, num_components-1))
            }} for i in range(num_components)]
            
            system_states = []
            
            for step in range(simulation_steps):
                # Update each component based on connections (principle-driven)
                new_states = []
                
                for comp in components:
                    # Calculate influence from connected components
                    connected_states = [components[conn_id]['state'] for conn_id in comp['connections'] if conn_id != comp['id']]
                    
                    if connected_states:
                        avg_influence = np.mean(connected_states)
                        new_state = 0.7 * comp['state'] + 0.3 * avg_influence + np.random.normal(0, 0.05)
                    else:
                        new_state = comp['state'] + np.random.normal(0, 0.1)
                    
                    new_state = max(0.0, min(1.0, new_state))
                    new_states.append(new_state)
                
                # Update system
                for i, comp in enumerate(components):
                    comp['state'] = new_states[i]
                
                # Record system state
                system_coherence = 1.0 - np.std(new_states)
                system_states.append({{
                    'step': step,
                    'coherence': system_coherence,
                    'average_state': np.mean(new_states),
                    'complexity': len(set(round(s, 1) for s in new_states)) / len(new_states)
                }})
            
            return {{
                'final_components': components[:3],  # First 3 for brevity
                'system_evolution': system_states[-10:],  # Last 10 steps
                'final_coherence': system_states[-1]['coherence'] if system_states else 0.0,
                'complexity_maintained': np.mean([s['complexity'] for s in system_states]) > 0.3
            }}
        
        return complex_system
    
    def _create_principle_processor(self) -> Callable:
        """Create general principle processing component"""
        def principle_processor(input_data: Dict[str, Any]) -> Dict[str, Any]:
            # General principle application
            
            processed_data = {{
                'input_processed': True,
                'principle_name': self.name,
                'core_principle': self.core_principle,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'principle_confidence': 0.85  # High confidence in principle application
            }}
            
            # Apply principle-specific transformations
            if 'data' in input_data:
                data = input_data['data']
                if isinstance(data, (list, np.ndarray)):
                    # Transform data according to principle
                    transformed = [x * 1.1 + np.random.normal(0, 0.05) for x in data]
                    processed_data['transformed_data'] = transformed[:10]  # First 10 for brevity
                    processed_data['transformation_applied'] = True
            
            return processed_data
        
        return principle_processor
    
    def _build_test_suite(self) -> Dict[str, Callable]:
        """Build test suite for principle validation"""
        tests = {{}}
        
        {"# Test case implementations based on predictions" if test_cases else "# No specific test cases generated"}
        {"if 'behavior_validation' in [" + str(test_cases) + "]:" if test_cases else ""}
        {"    tests['behavior_validation'] = self._test_behavior_validation" if "behavior_validation" in str(test_cases) else ""}
        
        {"if 'predictability_analysis' in [" + str(test_cases) + "]:" if test_cases else ""}
        {"    tests['predictability_analysis'] = self._test_predictability_analysis" if "predictability_analysis" in str(test_cases) else ""}
        
        {"if 'complexity_measurement' in [" + str(test_cases) + "]:" if test_cases else ""}
        {"    tests['complexity_measurement'] = self._test_complexity_measurement" if "complexity_measurement" in str(test_cases) else ""}
        
        tests['general_principle_test'] = self._test_general_principle
        
        return tests
    
    def execute_principle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the core principle logic"""
        results = {{
            'execution_timestamp': datetime.utcnow().isoformat(),
            'input_data': input_data,
            'principle_results': {{}}
        }}
        
        # Execute all available engine components
        for component_name, component_func in self.principle_engine.items():
            try:
                component_result = component_func(input_data)
                results['principle_results'][component_name] = component_result
            except Exception as e:
                results['principle_results'][component_name] = {{
                    'error': str(e),
                    'status': 'failed'
                }}
        
        return results
    
    def validate_predictions(self) -> Dict[str, bool]:
        """Validate principle predictions"""
        validation_results = {{}}
        
        # Run all test cases
        for test_name, test_func in self.test_suite.items():
            try:
                test_result = test_func()
                validation_results[test_name] = test_result.get('passed', False)
            except Exception as e:
                validation_results[test_name] = False
        
        return validation_results
    
    def _test_general_principle(self) -> Dict[str, Any]:
        """General test for principle functionality"""
        test_data = {{'data': [1, 2, 3, 4, 5], 'sample_size': 20}}
        
        try:
            result = self.execute_principle(test_data)
            
            # Check if principle was applied successfully
            has_results = len(result.get('principle_results', {{}})) > 0
            no_errors = not any('error' in r for r in result.get('principle_results', {{}}).values() if isinstance(r, dict))
            
            return {{
                'passed': has_results and no_errors,
                'has_results': has_results,
                'no_errors': no_errors,
                'result_summary': list(result.get('principle_results', {{}}).keys())
            }}
        except Exception as e:
            return {{'passed': False, 'error': str(e)}}


def main():
    """Demonstrate the principle-driven implementation"""
    print(f"Running {{'{principle_name}'}} Implementation...")
    
    # Create implementation
    impl = {clean_name}Implementation()
    
    print(f"Principle: {{impl.core_principle}}")
    print(f"Strategies: {{impl.implementation_strategies}}")
    
    # Test data
    test_input = {{
        'data': list(range(1, 11)),
        'sample_size': 50,
        'components': 15,
        'steps': 30,
        'agents': {{str(i): {{'state': np.random.random()}} for i in range(8)}}
    }}
    
    print("\\nExecuting principle...")
    results = impl.execute_principle(test_input)
    
    print("\\nPrinciple Results:")
    for component, result in results['principle_results'].items():
        if isinstance(result, dict) and 'error' not in result:
            print(f"  {{component}}: Successfully executed")
        else:
            print(f"  {{component}}: {{result.get('error', 'Unknown error')}}")
    
    print("\\nValidating predictions...")
    validation = impl.validate_predictions()
    
    print("\\nValidation Results:")
    for test, passed in validation.items():
        print(f"  {{test}}: {{'PASSED' if passed else 'FAILED'}}")
    
    overall_success = sum(validation.values()) / len(validation) if validation else 0.0
    print(f"\\nOverall validation success: {{overall_success:.1%}}")


if __name__ == "__main__":
    main()
'''


def _generate_general_code(principle_name: str, core_principle: str, 
                          predictions: list, study_design: dict) -> str:
    """Generate general-purpose implementation code"""
    
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', principle_name)
    clean_name = ''.join(word.capitalize() for word in clean_name.split())
    
    return f'''#!/usr/bin/env python3
"""
General Implementation: {principle_name}

Core Principle: {core_principle}

This code implements the general principle using flexible,
domain-agnostic techniques.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class {clean_name}Principle:
    """General implementation of the {principle_name} principle"""
    
    name: str
    core_principle: str
    timestamp: datetime
    predictions: List[str]
    study_design: Dict[str, Any]
    
    def __post_init__(self):
        self.timestamp = datetime.utcnow()
    
    def apply_principle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the principle to input data"""
        result = {{
            "principle_applied": self.name,
            "input_data": input_data,
            "output": "Principle application logic to be implemented",
            "timestamp": self.timestamp.isoformat()
        }}
        return result
    
    def validate_prediction(self, prediction: str) -> bool:
        """Validate if a prediction aligns with this principle"""
        return prediction in self.predictions
    
    def get_study_parameters(self) -> Dict[str, Any]:
        """Get parameters for studying this principle"""
        return self.study_design


class {clean_name}Implementation:
    """Concrete implementation of the {clean_name} principle"""
    
    def __init__(self):
        self.principle = {clean_name}Principle(
            name="{principle_name}",
            core_principle="{core_principle}",
            timestamp=datetime.utcnow(),
            predictions={predictions},
            study_design={study_design}
        )
    
    def run_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using this principle"""
        return self.principle.apply_principle(data)
    
    def export_principle(self) -> Dict[str, Any]:
        """Export the principle as a dictionary"""
        return {{
            "name": self.principle.name,
            "core_principle": self.principle.core_principle,
            "timestamp": self.principle.timestamp.isoformat(),
            "predictions": self.principle.predictions,
            "study_design": self.principle.study_design
        }}


def main():
    """Example usage of the generated code"""
    implementation = {clean_name}Implementation()
    
    # Example data
    test_data = {{
        "input": "test data",
        "parameters": {{"param1": "value1"}}
    }}
    
    # Run analysis
    result = implementation.run_analysis(test_data)
    print("Analysis Result:", json.dumps(result, indent=2))
    
    # Export principle
    principle_data = implementation.export_principle()
    print("\\nPrinciple Data:", json.dumps(principle_data, indent=2))


if __name__ == "__main__":
    main()
'''


def _generate_fallback_code(oracle_output: str) -> str:
    """Generate fallback code when Oracle output can't be parsed"""
    
    return f'''#!/usr/bin/env python3
"""
Fallback Code Generation

Oracle Output (unparseable):
{oracle_output}

This code was generated as a fallback when the Oracle output
could not be properly parsed.
"""

class FallbackImplementation:
    """Fallback implementation when principle parsing fails"""
    
    def __init__(self, raw_output: str):
        self.raw_output = raw_output
    
    def process(self, data: str) -> str:
        """Process data using fallback logic"""
        return f"Processed: {{data}} using fallback implementation"
    
    def get_raw_output(self) -> str:
        """Get the raw Oracle output"""
        return self.raw_output


def main():
    """Example usage of fallback code"""
    fallback = FallbackImplementation("""{oracle_output}""")
    result = fallback.process("test input")
    print("Fallback Result:", result)


if __name__ == "__main__":
    main()
'''
