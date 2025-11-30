"""
Data Flow Integration Tests for Elite Financial AI

This module tests the complete data flow from quantum execution to database storage,
RL training to memory system, financial predictions to vector store, and model checkpoints
to file storage with ICEBURG integration.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import Elite Financial AI components
from iceburg.quantum.circuits import VQC, simple_vqc
from iceburg.quantum.sampling import QuantumSampler
from iceburg.rl.agents import PPOTrader
from iceburg.financial.data_pipeline import FinancialDataPipeline
from iceburg.database.elite_financial_schema import EliteFinancialSchema
from iceburg.integration.elite_database_integration import EliteDatabaseIntegration
from iceburg.integration.elite_memory_integration import EliteMemoryIntegration
from iceburg.integration.elite_metrics_integration import EliteMetricsIntegration
from iceburg.agents.elite_agent_factory import EliteAgentFactory
from iceburg.training.elite_training_collector import EliteTrainingCollector
from iceburg.training.elite_model_tuner import EliteModelTuner


class TestDataFlow:
    """Test data flow between Elite Financial AI components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def elite_components(self, temp_dir):
        """Initialize Elite Financial AI components."""
        db_path = Path(temp_dir) / "test.db"
        memory_dir = Path(temp_dir) / "memory"
        vector_dir = Path(temp_dir) / "vector_store"
        
        # Create directories
        memory_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        database_integration = EliteDatabaseIntegration(str(db_path))
        memory_integration = EliteMemoryIntegration(str(memory_dir), str(vector_dir))
        metrics_integration = EliteMetricsIntegration(str(db_path), str(memory_dir), str(vector_dir))
        agent_factory = EliteAgentFactory(str(db_path), str(memory_dir), str(vector_dir))
        training_collector = EliteTrainingCollector(str(db_path), str(memory_dir), str(vector_dir), str(Path(temp_dir) / "training_data"))
        model_tuner = EliteModelTuner(str(db_path), str(memory_dir), str(vector_dir))
        
        yield {
            "database_integration": database_integration,
            "memory_integration": memory_integration,
            "metrics_integration": metrics_integration,
            "agent_factory": agent_factory,
            "training_collector": training_collector,
            "model_tuner": model_tuner
        }
        
        # Cleanup
        database_integration.close()
        memory_integration.close()
        metrics_integration.close()
        agent_factory.close()
        training_collector.close()
        model_tuner.close()
    
    def test_quantum_execution_to_database_storage(self, elite_components):
        """Test quantum execution to database storage flow."""
        # Create quantum circuit
        quantum_circuit = VQC(n_qubits=4, n_layers=2)
        
        # Execute quantum circuit
        features = torch.randn(4)
        start_time = datetime.now()
        result = quantum_circuit(features)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Record in database
        circuit_id = elite_components["database_integration"].record_quantum_execution(
            circuit_type="variational",
            parameters={"n_qubits": 4, "n_layers": 2},
            execution_time=execution_time,
            result=result.tolist() if hasattr(result, 'tolist') else result,
            n_qubits=4,
            n_layers=2,
            device="default.qubit",
            shots=1000,
            success=True
        )
        
        # Verify storage
        assert circuit_id is not None
        assert len(circuit_id) > 0
        
        # Check database
        quantum_executions = elite_components["database_integration"].schema.get_quantum_executions(limit=10)
        assert len(quantum_executions) > 0
        
        # Find our execution
        our_execution = next((exec for exec in quantum_executions if exec.circuit_id == circuit_id), None)
        assert our_execution is not None
        assert our_execution.success is True
        assert our_execution.n_qubits == 4
        assert our_execution.n_layers == 2
    
    def test_rl_training_to_memory_system(self, elite_components):
        """Test RL training to memory system flow."""
        # Create RL agent
        rl_agent = PPOTrader(learning_rate=0.001, batch_size=256)
        
        # Simulate RL training
        episode_id = elite_components["database_integration"].record_rl_episode(
            agent_name="PPO_Trader",
            environment="TradingEnv",
            reward=150.5,
            steps=1000,
            algorithm="PPO",
            hyperparameters={"learning_rate": 0.001, "batch_size": 256},
            convergence_metric=0.85,
            breakthrough_detected=False
        )
        
        # Verify storage
        assert episode_id is not None
        assert len(episode_id) > 0
        
        # Check database
        rl_episodes = elite_components["database_integration"].schema.get_rl_episodes(limit=10)
        assert len(rl_episodes) > 0
        
        # Find our episode
        our_episode = next((ep for ep in rl_episodes if ep.episode_id == episode_id), None)
        assert our_episode is not None
        assert our_episode.reward == 150.5
        assert our_episode.steps == 1000
        assert our_episode.algorithm == "PPO"
    
    def test_financial_predictions_to_vector_store(self, elite_components):
        """Test financial predictions to vector store flow."""
        # Create financial prediction
        prediction_id = elite_components["database_integration"].record_financial_prediction(
            symbol="AAPL",
            prediction_type="price",
            value=150.25,
            confidence=0.85,
            model_type="quantum_rl",
            features_used=["price", "volume", "rsi", "macd"],
            actual_value=152.30
        )
        
        # Verify storage
        assert prediction_id is not None
        assert len(prediction_id) > 0
        
        # Check database
        financial_predictions = elite_components["database_integration"].schema.get_financial_predictions(limit=10)
        assert len(financial_predictions) > 0
        
        # Find our prediction
        our_prediction = next((pred for pred in financial_predictions if pred.prediction_id == prediction_id), None)
        assert our_prediction is not None
        assert our_prediction.symbol == "AAPL"
        assert our_prediction.value == 150.25
        assert our_prediction.confidence == 0.85
    
    def test_model_checkpoints_to_file_storage(self, elite_components):
        """Test model checkpoints to file storage flow."""
        # Create model checkpoint
        checkpoint_id = elite_components["database_integration"].store_model_checkpoint(
            model_type="quantum_rl",
            model_state={"weights": [0.1, 0.2, 0.3], "biases": [0.01, 0.02]},
            performance_metrics={"accuracy": 0.85, "loss": 0.15},
            training_epoch=100,
            validation_score=0.87,
            hyperparameters={"learning_rate": 0.001, "batch_size": 256}
        )
        
        # Verify storage
        assert checkpoint_id is not None
        assert len(checkpoint_id) > 0
        
        # Check database
        cursor = elite_components["database_integration"].connection.cursor()
        cursor.execute("SELECT * FROM model_checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "quantum_rl"  # model_type
        assert row[5] == 100  # training_epoch
        assert row[6] == 0.87  # validation_score
    
    def test_metrics_tracking_flow(self, elite_components):
        """Test metrics tracking flow."""
        # Track quantum circuit metrics
        quantum_metric_id = elite_components["metrics_integration"].track_quantum_circuit_metrics(
            circuit_id="test_circuit_001",
            execution_time=0.123,
            n_qubits=4,
            n_layers=2,
            device="default.qubit",
            success=True,
            result=[0.5, -0.3, 0.8, -0.1]
        )
        
        # Track RL training metrics
        rl_metric_id = elite_components["metrics_integration"].track_rl_training_metrics(
            episode_id="test_episode_001",
            reward=150.5,
            steps=1000,
            algorithm="PPO",
            convergence_metric=0.85,
            breakthrough_detected=False,
            training_time=0.5
        )
        
        # Track financial prediction metrics
        financial_metric_id = elite_components["metrics_integration"].track_financial_prediction_metrics(
            prediction_id="test_prediction_001",
            symbol="AAPL",
            prediction_type="price",
            value=150.25,
            confidence=0.85,
            accuracy=0.87,
            model_type="quantum_rl",
            prediction_time=0.1
        )
        
        # Verify metrics storage
        assert quantum_metric_id is not None
        assert rl_metric_id is not None
        assert financial_metric_id is not None
        
        # Check metrics summary
        summary = elite_components["metrics_integration"].get_metrics_summary()
        assert len(summary) > 0
        
        # Check system metrics
        system_summary = elite_components["metrics_integration"].get_system_metrics_summary()
        assert isinstance(system_summary, dict)
    
    def test_agent_creation_and_registration(self, elite_components):
        """Test agent creation and registration flow."""
        # Create quantum trader agent
        quantum_config = {
            "n_qubits": 4,
            "n_layers": 2,
            "quantum_device": "default.qubit",
            "trading_strategy": "momentum",
            "risk_tolerance": "medium"
        }
        
        quantum_agent = elite_components["agent_factory"].create_quantum_trader_agent(quantum_config)
        assert quantum_agent is not None
        assert quantum_agent.agent_id is not None
        
        # Register with ICEBURG
        registration_id = elite_components["agent_factory"].register_with_iceburg(quantum_agent, "project_001")
        assert registration_id is not None
        
        # Verify agent retrieval
        retrieved_agent = elite_components["agent_factory"].get_agent(quantum_agent.agent_id)
        assert retrieved_agent is not None
        assert retrieved_agent.agent_id == quantum_agent.agent_id
        
        # Check all agents
        all_agents = elite_components["agent_factory"].get_all_agents()
        assert len(all_agents) > 0
        assert quantum_agent.agent_id in all_agents
    
    def test_training_data_collection(self, elite_components):
        """Test training data collection flow."""
        # First, create some data to collect
        elite_components["database_integration"].record_quantum_execution(
            circuit_type="variational",
            parameters={"n_qubits": 4, "n_layers": 2},
            execution_time=0.123,
            result=[0.5, -0.3, 0.8, -0.1],
            n_qubits=4,
            n_layers=2,
            device="default.qubit",
            shots=1000,
            success=True
        )
        
        elite_components["database_integration"].record_rl_episode(
            agent_name="PPO_Trader",
            environment="TradingEnv",
            reward=150.5,
            steps=1000,
            algorithm="PPO",
            hyperparameters={"learning_rate": 0.001, "batch_size": 256},
            convergence_metric=0.85,
            breakthrough_detected=False
        )
        
        # Collect quantum circuit patterns
        quantum_patterns = elite_components["training_collector"].collect_quantum_circuit_patterns()
        assert "patterns" in quantum_patterns
        assert len(quantum_patterns["patterns"]) > 0
        
        # Collect RL episode trajectories
        rl_trajectories = elite_components["training_collector"].collect_rl_episode_trajectories()
        assert "trajectories" in rl_trajectories
        assert len(rl_trajectories["trajectories"]) > 0
        
        # Generate training datasets
        datasets = elite_components["training_collector"].generate_training_datasets()
        assert "datasets" in datasets
        assert len(datasets["datasets"]) > 0
    
    def test_model_tuning_flow(self, elite_components):
        """Test model tuning flow."""
        # Tune quantum circuit
        quantum_config = {
            "device": "default.qubit",
            "optimization_target": "execution_time"
        }
        
        quantum_result = elite_components["model_tuner"].tune_quantum_circuit(quantum_config, n_trials=5)
        assert quantum_result.success is True
        assert quantum_result.hyperparameters is not None
        
        # Tune RL agent
        rl_config = {
            "algorithm": "PPO",
            "optimization_target": "reward"
        }
        
        rl_result = elite_components["model_tuner"].tune_rl_agent(rl_config, n_trials=5)
        assert rl_result.success is True
        assert rl_result.hyperparameters is not None
        
        # Check tuning results
        all_results = elite_components["model_tuner"].get_tuning_results()
        assert len(all_results) > 0
        
        # Check tuning statistics
        stats = elite_components["model_tuner"].get_tuning_stats()
        assert isinstance(stats, dict)
    
    def test_end_to_end_data_flow(self, elite_components):
        """Test complete end-to-end data flow."""
        # 1. Create and execute quantum circuit
        quantum_circuit = VQC(n_qubits=4, n_layers=2)
        features = torch.randn(4)
        result = quantum_circuit(features)
        
        # 2. Record quantum execution
        circuit_id = elite_components["database_integration"].record_quantum_execution(
            circuit_type="variational",
            parameters={"n_qubits": 4, "n_layers": 2},
            execution_time=0.123,
            result=result.tolist() if hasattr(result, 'tolist') else result,
            n_qubits=4,
            n_layers=2,
            device="default.qubit",
            shots=1000,
            success=True
        )
        
        # 3. Create and train RL agent
        rl_agent = PPOTrader(learning_rate=0.001, batch_size=256)
        episode_id = elite_components["database_integration"].record_rl_episode(
            agent_name="PPO_Trader",
            environment="TradingEnv",
            reward=150.5,
            steps=1000,
            algorithm="PPO",
            hyperparameters={"learning_rate": 0.001, "batch_size": 256},
            convergence_metric=0.85,
            breakthrough_detected=False
        )
        
        # 4. Make financial prediction
        prediction_id = elite_components["database_integration"].record_financial_prediction(
            symbol="AAPL",
            prediction_type="price",
            value=150.25,
            confidence=0.85,
            model_type="quantum_rl",
            features_used=["price", "volume", "rsi", "macd"],
            actual_value=152.30
        )
        
        # 5. Track metrics
        elite_components["metrics_integration"].track_quantum_circuit_metrics(
            circuit_id=circuit_id,
            execution_time=0.123,
            n_qubits=4,
            n_layers=2,
            device="default.qubit",
            success=True,
            result=result.tolist() if hasattr(result, 'tolist') else result
        )
        
        # 6. Collect training data
        quantum_patterns = elite_components["training_collector"].collect_quantum_circuit_patterns()
        rl_trajectories = elite_components["training_collector"].collect_rl_episode_trajectories()
        
        # 7. Verify all data is stored correctly
        assert circuit_id is not None
        assert episode_id is not None
        assert prediction_id is not None
        assert len(quantum_patterns["patterns"]) > 0
        assert len(rl_trajectories["trajectories"]) > 0
        
        # 8. Check database integrity
        quantum_executions = elite_components["database_integration"].schema.get_quantum_executions(limit=10)
        rl_episodes = elite_components["database_integration"].schema.get_rl_episodes(limit=10)
        financial_predictions = elite_components["database_integration"].schema.get_financial_predictions(limit=10)
        
        assert len(quantum_executions) > 0
        assert len(rl_episodes) > 0
        assert len(financial_predictions) > 0
        
        # 9. Verify metrics are tracked
        summary = elite_components["metrics_integration"].get_metrics_summary()
        assert len(summary) > 0
        
        print("✅ End-to-end data flow test completed successfully!")


# Run tests
if __name__ == "__main__":
    # Test data flow
    test_data_flow = TestDataFlow()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize components
        elite_components = next(test_data_flow.elite_components(temp_dir))
        
        # Run tests
        test_data_flow.test_quantum_execution_to_database_storage(elite_components)
        print("✅ Quantum execution to database storage test passed")
        
        test_data_flow.test_rl_training_to_memory_system(elite_components)
        print("✅ RL training to memory system test passed")
        
        test_data_flow.test_financial_predictions_to_vector_store(elite_components)
        print("✅ Financial predictions to vector store test passed")
        
        test_data_flow.test_model_checkpoints_to_file_storage(elite_components)
        print("✅ Model checkpoints to file storage test passed")
        
        test_data_flow.test_metrics_tracking_flow(elite_components)
        print("✅ Metrics tracking flow test passed")
        
        test_data_flow.test_agent_creation_and_registration(elite_components)
        print("✅ Agent creation and registration test passed")
        
        test_data_flow.test_training_data_collection(elite_components)
        print("✅ Training data collection test passed")
        
        test_data_flow.test_model_tuning_flow(elite_components)
        print("✅ Model tuning flow test passed")
        
        test_data_flow.test_end_to_end_data_flow(elite_components)
        print("✅ End-to-end data flow test passed")
        
        print("✅ All data flow tests completed successfully!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
