"""
Elite Financial AI Memory Integration for ICEBURG

This module integrates Elite Financial AI operations with ICEBURG's unified memory system,
providing comprehensive event logging, vector indexing, and cross-session learning
for quantum circuits, RL training, and financial predictions.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging
import uuid
import time
import os

from ..memory.unified_memory import UnifiedMemory
from ..database.elite_financial_schema import (
    QuantumCircuitExecution,
    RLTrainingEpisode,
    FinancialPrediction,
    ModelCheckpoint,
    QuantumRLExperiment
)

logger = logging.getLogger(__name__)


class EliteMemoryIntegration:
    """
    Elite Financial AI memory integration with ICEBURG.
    
    Provides comprehensive memory operations for Elite Financial AI,
    including event logging, vector indexing, and cross-session learning
    with ICEBURG's unified memory system.
    """
    
    def __init__(self, memory_dir: str = "data/memory", vector_dir: str = "data/vector_store"):
        """
        Initialize Elite Financial AI memory integration.
        
        Args:
            memory_dir: Directory for memory storage
            vector_dir: Directory for vector storage
        """
        self.memory_dir = memory_dir
        self.vector_dir = vector_dir
        self.unified_memory = UnifiedMemory(memory_dir)
        self._setup_memory_directories()
        self._setup_vector_indexing()
    
    def _setup_memory_directories(self):
        """Setup memory directories."""
        try:
            os.makedirs(self.memory_dir, exist_ok=True)
            os.makedirs(self.vector_dir, exist_ok=True)
            logger.info("Memory directories created successfully")
        except Exception as e:
            logger.error(f"Error setting up memory directories: {e}")
            raise
    
    def _setup_vector_indexing(self):
        """Setup vector indexing for Elite Financial AI."""
        try:
            # Initialize vector store for different data types
            self.quantum_vector_store = self._create_vector_store("quantum_circuits")
            self.rl_vector_store = self._create_vector_store("rl_episodes")
            self.financial_vector_store = self._create_vector_store("financial_predictions")
            self.experiment_vector_store = self._create_vector_store("experiments")
            
            logger.info("Vector indexing setup completed")
        except Exception as e:
            logger.error(f"Error setting up vector indexing: {e}")
            raise
    
    def _create_vector_store(self, store_name: str):
        """Create vector store for specific data type."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create vector store directory
            store_dir = os.path.join(self.vector_dir, store_name)
            os.makedirs(store_dir, exist_ok=True)
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path=store_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            collection = client.get_or_create_collection(
                name=store_name,
                metadata={"description": f"Vector store for {store_name}"}
            )
            
            return collection
            
        except Exception as e:
            logger.error(f"Error creating vector store {store_name}: {e}")
            return None
    
    def log_quantum_execution(self, execution: QuantumCircuitExecution,
                             iceburg_project_id: Optional[str] = None):
        """
        Log quantum circuit execution to memory system.
        
        Args:
            execution: Quantum circuit execution record
            iceburg_project_id: ICEBURG project ID
        """
        try:
            # Create event log entry
            event_data = {
                "event_type": "quantum_execution",
                "circuit_id": execution.circuit_id,
                "circuit_type": execution.circuit_type,
                "parameters": execution.parameters,
                "execution_time": execution.execution_time,
                "result": execution.result,
                "n_qubits": execution.n_qubits,
                "n_layers": execution.n_layers,
                "device": execution.device,
                "shots": execution.shots,
                "success": execution.success,
                "error_message": execution.error_message,
                "timestamp": execution.timestamp.isoformat(),
                "iceburg_project_id": iceburg_project_id
            }
            
            # Log to unified memory
            self.unified_memory.log_event("quantum_execution", event_data)
            
            # Index in vector store
            self._index_quantum_execution(execution)
            
            logger.debug(f"Logged quantum execution: {execution.circuit_id}")
            
        except Exception as e:
            logger.error(f"Error logging quantum execution: {e}")
    
    def log_rl_episode(self, episode: RLTrainingEpisode,
                      iceburg_project_id: Optional[str] = None):
        """
        Log RL training episode to memory system.
        
        Args:
            episode: RL training episode record
            iceburg_project_id: ICEBURG project ID
        """
        try:
            # Create event log entry
            event_data = {
                "event_type": "rl_episode",
                "episode_id": episode.episode_id,
                "agent_name": episode.agent_name,
                "environment": episode.environment,
                "reward": episode.reward,
                "steps": episode.steps,
                "algorithm": episode.algorithm,
                "hyperparameters": episode.hyperparameters,
                "convergence_metric": episode.convergence_metric,
                "breakthrough_detected": episode.breakthrough_detected,
                "timestamp": episode.timestamp.isoformat(),
                "iceburg_project_id": iceburg_project_id
            }
            
            # Log to unified memory
            self.unified_memory.log_event("rl_episode", event_data)
            
            # Index in vector store
            self._index_rl_episode(episode)
            
            logger.debug(f"Logged RL episode: {episode.episode_id}")
            
        except Exception as e:
            logger.error(f"Error logging RL episode: {e}")
    
    def log_financial_prediction(self, prediction: FinancialPrediction,
                                iceburg_project_id: Optional[str] = None):
        """
        Log financial prediction to memory system.
        
        Args:
            prediction: Financial prediction record
            iceburg_project_id: ICEBURG project ID
        """
        try:
            # Create event log entry
            event_data = {
                "event_type": "financial_prediction",
                "prediction_id": prediction.prediction_id,
                "symbol": prediction.symbol,
                "prediction_type": prediction.prediction_type,
                "value": prediction.value,
                "confidence": prediction.confidence,
                "model_type": prediction.model_type,
                "features_used": prediction.features_used,
                "actual_value": prediction.actual_value,
                "accuracy": prediction.accuracy,
                "timestamp": prediction.timestamp.isoformat(),
                "iceburg_project_id": iceburg_project_id
            }
            
            # Log to unified memory
            self.unified_memory.log_event("financial_prediction", event_data)
            
            # Index in vector store
            self._index_financial_prediction(prediction)
            
            logger.debug(f"Logged financial prediction: {prediction.prediction_id}")
            
        except Exception as e:
            logger.error(f"Error logging financial prediction: {e}")
    
    def log_model_checkpoint(self, checkpoint: ModelCheckpoint,
                            iceburg_project_id: Optional[str] = None):
        """
        Log model checkpoint to memory system.
        
        Args:
            checkpoint: Model checkpoint record
            iceburg_project_id: ICEBURG project ID
        """
        try:
            # Create event log entry
            event_data = {
                "event_type": "model_checkpoint",
                "checkpoint_id": checkpoint.checkpoint_id,
                "model_type": checkpoint.model_type,
                "performance_metrics": checkpoint.performance_metrics,
                "training_epoch": checkpoint.training_epoch,
                "validation_score": checkpoint.validation_score,
                "model_size": checkpoint.model_size,
                "hyperparameters": checkpoint.hyperparameters,
                "timestamp": checkpoint.timestamp.isoformat(),
                "iceburg_project_id": iceburg_project_id
            }
            
            # Log to unified memory
            self.unified_memory.log_event("model_checkpoint", event_data)
            
            # Index in vector store
            self._index_model_checkpoint(checkpoint)
            
            logger.debug(f"Logged model checkpoint: {checkpoint.checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Error logging model checkpoint: {e}")
    
    def log_quantum_rl_experiment(self, experiment: QuantumRLExperiment,
                                 iceburg_project_id: Optional[str] = None):
        """
        Log quantum-RL experiment to memory system.
        
        Args:
            experiment: Quantum-RL experiment record
            iceburg_project_id: ICEBURG project ID
        """
        try:
            # Create event log entry
            event_data = {
                "event_type": "quantum_rl_experiment",
                "experiment_id": experiment.experiment_id,
                "config": experiment.config,
                "results": experiment.results,
                "breakthrough_detected": experiment.breakthrough_detected,
                "duration": experiment.duration,
                "success": experiment.success,
                "error_message": experiment.error_message,
                "timestamp": experiment.timestamp.isoformat(),
                "iceburg_project_id": iceburg_project_id
            }
            
            # Log to unified memory
            self.unified_memory.log_event("quantum_rl_experiment", event_data)
            
            # Index in vector store
            self._index_quantum_rl_experiment(experiment)
            
            logger.debug(f"Logged quantum-RL experiment: {experiment.experiment_id}")
            
        except Exception as e:
            logger.error(f"Error logging quantum-RL experiment: {e}")
    
    def _index_quantum_execution(self, execution: QuantumCircuitExecution):
        """Index quantum execution in vector store."""
        try:
            if self.quantum_vector_store is None:
                return
            
            # Create embedding from execution data
            embedding_data = {
                "circuit_type": execution.circuit_type,
                "n_qubits": execution.n_qubits,
                "n_layers": execution.n_layers,
                "device": execution.device,
                "success": execution.success
            }
            
            # Create text representation for embedding
            text = f"Quantum circuit {execution.circuit_type} with {execution.n_qubits} qubits, {execution.n_layers} layers, device {execution.device}, success {execution.success}"
            
            # Add to vector store
            self.quantum_vector_store.add(
                documents=[text],
                metadatas=[embedding_data],
                ids=[execution.circuit_id]
            )
            
        except Exception as e:
            logger.error(f"Error indexing quantum execution: {e}")
    
    def _index_rl_episode(self, episode: RLTrainingEpisode):
        """Index RL episode in vector store."""
        try:
            if self.rl_vector_store is None:
                return
            
            # Create embedding from episode data
            embedding_data = {
                "agent_name": episode.agent_name,
                "environment": episode.environment,
                "algorithm": episode.algorithm,
                "reward": episode.reward,
                "steps": episode.steps,
                "breakthrough_detected": episode.breakthrough_detected
            }
            
            # Create text representation for embedding
            text = f"RL episode {episode.agent_name} in {episode.environment} using {episode.algorithm}, reward {episode.reward}, steps {episode.steps}, breakthrough {episode.breakthrough_detected}"
            
            # Add to vector store
            self.rl_vector_store.add(
                documents=[text],
                metadatas=[embedding_data],
                ids=[episode.episode_id]
            )
            
        except Exception as e:
            logger.error(f"Error indexing RL episode: {e}")
    
    def _index_financial_prediction(self, prediction: FinancialPrediction):
        """Index financial prediction in vector store."""
        try:
            if self.financial_vector_store is None:
                return
            
            # Create embedding from prediction data
            embedding_data = {
                "symbol": prediction.symbol,
                "prediction_type": prediction.prediction_type,
                "model_type": prediction.model_type,
                "confidence": prediction.confidence,
                "accuracy": prediction.accuracy
            }
            
            # Create text representation for embedding
            text = f"Financial prediction for {prediction.symbol} {prediction.prediction_type} using {prediction.model_type}, confidence {prediction.confidence}, accuracy {prediction.accuracy}"
            
            # Add to vector store
            self.financial_vector_store.add(
                documents=[text],
                metadatas=[embedding_data],
                ids=[prediction.prediction_id]
            )
            
        except Exception as e:
            logger.error(f"Error indexing financial prediction: {e}")
    
    def _index_model_checkpoint(self, checkpoint: ModelCheckpoint):
        """Index model checkpoint in vector store."""
        try:
            if self.experiment_vector_store is None:
                return
            
            # Create embedding from checkpoint data
            embedding_data = {
                "model_type": checkpoint.model_type,
                "training_epoch": checkpoint.training_epoch,
                "validation_score": checkpoint.validation_score,
                "model_size": checkpoint.model_size
            }
            
            # Create text representation for embedding
            text = f"Model checkpoint {checkpoint.model_type} at epoch {checkpoint.training_epoch}, validation score {checkpoint.validation_score}, size {checkpoint.model_size}"
            
            # Add to vector store
            self.experiment_vector_store.add(
                documents=[text],
                metadatas=[embedding_data],
                ids=[checkpoint.checkpoint_id]
            )
            
        except Exception as e:
            logger.error(f"Error indexing model checkpoint: {e}")
    
    def _index_quantum_rl_experiment(self, experiment: QuantumRLExperiment):
        """Index quantum-RL experiment in vector store."""
        try:
            if self.experiment_vector_store is None:
                return
            
            # Create embedding from experiment data
            embedding_data = {
                "breakthrough_detected": experiment.breakthrough_detected,
                "success": experiment.success,
                "duration": experiment.duration
            }
            
            # Create text representation for embedding
            text = f"Quantum-RL experiment {experiment.experiment_id}, breakthrough {experiment.breakthrough_detected}, success {experiment.success}, duration {experiment.duration}"
            
            # Add to vector store
            self.experiment_vector_store.add(
                documents=[text],
                metadatas=[embedding_data],
                ids=[experiment.experiment_id]
            )
            
        except Exception as e:
            logger.error(f"Error indexing quantum-RL experiment: {e}")
    
    def search_similar_quantum_executions(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar quantum executions.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar quantum executions
        """
        try:
            if self.quantum_vector_store is None:
                return []
            
            # Search vector store
            results = self.quantum_vector_store.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            similar_executions = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similar_executions.append({
                    "document": doc,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": i + 1
                })
            
            return similar_executions
            
        except Exception as e:
            logger.error(f"Error searching similar quantum executions: {e}")
            return []
    
    def search_similar_rl_episodes(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar RL episodes.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar RL episodes
        """
        try:
            if self.rl_vector_store is None:
                return []
            
            # Search vector store
            results = self.rl_vector_store.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            similar_episodes = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similar_episodes.append({
                    "document": doc,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": i + 1
                })
            
            return similar_episodes
            
        except Exception as e:
            logger.error(f"Error searching similar RL episodes: {e}")
            return []
    
    def search_similar_financial_predictions(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar financial predictions.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar financial predictions
        """
        try:
            if self.financial_vector_store is None:
                return []
            
            # Search vector store
            results = self.financial_vector_store.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            similar_predictions = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similar_predictions.append({
                    "document": doc,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": i + 1
                })
            
            return similar_predictions
            
        except Exception as e:
            logger.error(f"Error searching similar financial predictions: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Memory system statistics
        """
        try:
            stats = {
                "memory_dir": self.memory_dir,
                "vector_dir": self.vector_dir,
                "vector_stores": {
                    "quantum_circuits": self.quantum_vector_store.count() if self.quantum_vector_store else 0,
                    "rl_episodes": self.rl_vector_store.count() if self.rl_vector_store else 0,
                    "financial_predictions": self.financial_vector_store.count() if self.financial_vector_store else 0,
                    "experiments": self.experiment_vector_store.count() if self.experiment_vector_store else 0
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def close(self):
        """Close memory integration."""
        if self.unified_memory:
            self.unified_memory.close()


# Example usage and testing
if __name__ == "__main__":
    # Test Elite Financial AI memory integration
    memory_integration = EliteMemoryIntegration()
    
    # Test quantum execution logging
    quantum_execution = QuantumCircuitExecution(
        circuit_id="test_circuit_001",
        circuit_type="variational",
        parameters={"n_qubits": 4, "n_layers": 2},
        execution_time=0.123,
        result=[0.5, -0.3, 0.8, -0.1],
        timestamp=datetime.now(),
        n_qubits=4,
        n_layers=2,
        device="default.qubit",
        shots=1000,
        success=True
    )
    
    memory_integration.log_quantum_execution(quantum_execution, "project_001")
    # Logged quantum execution
    
    # Test RL episode logging
    rl_episode = RLTrainingEpisode(
        episode_id="test_episode_001",
        agent_name="PPO_Trader",
        environment="TradingEnv",
        reward=150.5,
        steps=1000,
        timestamp=datetime.now(),
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.001, "batch_size": 256},
        convergence_metric=0.85,
        breakthrough_detected=False
    )
    
    memory_integration.log_rl_episode(rl_episode, "project_001")
    # Logged RL episode
    
    # Test financial prediction logging
    financial_prediction = FinancialPrediction(
        prediction_id="test_prediction_001",
        symbol="AAPL",
        prediction_type="price",
        value=150.25,
        confidence=0.85,
        timestamp=datetime.now(),
        model_type="quantum_rl",
        features_used=["price", "volume", "rsi", "macd"],
        actual_value=152.30,
        accuracy=0.87
    )
    
    memory_integration.log_financial_prediction(financial_prediction, "project_001")
    # Logged financial prediction
    
    # Test model checkpoint logging
    model_checkpoint = ModelCheckpoint(
        checkpoint_id="test_checkpoint_001",
        model_type="quantum_rl",
        model_state={"weights": [0.1, 0.2, 0.3], "biases": [0.01, 0.02]},
        performance_metrics={"accuracy": 0.85, "loss": 0.15},
        timestamp=datetime.now(),
        training_epoch=100,
        validation_score=0.87,
        model_size=1024,
        hyperparameters={"learning_rate": 0.001, "batch_size": 256}
    )
    
    memory_integration.log_model_checkpoint(model_checkpoint, "project_001")
    # Logged model checkpoint
    
    # Test quantum-RL experiment logging
    quantum_rl_experiment = QuantumRLExperiment(
        experiment_id="test_experiment_001",
        config={"n_qubits": 4, "n_layers": 2, "algorithm": "PPO"},
        results={"reward": 150.5, "accuracy": 0.85, "convergence": 0.9},
        breakthrough_detected=True,
        timestamp=datetime.now(),
        duration=3600.0,
        success=True
    )
    
    memory_integration.log_quantum_rl_experiment(quantum_rl_experiment, "project_001")
    # Logged quantum-RL experiment
    
    # Test similarity search
    similar_quantum = memory_integration.search_similar_quantum_executions("variational circuit", n_results=3)
    # Found similar quantum executions
    
    similar_rl = memory_integration.search_similar_rl_episodes("PPO trader", n_results=3)
    
    similar_financial = memory_integration.search_similar_financial_predictions("AAPL price prediction", n_results=3)
    
    # Test memory statistics
    stats = memory_integration.get_memory_stats()
    
    # Close memory integration
    memory_integration.close()
