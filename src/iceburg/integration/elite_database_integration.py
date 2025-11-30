"""
Elite Financial AI Database Integration for ICEBURG

This module integrates Elite Financial AI operations with ICEBURG's unified database system,
providing comprehensive data storage and retrieval for quantum circuits, RL training,
financial predictions, and model checkpoints.
"""

import sqlite3
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging
import uuid
import time

from ..database.elite_financial_schema import (
    EliteFinancialSchema,
    QuantumCircuitExecution,
    RLTrainingEpisode,
    FinancialPrediction,
    ModelCheckpoint,
    QuantumRLExperiment
)

logger = logging.getLogger(__name__)


class EliteDatabaseIntegration:
    """
    Elite Financial AI database integration with ICEBURG.
    
    Provides comprehensive database operations for Elite Financial AI,
    including quantum circuit executions, RL training episodes, financial predictions,
    and model checkpoints with ICEBURG protocol integration.
    """
    
    def __init__(self, db_path: str = "iceburg_unified.db"):
        """
        Initialize Elite Financial AI database integration.
        
        Args:
            db_path: Path to ICEBURG unified database
        """
        self.db_path = db_path
        self.schema = EliteFinancialSchema(db_path)
        self.connection = sqlite3.connect(db_path)
        self._setup_integration_tables()
    
    def _setup_integration_tables(self):
        """Setup integration tables with ICEBURG system."""
        try:
            cursor = self.connection.cursor()
            
            # Create integration tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS elite_financial_integration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    iceburg_project_id TEXT,
                    iceburg_research_id TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS elite_financial_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT,
                    tags TEXT
                )
            """)
            
            # Create data flow tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS elite_financial_data_flow (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_size INTEGER,
                    processing_time REAL,
                    success BOOLEAN
                )
            """)
            
            self.connection.commit()
            logger.info("Elite Financial AI integration tables created")
            
        except Exception as e:
            logger.error(f"Error setting up integration tables: {e}")
            raise
    
    def record_quantum_execution(self, circuit_type: str, parameters: Dict[str, Any],
                               execution_time: float, result: List[float],
                               n_qubits: int, n_layers: int, device: str, shots: int,
                               success: bool = True, error_message: Optional[str] = None,
                               iceburg_project_id: Optional[str] = None) -> str:
        """
        Record quantum circuit execution with ICEBURG integration.
        
        Args:
            circuit_type: Type of quantum circuit
            parameters: Circuit parameters
            execution_time: Execution time in seconds
            result: Circuit execution result
            n_qubits: Number of qubits
            n_layers: Number of layers
            device: Quantum device used
            shots: Number of shots
            success: Whether execution was successful
            error_message: Error message if failed
            iceburg_project_id: ICEBURG project ID
            
        Returns:
            Circuit execution ID
        """
        circuit_id = f"quantum_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now()
        
        # Create quantum execution record
        execution = QuantumCircuitExecution(
            circuit_id=circuit_id,
            circuit_type=circuit_type,
            parameters=parameters,
            execution_time=execution_time,
            result=result,
            timestamp=timestamp,
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=device,
            shots=shots,
            success=success,
            error_message=error_message
        )
        
        # Insert into database
        self.schema.insert_quantum_execution(execution)
        
        # Record integration
        self._record_integration("quantum_execution", circuit_id, iceburg_project_id, success)
        
        # Record performance metrics
        self._record_performance_metric("quantum_execution_time", execution_time, {
            "circuit_type": circuit_type,
            "n_qubits": n_qubits,
            "device": device
        })
        
        logger.info(f"Recorded quantum execution: {circuit_id}")
        return circuit_id
    
    def record_rl_episode(self, agent_name: str, environment: str, reward: float,
                         steps: int, algorithm: str, hyperparameters: Dict[str, Any],
                         convergence_metric: Optional[float] = None,
                         breakthrough_detected: bool = False,
                         iceburg_project_id: Optional[str] = None) -> str:
        """
        Record RL training episode with ICEBURG integration.
        
        Args:
            agent_name: Name of RL agent
            environment: Environment name
            reward: Episode reward
            steps: Number of steps
            algorithm: RL algorithm used
            hyperparameters: Training hyperparameters
            convergence_metric: Convergence metric
            breakthrough_detected: Whether breakthrough was detected
            iceburg_project_id: ICEBURG project ID
            
        Returns:
            Episode ID
        """
        episode_id = f"rl_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now()
        
        # Create RL episode record
        episode = RLTrainingEpisode(
            episode_id=episode_id,
            agent_name=agent_name,
            environment=environment,
            reward=reward,
            steps=steps,
            timestamp=timestamp,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            convergence_metric=convergence_metric,
            breakthrough_detected=breakthrough_detected
        )
        
        # Insert into database
        self.schema.insert_rl_episode(episode)
        
        # Record integration
        self._record_integration("rl_episode", episode_id, iceburg_project_id, True)
        
        # Record performance metrics
        self._record_performance_metric("rl_episode_reward", reward, {
            "agent_name": agent_name,
            "algorithm": algorithm,
            "breakthrough": breakthrough_detected
        })
        
        if convergence_metric is not None:
            self._record_performance_metric("rl_convergence", convergence_metric, {
                "agent_name": agent_name,
                "algorithm": algorithm
            })
        
        logger.info(f"Recorded RL episode: {episode_id}")
        return episode_id
    
    def record_financial_prediction(self, symbol: str, prediction_type: str,
                                   value: float, confidence: float, model_type: str,
                                   features_used: List[str], actual_value: Optional[float] = None,
                                   iceburg_project_id: Optional[str] = None) -> str:
        """
        Record financial prediction with ICEBURG integration.
        
        Args:
            symbol: Financial symbol
            prediction_type: Type of prediction
            value: Predicted value
            confidence: Prediction confidence
            model_type: Model type used
            features_used: Features used for prediction
            actual_value: Actual value (for accuracy calculation)
            iceburg_project_id: ICEBURG project ID
            
        Returns:
            Prediction ID
        """
        prediction_id = f"financial_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now()
        
        # Calculate accuracy if actual value is provided
        accuracy = None
        if actual_value is not None:
            accuracy = 1.0 - abs(value - actual_value) / abs(actual_value)
        
        # Create financial prediction record
        prediction = FinancialPrediction(
            prediction_id=prediction_id,
            symbol=symbol,
            prediction_type=prediction_type,
            value=value,
            confidence=confidence,
            timestamp=timestamp,
            model_type=model_type,
            features_used=features_used,
            actual_value=actual_value,
            accuracy=accuracy
        )
        
        # Insert into database
        self.schema.insert_financial_prediction(prediction)
        
        # Record integration
        self._record_integration("financial_prediction", prediction_id, iceburg_project_id, True)
        
        # Record performance metrics
        self._record_performance_metric("financial_prediction_confidence", confidence, {
            "symbol": symbol,
            "model_type": model_type
        })
        
        if accuracy is not None:
            self._record_performance_metric("financial_prediction_accuracy", accuracy, {
                "symbol": symbol,
                "model_type": model_type
            })
        
        logger.info(f"Recorded financial prediction: {prediction_id}")
        return prediction_id
    
    def store_model_checkpoint(self, model_type: str, model_state: Dict[str, Any],
                              performance_metrics: Dict[str, float], training_epoch: int,
                              validation_score: float, hyperparameters: Dict[str, Any],
                              iceburg_project_id: Optional[str] = None) -> str:
        """
        Store model checkpoint with ICEBURG integration.
        
        Args:
            model_type: Type of model
            model_state: Model state dictionary
            performance_metrics: Performance metrics
            training_epoch: Training epoch
            validation_score: Validation score
            hyperparameters: Model hyperparameters
            iceburg_project_id: ICEBURG project ID
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"checkpoint_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now()
        
        # Calculate model size
        model_size = len(json.dumps(model_state))
        
        # Create model checkpoint record
        checkpoint = ModelCheckpoint(
            checkpoint_id=checkpoint_id,
            model_type=model_type,
            model_state=model_state,
            performance_metrics=performance_metrics,
            timestamp=timestamp,
            training_epoch=training_epoch,
            validation_score=validation_score,
            model_size=model_size,
            hyperparameters=hyperparameters
        )
        
        # Insert into database
        self.schema.insert_model_checkpoint(checkpoint)
        
        # Record integration
        self._record_integration("model_checkpoint", checkpoint_id, iceburg_project_id, True)
        
        # Record performance metrics
        self._record_performance_metric("model_validation_score", validation_score, {
            "model_type": model_type,
            "epoch": training_epoch
        })
        
        for metric_name, metric_value in performance_metrics.items():
            self._record_performance_metric(f"model_{metric_name}", metric_value, {
                "model_type": model_type,
                "epoch": training_epoch
            })
        
        logger.info(f"Stored model checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def record_quantum_rl_experiment(self, config: Dict[str, Any], results: Dict[str, Any],
                                   breakthrough_detected: bool, duration: float,
                                   success: bool = True, error_message: Optional[str] = None,
                                   iceburg_project_id: Optional[str] = None) -> str:
        """
        Record quantum-RL experiment with ICEBURG integration.
        
        Args:
            config: Experiment configuration
            results: Experiment results
            breakthrough_detected: Whether breakthrough was detected
            duration: Experiment duration
            success: Whether experiment was successful
            error_message: Error message if failed
            iceburg_project_id: ICEBURG project ID
            
        Returns:
            Experiment ID
        """
        experiment_id = f"experiment_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now()
        
        # Create quantum-RL experiment record
        experiment = QuantumRLExperiment(
            experiment_id=experiment_id,
            config=config,
            results=results,
            breakthrough_detected=breakthrough_detected,
            timestamp=timestamp,
            duration=duration,
            success=success,
            error_message=error_message
        )
        
        # Insert into database
        self.schema.insert_quantum_rl_experiment(experiment)
        
        # Record integration
        self._record_integration("quantum_rl_experiment", experiment_id, iceburg_project_id, success)
        
        # Record performance metrics
        self._record_performance_metric("experiment_duration", duration, {
            "breakthrough": breakthrough_detected,
            "success": success
        })
        
        if breakthrough_detected:
            self._record_performance_metric("breakthrough_detected", 1.0, {
                "experiment_id": experiment_id
            })
        
        logger.info(f"Recorded quantum-RL experiment: {experiment_id}")
        return experiment_id
    
    def _record_integration(self, operation_type: str, operation_id: str,
                          iceburg_project_id: Optional[str], success: bool):
        """Record integration operation."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO elite_financial_integration
                (operation_type, operation_id, iceburg_project_id, timestamp, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                operation_type,
                operation_id,
                iceburg_project_id,
                datetime.now().isoformat(),
                "success" if success else "failed"
            ))
            self.connection.commit()
        except Exception as e:
            logger.error(f"Error recording integration: {e}")
    
    def _record_performance_metric(self, metric_name: str, metric_value: float,
                                 context: Dict[str, Any]):
        """Record performance metric."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO elite_financial_performance
                (metric_name, metric_value, timestamp, context, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric_name,
                metric_value,
                datetime.now().isoformat(),
                json.dumps(context),
                json.dumps(list(context.keys()))
            ))
            self.connection.commit()
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """
        Get integration statistics.
        
        Returns:
            Integration statistics
        """
        try:
            cursor = self.connection.cursor()
            
            # Get operation counts
            cursor.execute("""
                SELECT operation_type, COUNT(*) as count
                FROM elite_financial_integration
                GROUP BY operation_type
            """)
            operation_counts = dict(cursor.fetchall())
            
            # Get success rates
            cursor.execute("""
                SELECT operation_type, 
                       SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                       COUNT(*) as total
                FROM elite_financial_integration
                GROUP BY operation_type
            """)
            success_rates = {}
            for row in cursor.fetchall():
                success_rates[row[0]] = row[1] / row[2] if row[2] > 0 else 0
            
            # Get performance metrics
            cursor.execute("""
                SELECT metric_name, AVG(metric_value) as avg_value, COUNT(*) as count
                FROM elite_financial_performance
                GROUP BY metric_name
            """)
            performance_metrics = {}
            for row in cursor.fetchall():
                performance_metrics[row[0]] = {
                    "average": row[1],
                    "count": row[2]
                }
            
            return {
                "operation_counts": operation_counts,
                "success_rates": success_rates,
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting integration stats: {e}")
            return {}
    
    def get_breakthrough_analysis(self) -> Dict[str, Any]:
        """
        Get breakthrough analysis.
        
        Returns:
            Breakthrough analysis
        """
        try:
            # Get breakthrough experiments
            breakthrough_experiments = self.schema.get_breakthrough_experiments(limit=100)
            
            # Get breakthrough RL episodes
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT COUNT(*) as breakthrough_episodes
                FROM rl_training_episodes
                WHERE breakthrough_detected = TRUE
            """)
            breakthrough_episodes = cursor.fetchone()[0]
            
            # Analyze breakthrough patterns
            breakthrough_analysis = {
                "total_breakthroughs": len(breakthrough_experiments) + breakthrough_episodes,
                "breakthrough_experiments": len(breakthrough_experiments),
                "breakthrough_episodes": breakthrough_episodes,
                "recent_breakthroughs": [
                    {
                        "experiment_id": exp.experiment_id,
                        "timestamp": exp.timestamp.isoformat(),
                        "config": exp.config,
                        "results": exp.results
                    }
                    for exp in breakthrough_experiments[:10]
                ]
            }
            
            return breakthrough_analysis
            
        except Exception as e:
            logger.error(f"Error getting breakthrough analysis: {e}")
            return {}
    
    def close(self):
        """Close database connections."""
        if self.connection:
            self.connection.close()
        if self.schema:
            self.schema.close()


# Example usage and testing
if __name__ == "__main__":
    # Test Elite Financial AI database integration
    integration = EliteDatabaseIntegration("test_elite_financial.db")
    
    # Test quantum execution recording
    circuit_id = integration.record_quantum_execution(
        circuit_type="variational",
        parameters={"n_qubits": 4, "n_layers": 2},
        execution_time=0.123,
        result=[0.5, -0.3, 0.8, -0.1],
        n_qubits=4,
        n_layers=2,
        device="default.qubit",
        shots=1000,
        success=True,
        iceburg_project_id="project_001"
    )
    # Recorded quantum execution
    
    # Test RL episode recording
    episode_id = integration.record_rl_episode(
        agent_name="PPO_Trader",
        environment="TradingEnv",
        reward=150.5,
        steps=1000,
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.001, "batch_size": 256},
        convergence_metric=0.85,
        breakthrough_detected=False,
        iceburg_project_id="project_001"
    )
    # Recorded RL episode
    
    # Test financial prediction recording
    prediction_id = integration.record_financial_prediction(
        symbol="AAPL",
        prediction_type="price",
        value=150.25,
        confidence=0.85,
        model_type="quantum_rl",
        features_used=["price", "volume", "rsi", "macd"],
        actual_value=152.30,
        iceburg_project_id="project_001"
    )
    # Recorded financial prediction
    
    # Test model checkpoint storage
    checkpoint_id = integration.store_model_checkpoint(
        model_type="quantum_rl",
        model_state={"weights": [0.1, 0.2, 0.3], "biases": [0.01, 0.02]},
        performance_metrics={"accuracy": 0.85, "loss": 0.15},
        training_epoch=100,
        validation_score=0.87,
        hyperparameters={"learning_rate": 0.001, "batch_size": 256},
        iceburg_project_id="project_001"
    )
    # Stored model checkpoint
    
    # Test quantum-RL experiment recording
    experiment_id = integration.record_quantum_rl_experiment(
        config={"n_qubits": 4, "n_layers": 2, "algorithm": "PPO"},
        results={"reward": 150.5, "accuracy": 0.85, "convergence": 0.9},
        breakthrough_detected=True,
        duration=3600.0,
        success=True,
        iceburg_project_id="project_001"
    )
    # Recorded quantum-RL experiment
    
    # Test integration statistics
    stats = integration.get_integration_stats()
    
    # Test breakthrough analysis
    breakthrough_analysis = integration.get_breakthrough_analysis()
    
    # Close integration
    integration.close()
