"""
Elite Financial AI Database Schema for ICEBURG

This module defines the database schema for Elite Financial AI operations,
including quantum circuit executions, RL training episodes, financial predictions,
model checkpoints, and quantum-RL experiments.
"""

import sqlite3
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumCircuitExecution:
    """Quantum circuit execution record."""
    circuit_id: str
    circuit_type: str
    parameters: Dict[str, Any]
    execution_time: float
    result: List[float]
    timestamp: datetime
    n_qubits: int
    n_layers: int
    device: str
    shots: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class RLTrainingEpisode:
    """RL training episode record."""
    episode_id: str
    agent_name: str
    environment: str
    reward: float
    steps: int
    timestamp: datetime
    algorithm: str
    hyperparameters: Dict[str, Any]
    convergence_metric: Optional[float] = None
    breakthrough_detected: bool = False


@dataclass
class FinancialPrediction:
    """Financial prediction record."""
    prediction_id: str
    symbol: str
    prediction_type: str
    value: float
    confidence: float
    timestamp: datetime
    model_type: str
    features_used: List[str]
    actual_value: Optional[float] = None
    accuracy: Optional[float] = None


@dataclass
class ModelCheckpoint:
    """Model checkpoint record."""
    checkpoint_id: str
    model_type: str
    model_state: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: datetime
    training_epoch: int
    validation_score: float
    model_size: int
    hyperparameters: Dict[str, Any]


@dataclass
class QuantumRLExperiment:
    """Quantum-RL experiment record."""
    experiment_id: str
    config: Dict[str, Any]
    results: Dict[str, Any]
    breakthrough_detected: bool
    timestamp: datetime
    duration: float
    success: bool
    error_message: Optional[str] = None


class EliteFinancialSchema:
    """
    Database schema for Elite Financial AI operations.
    
    Provides database table definitions and operations for storing
    quantum circuit executions, RL training episodes, financial predictions,
    model checkpoints, and quantum-RL experiments.
    """
    
    def __init__(self, db_path: str = "iceburg_unified.db"):
        """
        Initialize Elite Financial schema.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.connection = None
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for Elite Financial AI."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            cursor = self.connection.cursor()
            
            # Quantum circuit executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quantum_circuit_executions (
                    circuit_id TEXT PRIMARY KEY,
                    circuit_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    result TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    n_qubits INTEGER NOT NULL,
                    n_layers INTEGER NOT NULL,
                    device TEXT NOT NULL,
                    shots INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT
                )
            """)
            
            # RL training episodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_training_episodes (
                    episode_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    reward REAL NOT NULL,
                    steps INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    convergence_metric REAL,
                    breakthrough_detected BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Financial predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS financial_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    features_used TEXT NOT NULL,
                    actual_value REAL,
                    accuracy REAL
                )
            """)
            
            # Model checkpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    model_state TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    training_epoch INTEGER NOT NULL,
                    validation_score REAL NOT NULL,
                    model_size INTEGER NOT NULL,
                    hyperparameters TEXT NOT NULL
                )
            """)
            
            # Quantum-RL experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quantum_rl_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    results TEXT NOT NULL,
                    breakthrough_detected BOOLEAN NOT NULL,
                    timestamp TEXT NOT NULL,
                    duration REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quantum_timestamp ON quantum_circuit_executions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quantum_type ON quantum_circuit_executions(circuit_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_timestamp ON rl_training_episodes(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_agent ON rl_training_episodes(agent_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_financial_timestamp ON financial_predictions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_financial_symbol ON financial_predictions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoint_timestamp ON model_checkpoints(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiment_timestamp ON quantum_rl_experiments(timestamp)")
            
            self.connection.commit()
            logger.info("Elite Financial AI database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def insert_quantum_execution(self, execution: QuantumCircuitExecution):
        """
        Insert quantum circuit execution record.
        
        Args:
            execution: Quantum circuit execution record
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO quantum_circuit_executions
                (circuit_id, circuit_type, parameters, execution_time, result, timestamp,
                 n_qubits, n_layers, device, shots, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution.circuit_id,
                execution.circuit_type,
                json.dumps(execution.parameters),
                execution.execution_time,
                json.dumps(execution.result),
                execution.timestamp.isoformat(),
                execution.n_qubits,
                execution.n_layers,
                execution.device,
                execution.shots,
                execution.success,
                execution.error_message
            ))
            self.connection.commit()
            logger.debug(f"Inserted quantum execution: {execution.circuit_id}")
        except Exception as e:
            logger.error(f"Error inserting quantum execution: {e}")
            raise
    
    def insert_rl_episode(self, episode: RLTrainingEpisode):
        """
        Insert RL training episode record.
        
        Args:
            episode: RL training episode record
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO rl_training_episodes
                (episode_id, agent_name, environment, reward, steps, timestamp,
                 algorithm, hyperparameters, convergence_metric, breakthrough_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.episode_id,
                episode.agent_name,
                episode.environment,
                episode.reward,
                episode.steps,
                episode.timestamp.isoformat(),
                episode.algorithm,
                json.dumps(episode.hyperparameters),
                episode.convergence_metric,
                episode.breakthrough_detected
            ))
            self.connection.commit()
            logger.debug(f"Inserted RL episode: {episode.episode_id}")
        except Exception as e:
            logger.error(f"Error inserting RL episode: {e}")
            raise
    
    def insert_financial_prediction(self, prediction: FinancialPrediction):
        """
        Insert financial prediction record.
        
        Args:
            prediction: Financial prediction record
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO financial_predictions
                (prediction_id, symbol, prediction_type, value, confidence, timestamp,
                 model_type, features_used, actual_value, accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.prediction_id,
                prediction.symbol,
                prediction.prediction_type,
                prediction.value,
                prediction.confidence,
                prediction.timestamp.isoformat(),
                prediction.model_type,
                json.dumps(prediction.features_used),
                prediction.actual_value,
                prediction.accuracy
            ))
            self.connection.commit()
            logger.debug(f"Inserted financial prediction: {prediction.prediction_id}")
        except Exception as e:
            logger.error(f"Error inserting financial prediction: {e}")
            raise
    
    def insert_model_checkpoint(self, checkpoint: ModelCheckpoint):
        """
        Insert model checkpoint record.
        
        Args:
            checkpoint: Model checkpoint record
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO model_checkpoints
                (checkpoint_id, model_type, model_state, performance_metrics, timestamp,
                 training_epoch, validation_score, model_size, hyperparameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint.checkpoint_id,
                checkpoint.model_type,
                json.dumps(checkpoint.model_state),
                json.dumps(checkpoint.performance_metrics),
                checkpoint.timestamp.isoformat(),
                checkpoint.training_epoch,
                checkpoint.validation_score,
                checkpoint.model_size,
                json.dumps(checkpoint.hyperparameters)
            ))
            self.connection.commit()
            logger.debug(f"Inserted model checkpoint: {checkpoint.checkpoint_id}")
        except Exception as e:
            logger.error(f"Error inserting model checkpoint: {e}")
            raise
    
    def insert_quantum_rl_experiment(self, experiment: QuantumRLExperiment):
        """
        Insert quantum-RL experiment record.
        
        Args:
            experiment: Quantum-RL experiment record
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO quantum_rl_experiments
                (experiment_id, config, results, breakthrough_detected, timestamp,
                 duration, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.experiment_id,
                json.dumps(experiment.config),
                json.dumps(experiment.results),
                experiment.breakthrough_detected,
                experiment.timestamp.isoformat(),
                experiment.duration,
                experiment.success,
                experiment.error_message
            ))
            self.connection.commit()
            logger.debug(f"Inserted quantum-RL experiment: {experiment.experiment_id}")
        except Exception as e:
            logger.error(f"Error inserting quantum-RL experiment: {e}")
            raise
    
    def get_quantum_executions(self, limit: int = 100, 
                             circuit_type: Optional[str] = None) -> List[QuantumCircuitExecution]:
        """
        Get quantum circuit executions.
        
        Args:
            limit: Maximum number of records
            circuit_type: Filter by circuit type
            
        Returns:
            List of quantum circuit executions
        """
        try:
            cursor = self.connection.cursor()
            
            if circuit_type:
                cursor.execute("""
                    SELECT * FROM quantum_circuit_executions
                    WHERE circuit_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (circuit_type, limit))
            else:
                cursor.execute("""
                    SELECT * FROM quantum_circuit_executions
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            executions = []
            
            for row in rows:
                execution = QuantumCircuitExecution(
                    circuit_id=row[0],
                    circuit_type=row[1],
                    parameters=json.loads(row[2]),
                    execution_time=row[3],
                    result=json.loads(row[4]),
                    timestamp=datetime.fromisoformat(row[5]),
                    n_qubits=row[6],
                    n_layers=row[7],
                    device=row[8],
                    shots=row[9],
                    success=bool(row[10]),
                    error_message=row[11]
                )
                executions.append(execution)
            
            return executions
            
        except Exception as e:
            logger.error(f"Error getting quantum executions: {e}")
            return []
    
    def get_rl_episodes(self, limit: int = 100, 
                       agent_name: Optional[str] = None) -> List[RLTrainingEpisode]:
        """
        Get RL training episodes.
        
        Args:
            limit: Maximum number of records
            agent_name: Filter by agent name
            
        Returns:
            List of RL training episodes
        """
        try:
            cursor = self.connection.cursor()
            
            if agent_name:
                cursor.execute("""
                    SELECT * FROM rl_training_episodes
                    WHERE agent_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (agent_name, limit))
            else:
                cursor.execute("""
                    SELECT * FROM rl_training_episodes
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            episodes = []
            
            for row in rows:
                episode = RLTrainingEpisode(
                    episode_id=row[0],
                    agent_name=row[1],
                    environment=row[2],
                    reward=row[3],
                    steps=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    algorithm=row[6],
                    hyperparameters=json.loads(row[7]),
                    convergence_metric=row[8],
                    breakthrough_detected=bool(row[9])
                )
                episodes.append(episode)
            
            return episodes
            
        except Exception as e:
            logger.error(f"Error getting RL episodes: {e}")
            return []
    
    def get_financial_predictions(self, limit: int = 100, 
                                 symbol: Optional[str] = None) -> List[FinancialPrediction]:
        """
        Get financial predictions.
        
        Args:
            limit: Maximum number of records
            symbol: Filter by symbol
            
        Returns:
            List of financial predictions
        """
        try:
            cursor = self.connection.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM financial_predictions
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM financial_predictions
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            predictions = []
            
            for row in rows:
                prediction = FinancialPrediction(
                    prediction_id=row[0],
                    symbol=row[1],
                    prediction_type=row[2],
                    value=row[3],
                    confidence=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    model_type=row[6],
                    features_used=json.loads(row[7]),
                    actual_value=row[8],
                    accuracy=row[9]
                )
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting financial predictions: {e}")
            return []
    
    def get_breakthrough_experiments(self, limit: int = 50) -> List[QuantumRLExperiment]:
        """
        Get breakthrough quantum-RL experiments.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of breakthrough experiments
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM quantum_rl_experiments
                WHERE breakthrough_detected = TRUE
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            experiments = []
            
            for row in rows:
                experiment = QuantumRLExperiment(
                    experiment_id=row[0],
                    config=json.loads(row[1]),
                    results=json.loads(row[2]),
                    breakthrough_detected=bool(row[3]),
                    timestamp=datetime.fromisoformat(row[4]),
                    duration=row[5],
                    success=bool(row[6]),
                    error_message=row[7]
                )
                experiments.append(experiment)
            
            return experiments
            
        except Exception as e:
            logger.error(f"Error getting breakthrough experiments: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from database.
        
        Returns:
            Performance metrics
        """
        try:
            cursor = self.connection.cursor()
            
            # Quantum execution metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    AVG(execution_time) as avg_execution_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions
                FROM quantum_circuit_executions
            """)
            quantum_metrics = cursor.fetchone()
            
            # RL training metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_episodes,
                    AVG(reward) as avg_reward,
                    SUM(CASE WHEN breakthrough_detected = 1 THEN 1 ELSE 0 END) as breakthrough_episodes
                FROM rl_training_episodes
            """)
            rl_metrics = cursor.fetchone()
            
            # Financial prediction metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(accuracy) as avg_accuracy
                FROM financial_predictions
                WHERE accuracy IS NOT NULL
            """)
            financial_metrics = cursor.fetchone()
            
            return {
                "quantum_executions": {
                    "total": quantum_metrics[0],
                    "avg_execution_time": quantum_metrics[1],
                    "success_rate": quantum_metrics[2] / quantum_metrics[0] if quantum_metrics[0] > 0 else 0
                },
                "rl_training": {
                    "total_episodes": rl_metrics[0],
                    "avg_reward": rl_metrics[1],
                    "breakthrough_rate": rl_metrics[2] / rl_metrics[0] if rl_metrics[0] > 0 else 0
                },
                "financial_predictions": {
                    "total_predictions": financial_metrics[0],
                    "avg_confidence": financial_metrics[1],
                    "avg_accuracy": financial_metrics[2]
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


# Example usage and testing
if __name__ == "__main__":
    # Test Elite Financial schema
    schema = EliteFinancialSchema("test_elite_financial.db")
    
    # Test quantum circuit execution
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
    
    schema.insert_quantum_execution(quantum_execution)
    # Quantum execution inserted
    
    # Test RL episode
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
    
    schema.insert_rl_episode(rl_episode)
    
    # Test financial prediction
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
    
    schema.insert_financial_prediction(financial_prediction)
    
    # Test model checkpoint
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
    
    schema.insert_model_checkpoint(model_checkpoint)
    
    # Test quantum-RL experiment
    quantum_rl_experiment = QuantumRLExperiment(
        experiment_id="test_experiment_001",
        config={"n_qubits": 4, "n_layers": 2, "algorithm": "PPO"},
        results={"reward": 150.5, "accuracy": 0.85, "convergence": 0.9},
        breakthrough_detected=True,
        timestamp=datetime.now(),
        duration=3600.0,
        success=True
    )
    
    schema.insert_quantum_rl_experiment(quantum_rl_experiment)
    
    # Test retrieval
    quantum_executions = schema.get_quantum_executions(limit=10)
    rl_episodes = schema.get_rl_episodes(limit=10)
    financial_predictions = schema.get_financial_predictions(limit=10)
    breakthrough_experiments = schema.get_breakthrough_experiments(limit=10)
    
    # Test performance metrics
    metrics = schema.get_performance_metrics()
    
    # Close schema
    schema.close()
