"""
Elite Financial AI Training Data Collector for ICEBURG

This module collects training data from Elite Financial AI operations,
including successful quantum circuit patterns, RL episode trajectories,
and financial prediction features for model training and improvement.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
import os
import uuid
import time

from ..database.elite_financial_schema import (
    QuantumCircuitExecution,
    RLTrainingEpisode,
    FinancialPrediction,
    ModelCheckpoint,
    QuantumRLExperiment
)
from ..integration.elite_database_integration import EliteDatabaseIntegration
from ..integration.elite_memory_integration import EliteMemoryIntegration

logger = logging.getLogger(__name__)


class EliteTrainingCollector:
    """
    Elite Financial AI training data collector.
    
    Collects and processes training data from Elite Financial AI operations,
    including quantum circuits, RL episodes, and financial predictions
    for model training and improvement.
    """
    
    def __init__(self, db_path: str = "iceburg_unified.db", 
                 memory_dir: str = "data/memory", vector_dir: str = "data/vector_store",
                 training_data_dir: str = "data/training_data"):
        """
        Initialize Elite Financial AI training data collector.
        
        Args:
            db_path: Path to ICEBURG unified database
            memory_dir: Directory for memory storage
            vector_dir: Directory for vector storage
            training_data_dir: Directory for training data storage
        """
        self.db_path = db_path
        self.memory_dir = memory_dir
        self.vector_dir = vector_dir
        self.training_data_dir = training_data_dir
        self.database_integration = EliteDatabaseIntegration(db_path)
        self.memory_integration = EliteMemoryIntegration(memory_dir, vector_dir)
        self._setup_training_directories()
        self.collection_stats = {}
    
    def _setup_training_directories(self):
        """Setup training data directories."""
        try:
            os.makedirs(self.training_data_dir, exist_ok=True)
            os.makedirs(os.path.join(self.training_data_dir, "quantum_circuits"), exist_ok=True)
            os.makedirs(os.path.join(self.training_data_dir, "rl_episodes"), exist_ok=True)
            os.makedirs(os.path.join(self.training_data_dir, "financial_predictions"), exist_ok=True)
            os.makedirs(os.path.join(self.training_data_dir, "model_checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(self.training_data_dir, "quantum_rl_experiments"), exist_ok=True)
            logger.info("Training data directories created successfully")
        except Exception as e:
            logger.error(f"Error setting up training directories: {e}")
            raise
    
    def collect_quantum_circuit_patterns(self, time_window: timedelta = timedelta(days=7),
                                        success_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Collect successful quantum circuit patterns.
        
        Args:
            time_window: Time window for data collection
            success_threshold: Success threshold for pattern selection
            
        Returns:
            Collected quantum circuit patterns
        """
        try:
            start_time = datetime.now() - time_window
            
            # Get quantum circuit executions from database
            quantum_executions = self.database_integration.schema.get_quantum_executions(
                limit=1000
            )
            
            # Filter successful executions
            successful_executions = [
                exec for exec in quantum_executions
                if exec.success and exec.timestamp >= start_time
            ]
            
            # Extract patterns
            patterns = []
            for execution in successful_executions:
                pattern = {
                    "circuit_id": execution.circuit_id,
                    "circuit_type": execution.circuit_type,
                    "parameters": execution.parameters,
                    "execution_time": execution.execution_time,
                    "result": execution.result,
                    "n_qubits": execution.n_qubits,
                    "n_layers": execution.n_layers,
                    "device": execution.device,
                    "success": execution.success,
                    "timestamp": execution.timestamp.isoformat()
                }
                patterns.append(pattern)
            
            # Store patterns
            patterns_file = os.path.join(
                self.training_data_dir, "quantum_circuits", 
                f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(patterns_file, 'w') as f:
                json.dump(patterns, f, indent=2)
            
            # Update collection stats
            self.collection_stats["quantum_circuits"] = {
                "total_executions": len(quantum_executions),
                "successful_executions": len(successful_executions),
                "patterns_collected": len(patterns),
                "success_rate": len(successful_executions) / len(quantum_executions) if quantum_executions else 0
            }
            
            logger.info(f"Collected {len(patterns)} quantum circuit patterns")
            return {
                "patterns": patterns,
                "stats": self.collection_stats["quantum_circuits"],
                "file_path": patterns_file
            }
            
        except Exception as e:
            logger.error(f"Error collecting quantum circuit patterns: {e}")
            return {"error": str(e)}
    
    def collect_rl_episode_trajectories(self, time_window: timedelta = timedelta(days=7),
                                       breakthrough_only: bool = False) -> Dict[str, Any]:
        """
        Collect RL episode trajectories for imitation learning.
        
        Args:
            time_window: Time window for data collection
            breakthrough_only: Whether to collect only breakthrough episodes
            
        Returns:
            Collected RL episode trajectories
        """
        try:
            start_time = datetime.now() - time_window
            
            # Get RL episodes from database
            rl_episodes = self.database_integration.schema.get_rl_episodes(limit=1000)
            
            # Filter episodes
            if breakthrough_only:
                filtered_episodes = [
                    episode for episode in rl_episodes
                    if episode.breakthrough_detected and episode.timestamp >= start_time
                ]
            else:
                filtered_episodes = [
                    episode for episode in rl_episodes
                    if episode.timestamp >= start_time
                ]
            
            # Extract trajectories
            trajectories = []
            for episode in filtered_episodes:
                trajectory = {
                    "episode_id": episode.episode_id,
                    "agent_name": episode.agent_name,
                    "environment": episode.environment,
                    "reward": episode.reward,
                    "steps": episode.steps,
                    "algorithm": episode.algorithm,
                    "hyperparameters": episode.hyperparameters,
                    "convergence_metric": episode.convergence_metric,
                    "breakthrough_detected": episode.breakthrough_detected,
                    "timestamp": episode.timestamp.isoformat()
                }
                trajectories.append(trajectory)
            
            # Store trajectories
            trajectories_file = os.path.join(
                self.training_data_dir, "rl_episodes",
                f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(trajectories_file, 'w') as f:
                json.dump(trajectories, f, indent=2)
            
            # Update collection stats
            self.collection_stats["rl_episodes"] = {
                "total_episodes": len(rl_episodes),
                "filtered_episodes": len(filtered_episodes),
                "breakthrough_episodes": len([ep for ep in filtered_episodes if ep.breakthrough_detected]),
                "trajectories_collected": len(trajectories)
            }
            
            logger.info(f"Collected {len(trajectories)} RL episode trajectories")
            return {
                "trajectories": trajectories,
                "stats": self.collection_stats["rl_episodes"],
                "file_path": trajectories_file
            }
            
        except Exception as e:
            logger.error(f"Error collecting RL episode trajectories: {e}")
            return {"error": str(e)}
    
    def collect_financial_prediction_features(self, time_window: timedelta = timedelta(days=7),
                                            accuracy_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Collect financial prediction features.
        
        Args:
            time_window: Time window for data collection
            accuracy_threshold: Accuracy threshold for feature selection
            
        Returns:
            Collected financial prediction features
        """
        try:
            start_time = datetime.now() - time_window
            
            # Get financial predictions from database
            financial_predictions = self.database_integration.schema.get_financial_predictions(limit=1000)
            
            # Filter high-accuracy predictions
            filtered_predictions = [
                pred for pred in financial_predictions
                if pred.timestamp >= start_time and 
                (pred.accuracy is None or pred.accuracy >= accuracy_threshold)
            ]
            
            # Extract features
            features = []
            for prediction in filtered_predictions:
                feature = {
                    "prediction_id": prediction.prediction_id,
                    "symbol": prediction.symbol,
                    "prediction_type": prediction.prediction_type,
                    "value": prediction.value,
                    "confidence": prediction.confidence,
                    "model_type": prediction.model_type,
                    "features_used": prediction.features_used,
                    "actual_value": prediction.actual_value,
                    "accuracy": prediction.accuracy,
                    "timestamp": prediction.timestamp.isoformat()
                }
                features.append(feature)
            
            # Store features
            features_file = os.path.join(
                self.training_data_dir, "financial_predictions",
                f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(features_file, 'w') as f:
                json.dump(features, f, indent=2)
            
            # Update collection stats
            self.collection_stats["financial_predictions"] = {
                "total_predictions": len(financial_predictions),
                "filtered_predictions": len(filtered_predictions),
                "high_accuracy_predictions": len([pred for pred in filtered_predictions if pred.accuracy and pred.accuracy >= accuracy_threshold]),
                "features_collected": len(features)
            }
            
            logger.info(f"Collected {len(features)} financial prediction features")
            return {
                "features": features,
                "stats": self.collection_stats["financial_predictions"],
                "file_path": features_file
            }
            
        except Exception as e:
            logger.error(f"Error collecting financial prediction features: {e}")
            return {"error": str(e)}
    
    def collect_model_checkpoints(self, time_window: timedelta = timedelta(days=7),
                                 performance_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Collect model checkpoints for training.
        
        Args:
            time_window: Time window for data collection
            performance_threshold: Performance threshold for checkpoint selection
            
        Returns:
            Collected model checkpoints
        """
        try:
            start_time = datetime.now() - time_window
            
            # Get model checkpoints from database
            cursor = self.database_integration.connection.cursor()
            cursor.execute("""
                SELECT * FROM model_checkpoints
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (start_time.isoformat(),))
            
            rows = cursor.fetchall()
            checkpoints = []
            
            for row in rows:
                checkpoint = {
                    "checkpoint_id": row[0],
                    "model_type": row[1],
                    "model_state": json.loads(row[2]),
                    "performance_metrics": json.loads(row[3]),
                    "timestamp": row[4],
                    "training_epoch": row[5],
                    "validation_score": row[6],
                    "model_size": row[7],
                    "hyperparameters": json.loads(row[8])
                }
                checkpoints.append(checkpoint)
            
            # Filter high-performance checkpoints
            filtered_checkpoints = [
                ckpt for ckpt in checkpoints
                if ckpt["validation_score"] >= performance_threshold
            ]
            
            # Store checkpoints
            checkpoints_file = os.path.join(
                self.training_data_dir, "model_checkpoints",
                f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(checkpoints_file, 'w') as f:
                json.dump(filtered_checkpoints, f, indent=2)
            
            # Update collection stats
            self.collection_stats["model_checkpoints"] = {
                "total_checkpoints": len(checkpoints),
                "filtered_checkpoints": len(filtered_checkpoints),
                "high_performance_checkpoints": len(filtered_checkpoints)
            }
            
            logger.info(f"Collected {len(filtered_checkpoints)} model checkpoints")
            return {
                "checkpoints": filtered_checkpoints,
                "stats": self.collection_stats["model_checkpoints"],
                "file_path": checkpoints_file
            }
            
        except Exception as e:
            logger.error(f"Error collecting model checkpoints: {e}")
            return {"error": str(e)}
    
    def collect_quantum_rl_experiments(self, time_window: timedelta = timedelta(days=7),
                                      breakthrough_only: bool = True) -> Dict[str, Any]:
        """
        Collect quantum-RL experiments for training.
        
        Args:
            time_window: Time window for data collection
            breakthrough_only: Whether to collect only breakthrough experiments
            
        Returns:
            Collected quantum-RL experiments
        """
        try:
            start_time = datetime.now() - time_window
            
            # Get breakthrough experiments
            breakthrough_experiments = self.database_integration.schema.get_breakthrough_experiments(limit=100)
            
            # Filter experiments
            if breakthrough_only:
                filtered_experiments = [
                    exp for exp in breakthrough_experiments
                    if exp.timestamp >= start_time and exp.breakthrough_detected
                ]
            else:
                filtered_experiments = [
                    exp for exp in breakthrough_experiments
                    if exp.timestamp >= start_time
                ]
            
            # Extract experiment data
            experiments = []
            for experiment in filtered_experiments:
                exp_data = {
                    "experiment_id": experiment.experiment_id,
                    "config": experiment.config,
                    "results": experiment.results,
                    "breakthrough_detected": experiment.breakthrough_detected,
                    "timestamp": experiment.timestamp.isoformat(),
                    "duration": experiment.duration,
                    "success": experiment.success,
                    "error_message": experiment.error_message
                }
                experiments.append(exp_data)
            
            # Store experiments
            experiments_file = os.path.join(
                self.training_data_dir, "quantum_rl_experiments",
                f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(experiments_file, 'w') as f:
                json.dump(experiments, f, indent=2)
            
            # Update collection stats
            self.collection_stats["quantum_rl_experiments"] = {
                "total_experiments": len(breakthrough_experiments),
                "filtered_experiments": len(filtered_experiments),
                "breakthrough_experiments": len([exp for exp in filtered_experiments if exp.breakthrough_detected]),
                "experiments_collected": len(experiments)
            }
            
            logger.info(f"Collected {len(experiments)} quantum-RL experiments")
            return {
                "experiments": experiments,
                "stats": self.collection_stats["quantum_rl_experiments"],
                "file_path": experiments_file
            }
            
        except Exception as e:
            logger.error(f"Error collecting quantum-RL experiments: {e}")
            return {"error": str(e)}
    
    def generate_training_datasets(self, data_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate training datasets from collected data.
        
        Args:
            data_types: Types of data to include in datasets
            
        Returns:
            Generated training datasets
        """
        try:
            if data_types is None:
                data_types = ["quantum_circuits", "rl_episodes", "financial_predictions", "model_checkpoints"]
            
            datasets = {}
            
            for data_type in data_types:
                if data_type == "quantum_circuits":
                    dataset = self._generate_quantum_circuit_dataset()
                elif data_type == "rl_episodes":
                    dataset = self._generate_rl_episode_dataset()
                elif data_type == "financial_predictions":
                    dataset = self._generate_financial_prediction_dataset()
                elif data_type == "model_checkpoints":
                    dataset = self._generate_model_checkpoint_dataset()
                else:
                    continue
                
                datasets[data_type] = dataset
            
            # Store combined dataset
            combined_dataset_file = os.path.join(
                self.training_data_dir,
                f"combined_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(combined_dataset_file, 'w') as f:
                json.dump(datasets, f, indent=2)
            
            logger.info(f"Generated training datasets: {list(datasets.keys())}")
            return {
                "datasets": datasets,
                "combined_file": combined_dataset_file
            }
            
        except Exception as e:
            logger.error(f"Error generating training datasets: {e}")
            return {"error": str(e)}
    
    def _generate_quantum_circuit_dataset(self) -> Dict[str, Any]:
        """Generate quantum circuit training dataset."""
        try:
            # Load quantum circuit patterns
            patterns_dir = os.path.join(self.training_data_dir, "quantum_circuits")
            pattern_files = [f for f in os.listdir(patterns_dir) if f.endswith('.json')]
            
            if not pattern_files:
                return {"error": "No quantum circuit patterns found"}
            
            # Load latest patterns
            latest_file = max(pattern_files, key=lambda x: os.path.getctime(os.path.join(patterns_dir, x)))
            with open(os.path.join(patterns_dir, latest_file), 'r') as f:
                patterns = json.load(f)
            
            # Convert to training format
            training_data = []
            for pattern in patterns:
                training_sample = {
                    "input": pattern["parameters"],
                    "output": pattern["result"],
                    "metadata": {
                        "circuit_type": pattern["circuit_type"],
                        "n_qubits": pattern["n_qubits"],
                        "n_layers": pattern["n_layers"],
                        "device": pattern["device"]
                    }
                }
                training_data.append(training_sample)
            
            return {
                "type": "quantum_circuits",
                "samples": len(training_data),
                "data": training_data
            }
            
        except Exception as e:
            logger.error(f"Error generating quantum circuit dataset: {e}")
            return {"error": str(e)}
    
    def _generate_rl_episode_dataset(self) -> Dict[str, Any]:
        """Generate RL episode training dataset."""
        try:
            # Load RL episode trajectories
            trajectories_dir = os.path.join(self.training_data_dir, "rl_episodes")
            trajectory_files = [f for f in os.listdir(trajectories_dir) if f.endswith('.json')]
            
            if not trajectory_files:
                return {"error": "No RL episode trajectories found"}
            
            # Load latest trajectories
            latest_file = max(trajectory_files, key=lambda x: os.path.getctime(os.path.join(trajectories_dir, x)))
            with open(os.path.join(trajectories_dir, latest_file), 'r') as f:
                trajectories = json.load(f)
            
            # Convert to training format
            training_data = []
            for trajectory in trajectories:
                training_sample = {
                    "state": trajectory["hyperparameters"],
                    "action": trajectory["reward"],
                    "reward": trajectory["reward"],
                    "metadata": {
                        "agent_name": trajectory["agent_name"],
                        "environment": trajectory["environment"],
                        "algorithm": trajectory["algorithm"],
                        "breakthrough": trajectory["breakthrough_detected"]
                    }
                }
                training_data.append(training_sample)
            
            return {
                "type": "rl_episodes",
                "samples": len(training_data),
                "data": training_data
            }
            
        except Exception as e:
            logger.error(f"Error generating RL episode dataset: {e}")
            return {"error": str(e)}
    
    def _generate_financial_prediction_dataset(self) -> Dict[str, Any]:
        """Generate financial prediction training dataset."""
        try:
            # Load financial prediction features
            features_dir = os.path.join(self.training_data_dir, "financial_predictions")
            feature_files = [f for f in os.listdir(features_dir) if f.endswith('.json')]
            
            if not feature_files:
                return {"error": "No financial prediction features found"}
            
            # Load latest features
            latest_file = max(feature_files, key=lambda x: os.path.getctime(os.path.join(features_dir, x)))
            with open(os.path.join(features_dir, latest_file), 'r') as f:
                features = json.load(f)
            
            # Convert to training format
            training_data = []
            for feature in features:
                training_sample = {
                    "input": feature["features_used"],
                    "output": feature["value"],
                    "target": feature["actual_value"],
                    "metadata": {
                        "symbol": feature["symbol"],
                        "prediction_type": feature["prediction_type"],
                        "model_type": feature["model_type"],
                        "confidence": feature["confidence"],
                        "accuracy": feature["accuracy"]
                    }
                }
                training_data.append(training_sample)
            
            return {
                "type": "financial_predictions",
                "samples": len(training_data),
                "data": training_data
            }
            
        except Exception as e:
            logger.error(f"Error generating financial prediction dataset: {e}")
            return {"error": str(e)}
    
    def _generate_model_checkpoint_dataset(self) -> Dict[str, Any]:
        """Generate model checkpoint training dataset."""
        try:
            # Load model checkpoints
            checkpoints_dir = os.path.join(self.training_data_dir, "model_checkpoints")
            checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.json')]
            
            if not checkpoint_files:
                return {"error": "No model checkpoints found"}
            
            # Load latest checkpoints
            latest_file = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
            with open(os.path.join(checkpoints_dir, latest_file), 'r') as f:
                checkpoints = json.load(f)
            
            # Convert to training format
            training_data = []
            for checkpoint in checkpoints:
                training_sample = {
                    "model_state": checkpoint["model_state"],
                    "performance": checkpoint["performance_metrics"],
                    "metadata": {
                        "model_type": checkpoint["model_type"],
                        "training_epoch": checkpoint["training_epoch"],
                        "validation_score": checkpoint["validation_score"],
                        "model_size": checkpoint["model_size"]
                    }
                }
                training_data.append(training_sample)
            
            return {
                "type": "model_checkpoints",
                "samples": len(training_data),
                "data": training_data
            }
            
        except Exception as e:
            logger.error(f"Error generating model checkpoint dataset: {e}")
            return {"error": str(e)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.collection_stats
    
    def close(self):
        """Close training data collector."""
        if self.database_integration:
            self.database_integration.close()
        if self.memory_integration:
            self.memory_integration.close()


# Example usage and testing
if __name__ == "__main__":
    # Test Elite Financial AI training data collector
    collector = EliteTrainingCollector()
    
    # Test quantum circuit pattern collection
    quantum_patterns = collector.collect_quantum_circuit_patterns()
    # Collected quantum circuit patterns
    
    # Test RL episode trajectory collection
    rl_trajectories = collector.collect_rl_episode_trajectories()
    
    # Test financial prediction feature collection
    financial_features = collector.collect_financial_prediction_features()
    
    # Test model checkpoint collection
    model_checkpoints = collector.collect_model_checkpoints()
    
    # Test quantum-RL experiment collection
    quantum_rl_experiments = collector.collect_quantum_rl_experiments()
    
    # Test training dataset generation
    datasets = collector.generate_training_datasets()
    
    # Test collection statistics
    stats = collector.get_collection_stats()
    
    # Close collector
    collector.close()
