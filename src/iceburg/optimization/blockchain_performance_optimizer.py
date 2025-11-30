"""
Blockchain Verification Performance Optimizer
Improves blockchain verification performance for faster processing

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import IceburgConfig

logger = logging.getLogger(__name__)

@dataclass
class BlockPerformanceMetrics:
    """Performance metrics for blockchain operations"""
    operation_type: str  # "mining", "verification", "validation", "storage"
    processing_time: float
    block_size: int
    difficulty: int
    success: bool
    resource_usage: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

@dataclass
class BlockchainOptimization:
    """Blockchain optimization strategy"""
    optimization_id: str
    optimization_type: str  # "difficulty_adjustment", "parallel_mining", "batch_processing", "compression"
    expected_improvement: float
    implementation_effort: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MiningPool:
    """Mining pool for parallel processing"""
    pool_id: str
    workers: int
    current_difficulty: int
    total_hashes: int
    successful_blocks: int
    avg_mining_time: float
    is_active: bool = True

class BlockchainPerformanceOptimizer:
    """
    Optimizes blockchain verification performance for ICEBURG
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.data_dir = Path("data/optimization/blockchain_performance")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.metrics_file = self.data_dir / "blockchain_metrics.json"
        self.optimizations_file = self.data_dir / "blockchain_optimizations.json"
        self.pools_file = self.data_dir / "mining_pools.json"
        
        # Data structures
        self.performance_metrics: List[BlockPerformanceMetrics] = []
        self.blockchain_optimizations: Dict[str, BlockchainOptimization] = {}
        self.mining_pools: Dict[str, MiningPool] = {}
        
        # Performance tracking
        self.operation_history: List[Dict[str, Any]] = []
        self.performance_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Optimization parameters
        self.dynamic_difficulty = True
        self.parallel_mining = True
        self.batch_processing = True
        self.compression_enabled = True
        
        # Threading
        self.mining_executor = ThreadPoolExecutor(max_workers=4)
        self.verification_executor = ThreadPoolExecutor(max_workers=2)
        
        # Load existing data
        self._load_data()
        self._initialize_default_optimizations()
        self._initialize_mining_pools()
        
        logger.info("⛓️ Blockchain Performance Optimizer initialized")
    
    def optimize_mining_performance(
        self,
        data: Dict[str, Any],
        target_difficulty: int = None
    ) -> Dict[str, Any]:
        """Optimize mining performance using various techniques"""
        
        start_time = time.time()
        
        # Determine optimal difficulty
        if target_difficulty is None:
            target_difficulty = self._calculate_optimal_difficulty()
        
        # Choose optimization strategy
        optimization_strategy = self._select_mining_strategy(target_difficulty)
        
        # Execute mining with optimization
        if optimization_strategy == "parallel_mining":
            result = self._parallel_mining(data, target_difficulty)
        elif optimization_strategy == "batch_mining":
            result = self._batch_mining(data, target_difficulty)
        elif optimization_strategy == "adaptive_mining":
            result = self._adaptive_mining(data, target_difficulty)
        else:
            result = self._standard_mining(data, target_difficulty)
        
        processing_time = time.time() - start_time
        
        # Record performance
        self._record_mining_performance(
            "mining", processing_time, len(str(data)), target_difficulty, result["success"]
        )
        
        logger.info(f"⛏️ Mining completed in {processing_time:.2f}s with {optimization_strategy}")
        
        return {
            **result,
            "processing_time": processing_time,
            "optimization_strategy": optimization_strategy,
            "difficulty": target_difficulty
        }
    
    def optimize_verification_performance(
        self,
        block_data: Dict[str, Any],
        chain_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize verification performance"""
        
        start_time = time.time()
        
        # Choose verification strategy
        verification_strategy = self._select_verification_strategy(len(chain_data))
        
        # Execute verification with optimization
        if verification_strategy == "parallel_verification":
            result = self._parallel_verification(block_data, chain_data)
        elif verification_strategy == "batch_verification":
            result = self._batch_verification(block_data, chain_data)
        elif verification_strategy == "incremental_verification":
            result = self._incremental_verification(block_data, chain_data)
        else:
            result = self._standard_verification(block_data, chain_data)
        
        processing_time = time.time() - start_time
        
        # Record performance
        self._record_mining_performance(
            "verification", processing_time, len(str(block_data)), 0, result["success"]
        )
        
        logger.info(f"🔍 Verification completed in {processing_time:.2f}s with {verification_strategy}")
        
        return {
            **result,
            "processing_time": processing_time,
            "verification_strategy": verification_strategy
        }
    
    def _calculate_optimal_difficulty(self) -> int:
        """Calculate optimal difficulty based on performance history"""
        
        if not self.performance_metrics:
            return 4  # Default difficulty
        
        # Get recent mining performance
        recent_mining = [
            m for m in self.performance_metrics
            if m.operation_type == "mining" and time.time() - m.timestamp < 3600  # Last hour
        ]
        
        if not recent_mining:
            return 4
        
        # Calculate average mining time
        avg_mining_time = np.mean([m.processing_time for m in recent_mining])
        
        # Adjust difficulty based on mining time
        if avg_mining_time > 10.0:  # More than 10 seconds
            return max(2, 4 - 1)  # Reduce difficulty
        elif avg_mining_time < 2.0:  # Less than 2 seconds
            return min(8, 4 + 1)  # Increase difficulty
        else:
            return 4  # Keep current difficulty
    
    def _select_mining_strategy(self, difficulty: int) -> str:
        """Select optimal mining strategy based on difficulty and performance"""
        
        if difficulty <= 3:
            return "standard_mining"
        elif difficulty <= 5:
            return "parallel_mining"
        elif difficulty <= 7:
            return "batch_mining"
        else:
            return "adaptive_mining"
    
    def _select_verification_strategy(self, chain_length: int) -> str:
        """Select optimal verification strategy based on chain length"""
        
        if chain_length <= 10:
            return "standard_verification"
        elif chain_length <= 100:
            return "parallel_verification"
        elif chain_length <= 1000:
            return "batch_verification"
        else:
            return "incremental_verification"
    
    def _parallel_mining(self, data: Dict[str, Any], difficulty: int) -> Dict[str, Any]:
        """Perform parallel mining using multiple threads"""
        
        # Create mining tasks for parallel execution
        mining_tasks = []
        
        for pool_id, pool in self.mining_pools.items():
            if pool.is_active:
                task = self.mining_executor.submit(
                    self._mine_block_worker, data, difficulty, pool_id
                )
                mining_tasks.append(task)
        
        # Wait for first successful result
        for future in as_completed(mining_tasks):
            try:
                result = future.result()
                if result["success"]:
                    # Cancel other tasks
                    for task in mining_tasks:
                        task.cancel()
                    return result
            except Exception as e:
                logger.warning(f"Mining task failed: {e}")
        
        return {"success": False, "error": "All mining tasks failed"}
    
    def _mine_block_worker(self, data: Dict[str, Any], difficulty: int, pool_id: str) -> Dict[str, Any]:
        """Worker function for parallel mining"""
        
        start_time = time.time()
        nonce = 0
        target = "0" * difficulty
        
        # Get pool for statistics
        pool = self.mining_pools.get(pool_id)
        if pool:
            pool.total_hashes += 1
        
        while True:
            # Create block hash
            block_string = f"{json.dumps(data, sort_keys=True)}{nonce}"
            block_hash = hashlib.sha256(block_string.encode()).hexdigest()
            
            if block_hash.startswith(target):
                processing_time = time.time() - start_time
                
                # Update pool statistics
                if pool:
                    pool.successful_blocks += 1
                    pool.avg_mining_time = (pool.avg_mining_time + processing_time) / 2
                
                return {
                    "success": True,
                    "hash": block_hash,
                    "nonce": nonce,
                    "processing_time": processing_time,
                    "pool_id": pool_id
                }
            
            nonce += 1
            
            # Update pool hash count
            if pool:
                pool.total_hashes += 1
            
            # Timeout after 30 seconds
            if time.time() - start_time > 30:
                return {"success": False, "error": "Mining timeout"}
    
    def _batch_mining(self, data: Dict[str, Any], difficulty: int) -> Dict[str, Any]:
        """Perform batch mining with optimized nonce ranges"""
        
        start_time = time.time()
        target = "0" * difficulty
        
        # Use larger nonce increments for batch processing
        batch_size = 1000
        nonce = 0
        
        while True:
            # Process batch of nonces
            for i in range(batch_size):
                block_string = f"{json.dumps(data, sort_keys=True)}{nonce + i}"
                block_hash = hashlib.sha256(block_string.encode()).hexdigest()
                
                if block_hash.startswith(target):
                    processing_time = time.time() - start_time
                    return {
                        "success": True,
                        "hash": block_hash,
                        "nonce": nonce + i,
                        "processing_time": processing_time,
                        "batch_size": batch_size
                    }
            
            nonce += batch_size
            
            # Timeout after 30 seconds
            if time.time() - start_time > 30:
                return {"success": False, "error": "Batch mining timeout"}
    
    def _adaptive_mining(self, data: Dict[str, Any], difficulty: int) -> Dict[str, Any]:
        """Perform adaptive mining with dynamic difficulty adjustment"""
        
        start_time = time.time()
        current_difficulty = difficulty
        target = "0" * current_difficulty
        nonce = 0
        
        # Adaptive parameters
        max_difficulty = difficulty + 2
        min_difficulty = max(1, difficulty - 2)
        
        while True:
            block_string = f"{json.dumps(data, sort_keys=True)}{nonce}"
            block_hash = hashlib.sha256(block_string.encode()).hexdigest()
            
            if block_hash.startswith(target):
                processing_time = time.time() - start_time
                return {
                    "success": True,
                    "hash": block_hash,
                    "nonce": nonce,
                    "processing_time": processing_time,
                    "final_difficulty": current_difficulty
                }
            
            nonce += 1
            
            # Adaptive difficulty adjustment
            if nonce % 10000 == 0:  # Every 10k nonces
                elapsed_time = time.time() - start_time
                if elapsed_time > 10.0 and current_difficulty > min_difficulty:
                    current_difficulty -= 1
                    target = "0" * current_difficulty
                elif elapsed_time < 2.0 and current_difficulty < max_difficulty:
                    current_difficulty += 1
                    target = "0" * current_difficulty
            
            # Timeout after 30 seconds
            if time.time() - start_time > 30:
                return {"success": False, "error": "Adaptive mining timeout"}
    
    def _standard_mining(self, data: Dict[str, Any], difficulty: int) -> Dict[str, Any]:
        """Perform standard mining"""
        
        start_time = time.time()
        nonce = 0
        target = "0" * difficulty
        
        while True:
            block_string = f"{json.dumps(data, sort_keys=True)}{nonce}"
            block_hash = hashlib.sha256(block_string.encode()).hexdigest()
            
            if block_hash.startswith(target):
                processing_time = time.time() - start_time
                return {
                    "success": True,
                    "hash": block_hash,
                    "nonce": nonce,
                    "processing_time": processing_time
                }
            
            nonce += 1
            
            # Timeout after 30 seconds
            if time.time() - start_time > 30:
                return {"success": False, "error": "Standard mining timeout"}
    
    def _parallel_verification(self, block_data: Dict[str, Any], chain_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform parallel verification of blockchain"""
        
        start_time = time.time()
        
        # Split chain into chunks for parallel verification
        chunk_size = max(1, len(chain_data) // 4)  # 4 parallel workers
        chunks = [chain_data[i:i + chunk_size] for i in range(0, len(chain_data), chunk_size)]
        
        # Create verification tasks
        verification_tasks = []
        for i, chunk in enumerate(chunks):
            task = self.verification_executor.submit(
                self._verify_chain_chunk, chunk, i
            )
            verification_tasks.append(task)
        
        # Collect results
        verification_results = []
        for future in as_completed(verification_tasks):
            try:
                result = future.result()
                verification_results.append(result)
            except Exception as e:
                logger.warning(f"Verification task failed: {e}")
                verification_results.append({"success": False, "error": str(e)})
        
        # Check if all verifications succeeded
        all_success = all(result["success"] for result in verification_results)
        
        processing_time = time.time() - start_time
        
        return {
            "success": all_success,
            "processing_time": processing_time,
            "chunks_verified": len(verification_results),
            "verification_results": verification_results
        }
    
    def _verify_chain_chunk(self, chain_chunk: List[Dict[str, Any]], chunk_id: int) -> Dict[str, Any]:
        """Verify a chunk of the blockchain"""
        
        try:
            for i in range(1, len(chain_chunk)):
                current_block = chain_chunk[i]
                previous_block = chain_chunk[i - 1]
                
                # Verify previous hash
                if current_block.get("previous_hash") != previous_block.get("hash"):
                    return {
                        "success": False,
                        "error": f"Invalid previous hash at block {i} in chunk {chunk_id}",
                        "chunk_id": chunk_id
                    }
                
                # Verify block hash
                expected_hash = self._calculate_block_hash(current_block)
                if current_block.get("hash") != expected_hash:
                    return {
                        "success": False,
                        "error": f"Invalid block hash at block {i} in chunk {chunk_id}",
                        "chunk_id": chunk_id
                    }
            
            return {
                "success": True,
                "chunk_id": chunk_id,
                "blocks_verified": len(chain_chunk)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Verification error in chunk {chunk_id}: {str(e)}",
                "chunk_id": chunk_id
            }
    
    def _calculate_block_hash(self, block_data: Dict[str, Any]) -> str:
        """Calculate hash for a block"""
        
        # Create hash string (excluding the hash field itself)
        hash_data = {k: v for k, v in block_data.items() if k != "hash"}
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def _batch_verification(self, block_data: Dict[str, Any], chain_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform batch verification of blockchain"""
        
        start_time = time.time()
        
        # Verify blocks in batches
        batch_size = 50
        total_verified = 0
        
        for i in range(0, len(chain_data), batch_size):
            batch = chain_data[i:i + batch_size]
            
            # Verify batch
            batch_result = self._verify_chain_chunk(batch, i // batch_size)
            if not batch_result["success"]:
                return {
                    "success": False,
                    "error": batch_result["error"],
                    "processing_time": time.time() - start_time
                }
            
            total_verified += len(batch)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "processing_time": processing_time,
            "total_verified": total_verified,
            "batch_size": batch_size
        }
    
    def _incremental_verification(self, block_data: Dict[str, Any], chain_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform incremental verification of blockchain"""
        
        start_time = time.time()
        
        # Only verify recent blocks (last 100)
        recent_blocks = chain_data[-100:] if len(chain_data) > 100 else chain_data
        
        # Verify recent blocks
        verification_result = self._verify_chain_chunk(recent_blocks, 0)
        
        processing_time = time.time() - start_time
        
        return {
            "success": verification_result["success"],
            "processing_time": processing_time,
            "blocks_verified": len(recent_blocks),
            "verification_type": "incremental"
        }
    
    def _standard_verification(self, block_data: Dict[str, Any], chain_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform standard verification of blockchain"""
        
        start_time = time.time()
        
        # Verify entire chain sequentially
        verification_result = self._verify_chain_chunk(chain_data, 0)
        
        processing_time = time.time() - start_time
        
        return {
            "success": verification_result["success"],
            "processing_time": processing_time,
            "blocks_verified": len(chain_data),
            "verification_type": "standard"
        }
    
    def _record_mining_performance(
        self,
        operation_type: str,
        processing_time: float,
        block_size: int,
        difficulty: int,
        success: bool
    ) -> None:
        """Record mining performance metrics"""
        
        metrics = BlockPerformanceMetrics(
            operation_type=operation_type,
            processing_time=processing_time,
            block_size=block_size,
            difficulty=difficulty,
            success=success,
            resource_usage={"cpu": 0.8, "memory": 0.6}  # Simulated resource usage
        )
        
        self.performance_metrics.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
        
        # Update performance statistics
        self.performance_stats["processing_time"].append(processing_time)
        self.performance_stats["success_rate"].append(1.0 if success else 0.0)
        
        # Analyze for optimization opportunities
        self._analyze_performance_patterns()
    
    def _analyze_performance_patterns(self) -> None:
        """Analyze performance patterns to identify optimization opportunities"""
        
        if len(self.performance_metrics) < 10:
            return
        
        # Analyze by operation type
        operation_types = set(m.operation_type for m in self.performance_metrics)
        
        for operation_type in operation_types:
            type_metrics = [m for m in self.performance_metrics if m.operation_type == operation_type]
            
            if len(type_metrics) < 5:
                continue
            
            # Calculate performance statistics
            avg_processing_time = np.mean([m.processing_time for m in type_metrics])
            success_rate = np.mean([m.success for m in type_metrics])
            
            # Identify optimization opportunities
            self._identify_optimization_opportunities(operation_type, avg_processing_time, success_rate)
    
    def _identify_optimization_opportunities(
        self,
        operation_type: str,
        avg_processing_time: float,
        success_rate: float
    ) -> None:
        """Identify optimization opportunities based on performance"""
        
        optimization_id = f"optimization_{operation_type}_{int(time.time())}"
        
        # Slow processing optimization
        if avg_processing_time > 5.0:  # More than 5 seconds
            if optimization_id not in self.blockchain_optimizations:
                self.blockchain_optimizations[optimization_id] = BlockchainOptimization(
                    optimization_id=optimization_id,
                    optimization_type="parallel_processing",
                    expected_improvement=0.4,  # 40% speed improvement
                    implementation_effort="medium",
                    description=f"Implement parallel processing for {operation_type} operations"
                )
        
        # Low success rate optimization
        elif success_rate < 0.8:  # Less than 80% success rate
            if optimization_id not in self.blockchain_optimizations:
                self.blockchain_optimizations[optimization_id] = BlockchainOptimization(
                    optimization_id=optimization_id,
                    optimization_type="difficulty_adjustment",
                    expected_improvement=0.3,  # 30% success rate improvement
                    implementation_effort="low",
                    description=f"Adjust difficulty for {operation_type} operations"
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and optimization recommendations"""
        
        if not self.performance_metrics:
            return {"status": "no_data"}
        
        # Calculate overall performance metrics
        recent_metrics = self.performance_metrics[-100:]  # Last 100 operations
        
        avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
        success_rate = np.mean([m.success for m in recent_metrics])
        
        # Performance by operation type
        performance_by_type = {}
        for operation_type in set(m.operation_type for m in recent_metrics):
            type_metrics = [m for m in recent_metrics if m.operation_type == operation_type]
            performance_by_type[operation_type] = {
                "avg_processing_time": np.mean([m.processing_time for m in type_metrics]),
                "success_rate": np.mean([m.success for m in type_metrics]),
                "sample_count": len(type_metrics)
            }
        
        # Mining pool statistics
        pool_stats = {}
        for pool_id, pool in self.mining_pools.items():
            pool_stats[pool_id] = {
                "workers": pool.workers,
                "total_hashes": pool.total_hashes,
                "successful_blocks": pool.successful_blocks,
                "avg_mining_time": pool.avg_mining_time,
                "is_active": pool.is_active
            }
        
        return {
            "overall_performance": {
                "avg_processing_time": avg_processing_time,
                "success_rate": success_rate,
                "total_operations": len(self.performance_metrics)
            },
            "performance_by_type": performance_by_type,
            "mining_pools": pool_stats,
            "optimization_strategies": len(self.blockchain_optimizations),
            "recommendations": self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        if not self.performance_metrics:
            return ["Collect more performance data for analysis"]
        
        recent_metrics = self.performance_metrics[-50:]  # Last 50 operations
        
        avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
        success_rate = np.mean([m.success for m in recent_metrics])
        
        if avg_processing_time > 5.0:
            recommendations.append("High processing times detected - consider implementing parallel processing")
        
        if success_rate < 0.8:
            recommendations.append("Low success rate detected - consider adjusting difficulty or improving algorithms")
        
        if len(self.blockchain_optimizations) > 0:
            recommendations.append(f"Apply {len(self.blockchain_optimizations)} available optimization strategies")
        
        # Mining pool recommendations
        active_pools = [p for p in self.mining_pools.values() if p.is_active]
        if len(active_pools) < 2:
            recommendations.append("Consider adding more mining pools for better parallel processing")
        
        return recommendations
    
    def _initialize_default_optimizations(self) -> None:
        """Initialize default blockchain optimizations"""
        
        if not self.blockchain_optimizations:
            # Create default optimizations
            default_optimizations = [
                BlockchainOptimization(
                    optimization_id="default_parallel_mining",
                    optimization_type="parallel_processing",
                    expected_improvement=0.3,
                    implementation_effort="medium",
                    description="Implement parallel mining for improved performance"
                ),
                BlockchainOptimization(
                    optimization_id="default_difficulty_adjustment",
                    optimization_type="difficulty_adjustment",
                    expected_improvement=0.2,
                    implementation_effort="low",
                    description="Implement dynamic difficulty adjustment"
                ),
                BlockchainOptimization(
                    optimization_id="default_batch_processing",
                    optimization_type="batch_processing",
                    expected_improvement=0.25,
                    implementation_effort="medium",
                    description="Implement batch processing for verification"
                )
            ]
            
            for optimization in default_optimizations:
                self.blockchain_optimizations[optimization.optimization_id] = optimization
    
    def _initialize_mining_pools(self) -> None:
        """Initialize mining pools for parallel processing"""
        
        if not self.mining_pools:
            # Create default mining pools
            default_pools = [
                MiningPool(
                    pool_id="pool_1",
                    workers=2,
                    current_difficulty=4,
                    total_hashes=0,
                    successful_blocks=0,
                    avg_mining_time=0.0
                ),
                MiningPool(
                    pool_id="pool_2",
                    workers=2,
                    current_difficulty=4,
                    total_hashes=0,
                    successful_blocks=0,
                    avg_mining_time=0.0
                )
            ]
            
            for pool in default_pools:
                self.mining_pools[pool.pool_id] = pool
    
    def _load_data(self) -> None:
        """Load data from storage files"""
        try:
            # Load performance metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.performance_metrics = [
                        BlockPerformanceMetrics(**metric_data)
                        for metric_data in data
                    ]
            
            # Load blockchain optimizations
            if self.optimizations_file.exists():
                with open(self.optimizations_file, 'r') as f:
                    data = json.load(f)
                    self.blockchain_optimizations = {
                        opt_id: BlockchainOptimization(**opt_data)
                        for opt_id, opt_data in data.items()
                    }
            
            # Load mining pools
            if self.pools_file.exists():
                with open(self.pools_file, 'r') as f:
                    data = json.load(f)
                    self.mining_pools = {
                        pool_id: MiningPool(**pool_data)
                        for pool_id, pool_data in data.items()
                    }
            
            logger.info(f"📁 Loaded blockchain performance data: {len(self.performance_metrics)} metrics, {len(self.mining_pools)} pools")
            
        except Exception as e:
            logger.warning(f"Failed to load blockchain performance data: {e}")
    
    def _save_data(self) -> None:
        """Save data to storage files"""
        try:
            # Save performance metrics
            metrics_data = [
                {
                    "operation_type": metric.operation_type,
                    "processing_time": metric.processing_time,
                    "block_size": metric.block_size,
                    "difficulty": metric.difficulty,
                    "success": metric.success,
                    "resource_usage": metric.resource_usage,
                    "timestamp": metric.timestamp
                }
                for metric in self.performance_metrics
            ]
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Save blockchain optimizations
            optimizations_data = {
                opt_id: {
                    "optimization_id": opt.optimization_id,
                    "optimization_type": opt.optimization_type,
                    "expected_improvement": opt.expected_improvement,
                    "implementation_effort": opt.implementation_effort,
                    "description": opt.description,
                    "parameters": opt.parameters
                }
                for opt_id, opt in self.blockchain_optimizations.items()
            }
            
            with open(self.optimizations_file, 'w') as f:
                json.dump(optimizations_data, f, indent=2)
            
            # Save mining pools
            pools_data = {
                pool_id: {
                    "pool_id": pool.pool_id,
                    "workers": pool.workers,
                    "current_difficulty": pool.current_difficulty,
                    "total_hashes": pool.total_hashes,
                    "successful_blocks": pool.successful_blocks,
                    "avg_mining_time": pool.avg_mining_time,
                    "is_active": pool.is_active
                }
                for pool_id, pool in self.mining_pools.items()
            }
            
            with open(self.pools_file, 'w') as f:
                json.dump(pools_data, f, indent=2)
            
            logger.debug("💾 Saved blockchain performance data to storage")
            
        except Exception as e:
            logger.error(f"Failed to save blockchain performance data: {e}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'mining_executor'):
            self.mining_executor.shutdown(wait=True)
        if hasattr(self, 'verification_executor'):
            self.verification_executor.shutdown(wait=True)


# Helper functions for integration
def create_blockchain_performance_optimizer(cfg: IceburgConfig) -> BlockchainPerformanceOptimizer:
    """Create blockchain performance optimizer instance"""
    return BlockchainPerformanceOptimizer(cfg)

def optimize_mining_performance(
    optimizer: BlockchainPerformanceOptimizer,
    data: Dict[str, Any],
    target_difficulty: int = None
) -> Dict[str, Any]:
    """Optimize mining performance"""
    return optimizer.optimize_mining_performance(data, target_difficulty)

def optimize_verification_performance(
    optimizer: BlockchainPerformanceOptimizer,
    block_data: Dict[str, Any],
    chain_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Optimize verification performance"""
    return optimizer.optimize_verification_performance(block_data, chain_data)
