"""
Unified Sensor Processing Pipeline for ICEBURG
Implements Tesla-style unified sensor processing for multiple data streams.
"""

import os
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Sensor data structure."""
    sensor_id: str
    data_type: str
    timestamp: float
    data: Any
    quality: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Processing result structure."""
    result_id: str
    input_sensors: List[str]
    output_data: Any
    confidence: float
    processing_time: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedSensorProcessor:
    """
    Unified sensor processing pipeline for ICEBURG.
    
    Features:
    - Multi-modal sensor fusion
    - Real-time processing
    - Quality assessment
    - Confidence scoring
    - Adaptive processing
    - Tesla-style end-to-end learning
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize unified sensor processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sensors = {}
        self.processing_pipeline = []
        self.data_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processing_threads = []
        self.is_processing = False
        
        # Processing configuration
        self.max_threads = self.config.get("max_threads", 4)
        self.processing_timeout = self.config.get("processing_timeout", 5.0)
        self.quality_threshold = self.config.get("quality_threshold", 0.7)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        
        # Initialize processing pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize processing pipeline."""
        logger.info("Initializing unified sensor processing pipeline")
        
        # Add default processing stages
        self.add_processing_stage("data_validation", self._validate_sensor_data)
        self.add_processing_stage("quality_assessment", self._assess_data_quality)
        self.add_processing_stage("sensor_fusion", self._fuse_sensor_data)
        self.add_processing_stage("confidence_scoring", self._score_confidence)
        self.add_processing_stage("result_generation", self._generate_result)
    
    def add_sensor(self, sensor_id: str, sensor_type: str, config: Dict[str, Any] = None):
        """Add a sensor to the processing pipeline."""
        sensor_config = {
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "config": config or {},
            "active": True,
            "data_count": 0,
            "last_update": 0.0
        }
        
        self.sensors[sensor_id] = sensor_config
        logger.info(f"Added sensor: {sensor_id} ({sensor_type})")
    
    def remove_sensor(self, sensor_id: str):
        """Remove a sensor from the processing pipeline."""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            logger.info(f"Removed sensor: {sensor_id}")
    
    def add_processing_stage(self, stage_name: str, processing_function: callable):
        """Add a processing stage to the pipeline."""
        stage_config = {
            "name": stage_name,
            "function": processing_function,
            "enabled": True,
            "timeout": self.processing_timeout
        }
        
        self.processing_pipeline.append(stage_config)
        logger.info(f"Added processing stage: {stage_name}")
    
    def remove_processing_stage(self, stage_name: str):
        """Remove a processing stage from the pipeline."""
        self.processing_pipeline = [
            stage for stage in self.processing_pipeline 
            if stage["name"] != stage_name
        ]
        logger.info(f"Removed processing stage: {stage_name}")
    
    async def process_sensor_data(self, sensor_data: SensorData) -> ProcessingResult:
        """
        Process sensor data through the unified pipeline.
        
        Args:
            sensor_data: Sensor data to process
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        # Validate sensor
        if sensor_data.sensor_id not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_data.sensor_id}")
        
        # Update sensor statistics
        self.sensors[sensor_data.sensor_id]["data_count"] += 1
        self.sensors[sensor_data.sensor_id]["last_update"] = time.time()
        
        # Process through pipeline
        current_data = sensor_data
        processing_metadata = {}
        
        for stage in self.processing_pipeline:
            if not stage["enabled"]:
                continue
            
            try:
                # Process stage
                stage_start = time.time()
                current_data = await self._process_stage(stage, current_data)
                stage_time = time.time() - stage_start
                
                # Store processing metadata
                processing_metadata[stage["name"]] = {
                    "processing_time": stage_time,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error in processing stage {stage['name']}: {e}")
                processing_metadata[stage["name"]] = {
                    "processing_time": 0.0,
                    "success": False,
                    "error": str(e)
                }
        
        # Generate result
        total_processing_time = time.time() - start_time
        result = ProcessingResult(
            result_id=f"result_{int(time.time() * 1000)}",
            input_sensors=[sensor_data.sensor_id],
            output_data=current_data,
            confidence=current_data.confidence,
            processing_time=total_processing_time,
            timestamp=time.time(),
            metadata=processing_metadata
        )
        
        return result
    
    async def _process_stage(self, stage: Dict[str, Any], data: SensorData) -> SensorData:
        """Process a single stage of the pipeline."""
        try:
            # Run processing function
            result = await asyncio.wait_for(
                stage["function"](data),
                timeout=stage["timeout"]
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Processing stage {stage['name']} timed out")
            raise
        except Exception as e:
            logger.error(f"Error in processing stage {stage['name']}: {e}")
            raise
    
    async def _validate_sensor_data(self, data: SensorData) -> SensorData:
        """Validate sensor data."""
        # Check data quality
        if data.quality < self.quality_threshold:
            logger.warning(f"Low quality data from sensor {data.sensor_id}: {data.quality}")
        
        # Check data type
        if data.data_type not in ["text", "image", "audio", "video", "sensor", "metadata"]:
            logger.warning(f"Unknown data type: {data.data_type}")
        
        # Check timestamp
        if data.timestamp <= 0:
            logger.warning(f"Invalid timestamp: {data.timestamp}")
            data.timestamp = time.time()
        
        return data
    
    async def _assess_data_quality(self, data: SensorData) -> SensorData:
        """Assess data quality."""
        quality_score = data.quality
        
        # Assess based on data type
        if data.data_type == "text":
            # Text quality assessment
            if isinstance(data.data, str):
                quality_score = min(len(data.data) / 100.0, 1.0)
            else:
                quality_score = 0.0
        
        elif data.data_type == "image":
            # Image quality assessment (simplified)
            if isinstance(data.data, dict) and "resolution" in data.data:
                resolution = data.data["resolution"]
                quality_score = min(resolution / 1000000.0, 1.0)  # 1MP = 1.0
            else:
                quality_score = 0.5
        
        elif data.data_type == "audio":
            # Audio quality assessment (simplified)
            if isinstance(data.data, dict) and "sample_rate" in data.data:
                sample_rate = data.data["sample_rate"]
                quality_score = min(sample_rate / 44100.0, 1.0)  # 44.1kHz = 1.0
            else:
                quality_score = 0.5
        
        elif data.data_type == "sensor":
            # Sensor data quality assessment
            if isinstance(data.data, (int, float)):
                quality_score = 0.8  # Assume good quality for numeric data
            else:
                quality_score = 0.5
        
        # Update quality score
        data.quality = quality_score
        
        return data
    
    async def _fuse_sensor_data(self, data: SensorData) -> SensorData:
        """Fuse sensor data with other sensors."""
        # Get related sensors
        related_sensors = self._get_related_sensors(data.sensor_id)
        
        if not related_sensors:
            return data
        
        # Fuse data from related sensors
        fused_data = data.data
        
        for sensor_id in related_sensors:
            if sensor_id in self.sensors:
                # Get recent data from related sensor
                related_data = self._get_recent_sensor_data(sensor_id)
                if related_data:
                    # Fuse data (simplified)
                    fused_data = self._fuse_data(fused_data, related_data)
        
        # Update data
        data.data = fused_data
        
        return data
    
    def _get_related_sensors(self, sensor_id: str) -> List[str]:
        """Get related sensors for fusion."""
        # Simple implementation: return all other sensors
        return [sid for sid in self.sensors.keys() if sid != sensor_id]
    
    def _get_recent_sensor_data(self, sensor_id: str) -> Optional[Any]:
        """Get recent data from a sensor."""
        # This would typically get data from a sensor buffer
        # For now, return None
        return None
    
    def _fuse_data(self, data1: Any, data2: Any) -> Any:
        """Fuse two data streams."""
        # Simple fusion implementation
        if isinstance(data1, (int, float)) and isinstance(data2, (int, float)):
            return (data1 + data2) / 2.0
        elif isinstance(data1, str) and isinstance(data2, str):
            return f"{data1} + {data2}"
        else:
            return data1  # Return first data if types don't match
    
    async def _score_confidence(self, data: SensorData) -> SensorData:
        """Score confidence in the data."""
        confidence_score = data.confidence
        
        # Adjust confidence based on quality
        confidence_score *= data.quality
        
        # Adjust confidence based on data type
        if data.data_type == "text":
            # Text confidence based on length and content
            if isinstance(data.data, str):
                confidence_score *= min(len(data.data) / 50.0, 1.0)
        
        elif data.data_type == "image":
            # Image confidence based on resolution
            if isinstance(data.data, dict) and "resolution" in data.data:
                resolution = data.data["resolution"]
                confidence_score *= min(resolution / 1000000.0, 1.0)
        
        elif data.data_type == "sensor":
            # Sensor confidence based on data validity
            if isinstance(data.data, (int, float)):
                confidence_score *= 0.9  # High confidence for numeric data
            else:
                confidence_score *= 0.5  # Lower confidence for non-numeric data
        
        # Update confidence score
        data.confidence = min(confidence_score, 1.0)
        
        return data
    
    async def _generate_result(self, data: SensorData) -> SensorData:
        """Generate final result."""
        # Add processing metadata
        data.metadata["processing_timestamp"] = time.time()
        data.metadata["pipeline_stages"] = len(self.processing_pipeline)
        data.metadata["quality_threshold"] = self.quality_threshold
        data.metadata["confidence_threshold"] = self.confidence_threshold
        
        return data
    
    async def process_multiple_sensors(self, sensor_data_list: List[SensorData]) -> List[ProcessingResult]:
        """Process multiple sensor data streams."""
        results = []
        
        # Process each sensor data
        for sensor_data in sensor_data_list:
            try:
                result = await self.process_sensor_data(sensor_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing sensor data: {e}")
                # Create error result
                error_result = ProcessingResult(
                    result_id=f"error_{int(time.time() * 1000)}",
                    input_sensors=[sensor_data.sensor_id],
                    output_data=None,
                    confidence=0.0,
                    processing_time=0.0,
                    timestamp=time.time(),
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        
        return results
    
    async def start_continuous_processing(self):
        """Start continuous processing of sensor data."""
        self.is_processing = True
        
        # Start processing threads
        for i in range(self.max_threads):
            thread = threading.Thread(
                target=self._processing_worker,
                name=f"ProcessingThread-{i}"
            )
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"Started {self.max_threads} processing threads")
    
    def _processing_worker(self):
        """Processing worker thread."""
        while self.is_processing:
            try:
                # Get data from queue
                sensor_data = self.data_queue.get(timeout=1.0)
                
                # Process data
                result = asyncio.run(self.process_sensor_data(sensor_data))
                
                # Put result in result queue
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
    
    def stop_continuous_processing(self):
        """Stop continuous processing."""
        self.is_processing = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=5.0)
        
        self.processing_threads.clear()
        logger.info("Stopped continuous processing")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "sensors": {
                sensor_id: {
                    "type": sensor_config["sensor_type"],
                    "active": sensor_config["active"],
                    "data_count": sensor_config["data_count"],
                    "last_update": sensor_config["last_update"]
                }
                for sensor_id, sensor_config in self.sensors.items()
            },
            "processing_pipeline": [
                {
                    "name": stage["name"],
                    "enabled": stage["enabled"],
                    "timeout": stage["timeout"]
                }
                for stage in self.processing_pipeline
            ],
            "queue_sizes": {
                "data_queue": self.data_queue.qsize(),
                "result_queue": self.result_queue.qsize()
            },
            "processing_status": {
                "is_processing": self.is_processing,
                "active_threads": len(self.processing_threads)
            }
        }
    
    def get_sensor_data(self, sensor_id: str) -> Optional[Any]:
        """Get recent data from a sensor."""
        if sensor_id not in self.sensors:
            return None
        
        # This would typically get data from a sensor buffer
        # For now, return None
        return None
    
    def set_sensor_data(self, sensor_id: str, data: Any, data_type: str = "sensor", quality: float = 1.0, confidence: float = 1.0):
        """Set sensor data."""
        if sensor_id not in self.sensors:
            logger.warning(f"Unknown sensor: {sensor_id}")
            return
        
        # Create sensor data
        sensor_data = SensorData(
            sensor_id=sensor_id,
            data_type=data_type,
            timestamp=time.time(),
            data=data,
            quality=quality,
            confidence=confidence
        )
        
        # Add to processing queue
        self.data_queue.put(sensor_data)
    
    def get_processing_result(self) -> Optional[ProcessingResult]:
        """Get a processing result from the result queue."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_processing_results(self) -> List[ProcessingResult]:
        """Get all processing results from the result queue."""
        results = []
        
        while True:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results


# Convenience functions
def create_unified_processor(config: Dict[str, Any] = None) -> UnifiedSensorProcessor:
    """Create unified sensor processor."""
    return UnifiedSensorProcessor(config)


async def process_sensor_data(sensor_data: SensorData, config: Dict[str, Any] = None) -> ProcessingResult:
    """Process sensor data."""
    processor = create_unified_processor(config)
    return await processor.process_sensor_data(sensor_data)
