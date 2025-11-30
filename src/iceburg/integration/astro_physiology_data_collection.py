"""
Astro-Physiology Data Collection and Validation Framework
Collects real-world data to validate predictions
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationDataPoint:
    """Single data point for validation"""
    timestamp: datetime
    user_id: str
    prediction_type: str  # "voltage_gate", "biophysical_parameter", "tcm_organ", etc.
    predicted_value: float
    actual_value: Optional[float] = None
    measurement_method: Optional[str] = None
    confidence: float = 0.5
    notes: Optional[str] = None


@dataclass
class ValidationResult:
    """Validation result for a prediction"""
    prediction_type: str
    predicted_value: float
    actual_values: List[float]
    correlation: Optional[float] = None
    p_value: Optional[float] = None
    sample_size: int = 0
    confidence_interval: Optional[tuple] = None
    validated: bool = False


class AstroPhysiologyDataCollection:
    """
    Collects and validates real-world data against predictions.
    Tracks user outcomes over time and compares to predicted values.
    """
    
    def __init__(self, db=None):
        self.db = db
        self.validation_data: List[ValidationDataPoint] = []
    
    async def collect_user_outcome(
        self,
        user_id: str,
        prediction_type: str,
        predicted_value: float,
        actual_value: Optional[float] = None,
        measurement_method: Optional[str] = None,
        notes: Optional[str] = None
    ) -> ValidationDataPoint:
        """
        Collect a single data point comparing prediction to actual outcome.
        
        Args:
            user_id: User identifier
            prediction_type: Type of prediction (e.g., "sodium_channel_sensitivity")
            predicted_value: Value predicted by the model
            actual_value: Actual measured value (if available)
            measurement_method: How the actual value was measured
            notes: Additional notes
            
        Returns:
            ValidationDataPoint
        """
        data_point = ValidationDataPoint(
            timestamp=datetime.now(),
            user_id=user_id,
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            actual_value=actual_value,
            measurement_method=measurement_method,
            confidence=0.5,  # Default confidence
            notes=notes
        )
        
        self.validation_data.append(data_point)
        
        # Store in database if available
        if self.db:
            try:
                await self._store_data_point(data_point)
            except Exception as e:
                logger.warning(f"Could not store data point in database: {e}")
        
        logger.info(f"📊 Collected validation data: {prediction_type} = {actual_value} (predicted: {predicted_value})")
        
        return data_point
    
    async def compare_predictions_to_actuals(
        self,
        prediction_type: str,
        days: int = 30
    ) -> ValidationResult:
        """
        Compare predictions to actual outcomes for a specific prediction type.
        
        Args:
            prediction_type: Type of prediction to validate
            days: Number of days to look back
            
        Returns:
            ValidationResult with correlation, p-value, etc.
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter data points
        relevant_data = [
            dp for dp in self.validation_data
            if dp.prediction_type == prediction_type
            and dp.timestamp >= cutoff_date
            and dp.actual_value is not None
        ]
        
        if len(relevant_data) < 10:  # Need minimum sample size
            return ValidationResult(
                prediction_type=prediction_type,
                predicted_value=0.0,
                actual_values=[],
                sample_size=len(relevant_data),
                validated=False
            )
        
        predicted_values = [dp.predicted_value for dp in relevant_data]
        actual_values = [dp.actual_value for dp in relevant_data]
        
        # Calculate correlation
        correlation = self._calculate_correlation(predicted_values, actual_values)
        
        # Calculate p-value (simplified - would use proper statistical test)
        p_value = self._calculate_p_value(predicted_values, actual_values)
        
        # Calculate confidence interval
        mean_actual = sum(actual_values) / len(actual_values)
        std_actual = self._calculate_std(actual_values)
        confidence_interval = (
            mean_actual - 1.96 * std_actual,
            mean_actual + 1.96 * std_actual
        )
        
        # Validation criteria: p < 0.05 and correlation > 0.3
        validated = p_value is not None and p_value < 0.05 and abs(correlation) > 0.3
        
        result = ValidationResult(
            prediction_type=prediction_type,
            predicted_value=sum(predicted_values) / len(predicted_values),
            actual_values=actual_values,
            correlation=correlation,
            p_value=p_value,
            sample_size=len(relevant_data),
            confidence_interval=confidence_interval,
            validated=validated
        )
        
        logger.info(f"📊 Validation result for {prediction_type}: correlation={correlation:.3f}, p={p_value:.3f}, validated={validated}")
        
        return result
    
    async def track_user_health_metrics(
        self,
        user_id: str,
        metrics: Dict[str, float],
        source: str = "user_input"
    ) -> Dict[str, Any]:
        """
        Track user health metrics over time for validation.
        
        Args:
            user_id: User identifier
            metrics: Dictionary of health metrics (e.g., {"hrv": 45.2, "sleep_quality": 7.5})
            source: Source of data ("user_input", "device", "medical_record")
            
        Returns:
            Tracking confirmation
        """
        tracking_data = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "source": source
        }
        
        # Store in database if available
        if self.db:
            try:
                await self._store_health_metrics(tracking_data)
            except Exception as e:
                logger.warning(f"Could not store health metrics in database: {e}")
        
        logger.info(f"📊 Tracked health metrics for user {user_id}: {len(metrics)} metrics")
        
        return tracking_data
    
    async def generate_validation_report(
        self,
        prediction_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            prediction_types: Specific types to validate (None = all)
            
        Returns:
            Validation report with statistics and conclusions
        """
        if prediction_types is None:
            # Get all unique prediction types
            prediction_types = list(set(dp.prediction_type for dp in self.validation_data))
        
        results = {}
        for pred_type in prediction_types:
            result = await self.compare_predictions_to_actuals(pred_type, days=365)
            results[pred_type] = {
                "correlation": result.correlation,
                "p_value": result.p_value,
                "sample_size": result.sample_size,
                "validated": result.validated,
                "confidence_interval": result.confidence_interval
            }
        
        # Overall statistics
        total_samples = sum(r["sample_size"] for r in results.values())
        validated_count = sum(1 for r in results.values() if r["validated"])
        avg_correlation = sum(r["correlation"] or 0 for r in results.values()) / len(results) if results else 0
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "prediction_types_validated": len(prediction_types),
            "total_samples": total_samples,
            "validated_predictions": validated_count,
            "validation_rate": validated_count / len(results) if results else 0,
            "average_correlation": avg_correlation,
            "results_by_type": results,
            "conclusions": self._generate_conclusions(results)
        }
        
        logger.info(f"📊 Generated validation report: {validated_count}/{len(results)} predictions validated")
        
        return report
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
        
        correlation = numerator / ((denominator_x * denominator_y) ** 0.5)
        return correlation
    
    def _calculate_p_value(self, x: List[float], y: List[float]) -> Optional[float]:
        """Calculate p-value for correlation (simplified)"""
        if len(x) < 3:
            return None
        
        correlation = self._calculate_correlation(x, y)
        n = len(x)
        
        # Simplified t-test for correlation
        # t = r * sqrt((n-2) / (1-r^2))
        if abs(correlation) >= 1.0:
            return 1.0
        
        t_stat = correlation * ((n - 2) / (1 - correlation ** 2)) ** 0.5
        
        # Simplified p-value approximation (would use proper t-distribution)
        # For large n, t > 2 roughly corresponds to p < 0.05
        if abs(t_stat) > 2.0:
            return 0.01  # Significant
        elif abs(t_stat) > 1.5:
            return 0.05  # Borderline
        else:
            return 0.10  # Not significant
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _generate_conclusions(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate conclusions from validation results"""
        conclusions = []
        
        validated = [k for k, v in results.items() if v.get("validated")]
        not_validated = [k for k, v in results.items() if not v.get("validated")]
        
        if validated:
            conclusions.append(f"Validated predictions: {', '.join(validated)}")
            conclusions.append("These predictions show statistically significant correlations with actual outcomes.")
        
        if not_validated:
            conclusions.append(f"Unvalidated predictions: {', '.join(not_validated)}")
            conclusions.append("These predictions do not yet show significant correlations - may need larger sample size or better measurements.")
        
        avg_corr = sum(r.get("correlation", 0) or 0 for r in results.values()) / len(results) if results else 0
        if abs(avg_corr) > 0.3:
            conclusions.append(f"Overall correlation: {avg_corr:.3f} - suggests model may have predictive value")
        else:
            conclusions.append(f"Overall correlation: {avg_corr:.3f} - weak correlation, model needs improvement")
        
        return conclusions
    
    async def _store_data_point(self, data_point: ValidationDataPoint):
        """Store data point in database"""
        if self.db and hasattr(self.db, 'store_validation_data'):
            await self.db.store_validation_data(
                user_id=data_point.user_id,
                prediction_type=data_point.prediction_type,
                predicted_value=data_point.predicted_value,
                actual_value=data_point.actual_value,
                measurement_method=data_point.measurement_method,
                timestamp=data_point.timestamp.isoformat(),
                notes=data_point.notes
            )
    
    async def _store_health_metrics(self, tracking_data: Dict[str, Any]):
        """Store health metrics in database"""
        if self.db and hasattr(self.db, 'store_health_tracking'):
            await self.db.store_health_tracking(
                user_id=tracking_data["user_id"],
                metrics=tracking_data["metrics"],
                source=tracking_data["source"],
                timestamp=tracking_data["timestamp"]
            )

