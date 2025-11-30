"""
Enhanced Safety Mechanisms and Validation System for ICEBURG Autonomous System
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import re


class SafetyLevel(Enum):
    """Safety validation levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyCategory(Enum):
    """Safety validation categories."""
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATA_INTEGRITY = "data_integrity"
    RESOURCE_USAGE = "resource_usage"
    SYSTEM_STABILITY = "system_stability"
    BUSINESS_LOGIC = "business_logic"
    COMPLIANCE = "compliance"


@dataclass
class SafetyConstraint:
    """Safety constraint definition."""
    name: str
    category: SafetyCategory
    level: SafetyLevel
    condition: Callable[[Dict[str, Any]], bool]
    message: str
    max_violations: int = 3
    cooldown: float = 300.0  # 5 minutes
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyViolation:
    """Safety violation record."""
    constraint_name: str
    category: SafetyCategory
    level: SafetyLevel
    message: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    violation_count: int = 1


@dataclass
class SafetyCheck:
    """Safety check result."""
    passed: bool
    violations: List[SafetyViolation]
    warnings: List[str]
    recommendations: List[str]
    safety_score: float  # 0-100
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyValidator:
    """Enhanced safety validation system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.constraints: List[SafetyConstraint] = []
        self.violations: List[SafetyViolation] = []
        self.violation_counts: Dict[str, int] = {}
        self.last_violation_times: Dict[str, float] = {}
        self.safety_handlers: List[Callable[[SafetyViolation], None]] = []
        self.validation_active = False
        self.validation_tasks: List[asyncio.Task] = []
        
        # Initialize default safety constraints
        self._init_default_constraints()
    
    def _init_default_constraints(self):
        """Initialize default safety constraints."""
        # Performance constraints
        self.add_constraint(SafetyConstraint(
            name="max_response_time",
            category=SafetyCategory.PERFORMANCE,
            level=SafetyLevel.MEDIUM,
            condition=lambda ctx: ctx.get("response_time", 0) <= 30.0,
            message="Response time exceeds 30 seconds",
            max_violations=5
        ))
        
        self.add_constraint(SafetyConstraint(
            name="max_memory_usage",
            category=SafetyCategory.RESOURCE_USAGE,
            level=SafetyLevel.HIGH,
            condition=lambda ctx: ctx.get("memory_usage_mb", 0) <= 2048.0,
            message="Memory usage exceeds 2GB",
            max_violations=3
        ))
        
        self.add_constraint(SafetyConstraint(
            name="max_cpu_usage",
            category=SafetyCategory.RESOURCE_USAGE,
            level=SafetyLevel.MEDIUM,
            condition=lambda ctx: ctx.get("cpu_usage_percent", 0) <= 90.0,
            message="CPU usage exceeds 90%",
            max_violations=5
        ))
        
        # Security constraints
        self.add_constraint(SafetyConstraint(
            name="no_sql_injection",
            category=SafetyCategory.SECURITY,
            level=SafetyLevel.CRITICAL,
            condition=lambda ctx: not self._contains_sql_injection(ctx.get("query", "")),
            message="Potential SQL injection detected",
            max_violations=1
        ))
        
        self.add_constraint(SafetyConstraint(
            name="no_path_traversal",
            category=SafetyCategory.SECURITY,
            level=SafetyLevel.CRITICAL,
            condition=lambda ctx: not self._contains_path_traversal(ctx.get("file_path", "")),
            message="Potential path traversal detected",
            max_violations=1
        ))
        
        # Data integrity constraints
        self.add_constraint(SafetyConstraint(
            name="data_validation",
            category=SafetyCategory.DATA_INTEGRITY,
            level=SafetyLevel.HIGH,
            condition=lambda ctx: self._validate_data_integrity(ctx),
            message="Data integrity validation failed",
            max_violations=3
        ))
        
        # System stability constraints
        self.add_constraint(SafetyConstraint(
            name="error_rate_limit",
            category=SafetyCategory.SYSTEM_STABILITY,
            level=SafetyLevel.HIGH,
            condition=lambda ctx: ctx.get("error_rate", 0) <= 10.0,
            message="Error rate exceeds 10%",
            max_violations=3
        ))
        
        self.add_constraint(SafetyConstraint(
            name="concurrent_operations_limit",
            category=SafetyCategory.SYSTEM_STABILITY,
            level=SafetyLevel.MEDIUM,
            condition=lambda ctx: ctx.get("concurrent_operations", 0) <= 100,
            message="Too many concurrent operations",
            max_violations=5
        ))
        
        # Business logic constraints
        self.add_constraint(SafetyConstraint(
            name="accuracy_threshold",
            category=SafetyCategory.BUSINESS_LOGIC,
            level=SafetyLevel.MEDIUM,
            condition=lambda ctx: ctx.get("accuracy", 1.0) >= 0.7,
            message="Accuracy below acceptable threshold",
            max_violations=5
        ))
    
    def _contains_sql_injection(self, query: str) -> bool:
        """Check for potential SQL injection patterns."""
        if not query:
            return False
        
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
            r"(\b(UNION|OR|AND)\b.*\b(SELECT|INSERT|UPDATE|DELETE)\b)",
            r"(\b(EXEC|EXECUTE|SP_)\b)",
            r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)",
            r"(\b(WAITFOR|DELAY)\b)",
            r"(\b(CHAR|ASCII|SUBSTRING)\b)",
            r"(\b(CAST|CONVERT)\b)",
            r"(\b(INFORMATION_SCHEMA|SYS\.|SYSOBJECTS)\b)",
            r"(\b(OPENROWSET|OPENDATASOURCE)\b)",
            r"(\b(BULK|BULKINSERT)\b)"
        ]
        
        query_lower = query.lower()
        for pattern in sql_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _contains_path_traversal(self, file_path: str) -> bool:
        """Check for potential path traversal patterns."""
        if not file_path:
            return False
        
        traversal_patterns = [
            "../", "..\\", "..%2f", "..%5c",
            "%2e%2e%2f", "%2e%2e%5c",
            "....//", "....\\\\",
            "..%252f", "..%255c"
        ]
        
        file_path_lower = file_path.lower()
        for pattern in traversal_patterns:
            if pattern in file_path_lower:
                return True
        
        return False
    
    def _validate_data_integrity(self, context: Dict[str, Any]) -> bool:
        """Validate data integrity constraints."""
        # Check for required fields
        required_fields = ["query_id", "timestamp"]
        for field in required_fields:
            if field not in context:
                return False
        
        # Validate data types
        if not isinstance(context.get("query_id"), str):
            return False
        
        if not isinstance(context.get("timestamp"), (int, float)):
            return False
        
        # Validate numeric ranges
        response_time = context.get("response_time", 0)
        if not isinstance(response_time, (int, float)) or response_time < 0:
            return False
        
        accuracy = context.get("accuracy", 1.0)
        if not isinstance(accuracy, (int, float)) or not (0 <= accuracy <= 1):
            return False
        
        return True
    
    def add_constraint(self, constraint: SafetyConstraint):
        """Add a safety constraint."""
        self.constraints.append(constraint)
        self.logger.info(f"Added safety constraint: {constraint.name}")
    
    def remove_constraint(self, constraint_name: str):
        """Remove a safety constraint by name."""
        self.constraints = [c for c in self.constraints if c.name != constraint_name]
        self.logger.info(f"Removed safety constraint: {constraint_name}")
    
    def add_safety_handler(self, handler: Callable[[SafetyViolation], None]):
        """Add a safety violation handler."""
        self.safety_handlers.append(handler)
        self.logger.info("Added safety violation handler")
    
    async def validate(self, context: Dict[str, Any]) -> SafetyCheck:
        """Perform comprehensive safety validation."""
        violations = []
        warnings = []
        recommendations = []
        
        for constraint in self.constraints:
            if not constraint.enabled:
                continue
            
            try:
                if not constraint.condition(context):
                    violation = SafetyViolation(
                        constraint_name=constraint.name,
                        category=constraint.category,
                        level=constraint.level,
                        message=constraint.message,
                        timestamp=time.time(),
                        context=context.copy()
                    )
                    
                    # Check violation count
                    violation_key = constraint.name
                    self.violation_counts[violation_key] = self.violation_counts.get(violation_key, 0) + 1
                    violation.violation_count = self.violation_counts[violation_key]
                    
                    # Check if max violations exceeded
                    if self.violation_counts[violation_key] > constraint.max_violations:
                        violation.level = SafetyLevel.CRITICAL
                        violation.message += f" (CRITICAL: {self.violation_counts[violation_key]} violations)"
                    
                    violations.append(violation)
                    self.violations.append(violation)
                    
                    # Call safety handlers
                    for handler in self.safety_handlers:
                        try:
                            handler(violation)
                        except Exception as e:
                            self.logger.error(f"Error in safety handler: {e}")
                    
                    # Generate recommendations
                    recommendations.extend(self._generate_recommendations(violation))
                    
            except Exception as e:
                self.logger.error(f"Error validating constraint {constraint.name}: {e}")
                warnings.append(f"Constraint {constraint.name} validation failed: {e}")
        
        # Calculate safety score
        safety_score = self._calculate_safety_score(violations)
        
        return SafetyCheck(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            safety_score=safety_score,
            metadata={
                "total_constraints": len(self.constraints),
                "active_constraints": len([c for c in self.constraints if c.enabled]),
                "validation_timestamp": time.time()
            }
        )
    
    def _generate_recommendations(self, violation: SafetyViolation) -> List[str]:
        """Generate safety recommendations based on violation."""
        recommendations = []
        
        if violation.category == SafetyCategory.PERFORMANCE:
            if "response_time" in violation.constraint_name:
                recommendations.append("Consider optimizing query processing or increasing resources")
            elif "accuracy" in violation.constraint_name:
                recommendations.append("Review model parameters or training data quality")
        
        elif violation.category == SafetyCategory.RESOURCE_USAGE:
            if "memory" in violation.constraint_name:
                recommendations.append("Implement memory optimization or increase available memory")
            elif "cpu" in violation.constraint_name:
                recommendations.append("Consider load balancing or resource scaling")
        
        elif violation.category == SafetyCategory.SECURITY:
            recommendations.append("Review input validation and sanitization procedures")
            recommendations.append("Implement additional security checks")
        
        elif violation.category == SafetyCategory.SYSTEM_STABILITY:
            recommendations.append("Investigate system stability issues")
            recommendations.append("Consider implementing circuit breakers")
        
        return recommendations
    
    def _calculate_safety_score(self, violations: List[SafetyViolation]) -> float:
        """Calculate safety score based on violations."""
        if not violations:
            return 100.0
        
        score = 100.0
        
        for violation in violations:
            if violation.level == SafetyLevel.CRITICAL:
                score -= 25.0
            elif violation.level == SafetyLevel.HIGH:
                score -= 15.0
            elif violation.level == SafetyLevel.MEDIUM:
                score -= 10.0
            elif violation.level == SafetyLevel.LOW:
                score -= 5.0
        
        return max(0.0, score)
    
    def resolve_violation(self, violation_id: str, resolved_by: str = "system"):
        """Resolve a safety violation."""
        for violation in self.violations:
            if violation.constraint_name == violation_id and not violation.resolved:
                violation.resolved = True
                violation.resolved_at = time.time()
                self.logger.info(f"Safety violation {violation_id} resolved by {resolved_by}")
                break
    
    def get_active_violations(self) -> List[SafetyViolation]:
        """Get all active (unresolved) violations."""
        return [v for v in self.violations if not v.resolved]
    
    def get_violations_by_level(self, level: SafetyLevel) -> List[SafetyViolation]:
        """Get violations by severity level."""
        return [v for v in self.violations if v.level == level]
    
    def get_violations_by_category(self, category: SafetyCategory) -> List[SafetyViolation]:
        """Get violations by category."""
        return [v for v in self.violations if v.category == category]
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety statistics and health metrics."""
        total_violations = len(self.violations)
        active_violations = len(self.get_active_violations())
        
        level_counts = {}
        category_counts = {}
        
        for violation in self.violations:
            level = violation.level.value
            category = violation.category.value
            
            level_counts[level] = level_counts.get(level, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Calculate overall safety score
        recent_violations = [v for v in self.violations if time.time() - v.timestamp < 3600]  # Last hour
        safety_score = self._calculate_safety_score(recent_violations)
        
        return {
            "total_violations": total_violations,
            "active_violations": active_violations,
            "resolved_violations": total_violations - active_violations,
            "level_distribution": level_counts,
            "category_distribution": category_counts,
            "safety_score": safety_score,
            "constraints_count": len(self.constraints),
            "active_constraints": len([c for c in self.constraints if c.enabled])
        }
    
    async def start_validation(self):
        """Start continuous safety validation."""
        if self.validation_active:
            return
        
        self.validation_active = True
        
        # Start background validation tasks
        self.validation_tasks = [
            asyncio.create_task(self._cleanup_old_violations()),
            asyncio.create_task(self._monitor_safety_health())
        ]
        
        self.logger.info("Safety validation system started")
    
    async def stop_validation(self):
        """Stop safety validation."""
        if not self.validation_active:
            return
        
        self.validation_active = False
        
        # Cancel validation tasks
        for task in self.validation_tasks:
            task.cancel()
        
        self.validation_tasks.clear()
        self.logger.info("Safety validation system stopped")
    
    async def _cleanup_old_violations(self):
        """Clean up old violations to prevent memory issues."""
        while self.validation_active:
            try:
                # Remove violations older than 24 hours
                cutoff_time = time.time() - 86400
                self.violations = [v for v in self.violations if v.timestamp > cutoff_time]
                
                await asyncio.sleep(3600)  # Run every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in violation cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_safety_health(self):
        """Monitor overall safety health."""
        while self.validation_active:
            try:
                # Check for critical violations
                critical_violations = self.get_violations_by_level(SafetyLevel.CRITICAL)
                if critical_violations:
                    self.logger.critical(f"System has {len(critical_violations)} critical safety violations")
                
                # Check safety score
                stats = self.get_safety_statistics()
                if stats["safety_score"] < 50.0:
                    self.logger.warning(f"Low safety score: {stats['safety_score']:.1f}")
                
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in safety health monitoring: {e}")
                await asyncio.sleep(60)


# Global safety validator instance
_global_safety_validator = None

def get_global_safety_validator() -> SafetyValidator:
    """Get global safety validator instance."""
    global _global_safety_validator
    if _global_safety_validator is None:
        _global_safety_validator = SafetyValidator()
    return _global_safety_validator

async def validate_safety(context: Dict[str, Any]) -> SafetyCheck:
    """Convenience function to validate safety."""
    return await get_global_safety_validator().validate(context)
