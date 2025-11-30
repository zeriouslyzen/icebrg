"""
Specification Generator for ICEBURG Self-Improvement

Analyzes ICEBURG performance and generates IIR specifications for improvements.
Provides multiple optimization strategies and validates specifications against contracts.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceAnalysis:
    """Analysis of current ICEBURG performance."""
    timestamp: float
    bottlenecks: List[str] = field(default_factory=list)
    slow_components: List[str] = field(default_factory=list)
    memory_hogs: List[str] = field(default_factory=list)
    accuracy_issues: List[str] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class OptimizationOpportunity:
    """Identified optimization opportunity."""
    name: str
    description: str
    impact_score: float  # 0-1, higher is better
    effort_score: float  # 0-1, lower is better
    risk_level: str  # low, medium, high
    affected_components: List[str] = field(default_factory=list)
    expected_improvement: float = 0.0
    optimization_type: str = ""  # performance, accuracy, efficiency, reliability


@dataclass
class TaskSpec:
    """IIR Task Specification for improvements."""
    name: str
    description: str = ""
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    implementation: Dict[str, Any] = field(default_factory=dict)
    optimization_targets: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyValidation:
    """Safety validation result for a specification."""
    spec_name: str
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    risk_assessment: str = "low"
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ImpactEstimate:
    """Impact estimate for a specification."""
    spec_name: str
    performance_improvement: float = 0.0
    accuracy_improvement: float = 0.0
    efficiency_improvement: float = 0.0
    reliability_improvement: float = 0.0
    resource_savings: Dict[str, float] = field(default_factory=dict)
    confidence_level: float = 0.0
    implementation_effort: float = 0.0


class SpecificationGenerator:
    """
    Generates IIR specifications for ICEBURG self-improvement.
    
    Analyzes current performance, identifies optimization opportunities,
    and creates validated specifications for improvements.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize specification generator."""
        self.config = config or {}
        self.performance_tracker = None
        self.benchmark_suite = None
        
        # Optimization strategies
        self.strategies = {
            "performance": self._generate_performance_spec,
            "accuracy": self._generate_accuracy_spec,
            "efficiency": self._generate_efficiency_spec,
            "reliability": self._generate_reliability_spec
        }
        
        # Component knowledge base
        self.component_knowledge = {
            "protocol": {
                "description": "Main protocol execution",
                "optimization_targets": ["response_time", "parallelization", "caching"],
                "dependencies": ["agents", "memory", "validation"]
            },
            "agents": {
                "description": "Agent execution and coordination",
                "optimization_targets": ["execution_time", "coordination", "resource_usage"],
                "dependencies": ["memory", "communication"]
            },
            "memory": {
                "description": "Memory and knowledge management",
                "optimization_targets": ["retrieval_speed", "storage_efficiency", "cache_hit_rate"],
                "dependencies": ["vectorstore", "database"]
            },
            "validation": {
                "description": "Response validation and quality control",
                "optimization_targets": ["validation_speed", "accuracy", "false_positive_rate"],
                "dependencies": ["agents", "memory"]
            }
        }
        
        # Safety constraints
        self.safety_constraints = {
            "max_memory_increase": 0.2,  # 20% max memory increase
            "max_cpu_increase": 0.3,     # 30% max CPU increase
            "min_accuracy_threshold": 0.7,
            "max_response_time_increase": 0.1,  # 10% max response time increase
            "forbidden_modifications": [
                "core_security",
                "authentication",
                "data_integrity"
            ]
        }
    
    def generate_improvement_specifications(self, performance_data: Dict[str, Any]) -> List[TaskSpec]:
        """Generate improvement specifications based on performance data."""
        logger.info("Generating improvement specifications")
        
        try:
            specs = []
            
            # Generate performance optimization specs
            if "response_time" in performance_data or "memory_usage" in performance_data:
                perf_spec = self._generate_performance_spec_from_data(performance_data)
                if perf_spec:
                    specs.append(perf_spec)
            
            # Generate accuracy improvement specs
            if "accuracy" in performance_data:
                acc_spec = self._generate_accuracy_spec_from_data(performance_data)
                if acc_spec:
                    specs.append(acc_spec)
            
            # Generate efficiency improvement specs
            if "memory_usage" in performance_data or "cpu_usage" in performance_data:
                eff_spec = self._generate_efficiency_spec_from_data(performance_data)
                if eff_spec:
                    specs.append(eff_spec)
            
            # Generate reliability improvement specs
            if "error_rate" in performance_data or "success_rate" in performance_data:
                rel_spec = self._generate_reliability_spec_from_data(performance_data)
                if rel_spec:
                    specs.append(rel_spec)
            
            # If no specific improvements found, generate generic improvements
            if not specs:
                generic_spec = self._generate_generic_spec(performance_data)
                if generic_spec:
                    specs.append(generic_spec)
            
            logger.info(f"Generated {len(specs)} improvement specifications")
            return specs
            
        except Exception as e:
            logger.error(f"Error generating improvement specifications: {e}")
            return []
    
    async def analyze_system_performance(self) -> PerformanceAnalysis:
        """Analyze current ICEBURG performance to identify issues."""
        logger.info("Analyzing ICEBURG system performance")
        
        try:
            # Get performance data from tracker
            if self.performance_tracker:
                summary = self.performance_tracker.get_performance_summary(hours=24)
            else:
                # Fallback to simulated data
                summary = self._get_simulated_performance_data()
            
            # Analyze bottlenecks
            bottlenecks = self._identify_bottlenecks(summary)
            
            # Identify slow components
            slow_components = self._identify_slow_components(summary)
            
            # Identify memory issues
            memory_hogs = self._identify_memory_issues(summary)
            
            # Identify accuracy issues
            accuracy_issues = self._identify_accuracy_issues(summary)
            
            # Extract resource usage
            resource_usage = summary.get("averages", {})
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                bottlenecks, slow_components, memory_hogs, accuracy_issues
            )
            
            analysis = PerformanceAnalysis(
                timestamp=time.time(),
                bottlenecks=bottlenecks,
                slow_components=slow_components,
                memory_hogs=memory_hogs,
                accuracy_issues=accuracy_issues,
                resource_usage=resource_usage,
                performance_metrics=summary.get("averages", {}),
                recommendations=recommendations
            )
            
            logger.info(f"Performance analysis completed: {len(bottlenecks)} bottlenecks, {len(recommendations)} recommendations")
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return PerformanceAnalysis(timestamp=time.time())
    
    def identify_optimization_opportunities(self, analysis: PerformanceAnalysis) -> List[OptimizationOpportunity]:
        """Identify optimization opportunities from performance analysis."""
        logger.info("Identifying optimization opportunities")
        
        opportunities = []
        
        # Performance opportunities
        if "response_time" in analysis.bottlenecks:
            opportunities.append(OptimizationOpportunity(
                name="response_time_optimization",
                description="Optimize response time through parallelization and caching",
                impact_score=0.8,
                effort_score=0.6,
                risk_level="low",
                affected_components=["protocol", "agents"],
                expected_improvement=0.3,
                optimization_type="performance"
            ))
        
        # Memory opportunities
        if analysis.memory_hogs:
            opportunities.append(OptimizationOpportunity(
                name="memory_optimization",
                description="Optimize memory usage through better data structures and caching",
                impact_score=0.7,
                effort_score=0.7,
                risk_level="medium",
                affected_components=["memory", "agents"],
                expected_improvement=0.25,
                optimization_type="efficiency"
            ))
        
        # Accuracy opportunities
        if analysis.accuracy_issues:
            opportunities.append(OptimizationOpportunity(
                name="accuracy_improvement",
                description="Improve accuracy through better validation and reasoning",
                impact_score=0.9,
                effort_score=0.8,
                risk_level="low",
                affected_components=["validation", "agents"],
                expected_improvement=0.15,
                optimization_type="accuracy"
            ))
        
        # Reliability opportunities
        if "error_rate" in analysis.performance_metrics:
            error_rate = analysis.performance_metrics["error_rate"]
            if error_rate > 0.05:  # 5% error rate threshold
                opportunities.append(OptimizationOpportunity(
                    name="reliability_improvement",
                    description="Improve reliability through better error handling and validation",
                    impact_score=0.8,
                    effort_score=0.5,
                    risk_level="low",
                    affected_components=["protocol", "validation"],
                    expected_improvement=0.2,
                    optimization_type="reliability"
                ))
        
        # Sort by impact/effort ratio
        opportunities.sort(key=lambda x: x.impact_score / (x.effort_score + 0.1), reverse=True)
        
        logger.info(f"Identified {len(opportunities)} optimization opportunities")
        return opportunities
    
    def generate_improvement_spec(self, opportunity: OptimizationOpportunity) -> TaskSpec:
        """Generate IIR specification for an optimization opportunity."""
        logger.info(f"Generating improvement spec for: {opportunity.name}")
        
        try:
            # Generate specification based on optimization type
            if opportunity.optimization_type == "performance":
                spec = self._generate_performance_spec(opportunity)
            elif opportunity.optimization_type == "accuracy":
                spec = self._generate_accuracy_spec(opportunity)
            elif opportunity.optimization_type == "efficiency":
                spec = self._generate_efficiency_spec(opportunity)
            elif opportunity.optimization_type == "reliability":
                spec = self._generate_reliability_spec(opportunity)
            else:
                spec = self._generate_generic_spec(opportunity)
            
            # Add safety constraints
            spec.safety_constraints = self._generate_safety_constraints(opportunity)
            
            # Add metadata
            spec.metadata = {
                "opportunity_name": opportunity.name,
                "optimization_type": opportunity.optimization_type,
                "expected_improvement": opportunity.expected_improvement,
                "risk_level": opportunity.risk_level,
                "generated_at": time.time()
            }
            
            logger.info(f"Generated spec: {spec.name}")
            return spec
            
        except Exception as e:
            logger.error(f"Failed to generate spec for {opportunity.name}: {e}")
            return self._generate_fallback_spec(opportunity)
    
    def validate_spec_safety(self, spec: TaskSpec) -> SafetyValidation:
        """Validate specification against safety constraints."""
        logger.info(f"Validating safety for spec: {spec.name}")
        
        violations = []
        warnings = []
        risk_assessment = "low"
        
        # Check resource constraints
        if "memory_usage" in spec.implementation:
            memory_increase = spec.implementation.get("memory_usage", 0)
            if memory_increase > self.safety_constraints["max_memory_increase"]:
                violations.append(f"Memory increase too high: {memory_increase:.1%} > {self.safety_constraints['max_memory_increase']:.1%}")
                risk_assessment = "high"
        
        if "cpu_usage" in spec.implementation:
            cpu_increase = spec.implementation.get("cpu_usage", 0)
            if cpu_increase > self.safety_constraints["max_cpu_increase"]:
                violations.append(f"CPU increase too high: {cpu_increase:.1%} > {self.safety_constraints['max_cpu_increase']:.1%}")
                risk_assessment = "high"
        
        # Check accuracy constraints
        if "min_accuracy" in spec.implementation:
            min_accuracy = spec.implementation.get("min_accuracy", 1.0)
            if min_accuracy < self.safety_constraints["min_accuracy_threshold"]:
                violations.append(f"Accuracy too low: {min_accuracy:.2f} < {self.safety_constraints['min_accuracy_threshold']:.2f}")
                risk_assessment = "high"
        
        # Check for forbidden modifications
        for forbidden in self.safety_constraints["forbidden_modifications"]:
            if forbidden in spec.name.lower() or forbidden in spec.description.lower():
                violations.append(f"Forbidden modification detected: {forbidden}")
                risk_assessment = "critical"
        
        # Check safety constraints in spec
        for constraint in spec.safety_constraints:
            if "bypass" in constraint.lower() or "disable" in constraint.lower():
                warnings.append(f"Potentially unsafe constraint: {constraint}")
                if risk_assessment == "low":
                    risk_assessment = "medium"
        
        # Generate recommendations
        recommendations = []
        if violations:
            recommendations.append("Address all violations before implementation")
        if warnings:
            recommendations.append("Review warnings and consider additional safety measures")
        if risk_assessment in ["high", "critical"]:
            recommendations.append("Require human approval before implementation")
        
        return SafetyValidation(
            spec_name=spec.name,
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
    
    def estimate_improvement_impact(self, spec: TaskSpec) -> ImpactEstimate:
        """Estimate the impact of implementing a specification."""
        logger.info(f"Estimating impact for spec: {spec.name}")
        
        # Base estimates from spec metadata
        expected_improvement = spec.metadata.get("expected_improvement", 0.1)
        optimization_type = spec.metadata.get("optimization_type", "performance")
        
        # Calculate impact based on optimization type
        if optimization_type == "performance":
            performance_improvement = expected_improvement
            accuracy_improvement = 0.0
            efficiency_improvement = expected_improvement * 0.5
            reliability_improvement = 0.0
        elif optimization_type == "accuracy":
            performance_improvement = 0.0
            accuracy_improvement = expected_improvement
            efficiency_improvement = 0.0
            reliability_improvement = expected_improvement * 0.3
        elif optimization_type == "efficiency":
            performance_improvement = expected_improvement * 0.3
            accuracy_improvement = 0.0
            efficiency_improvement = expected_improvement
            reliability_improvement = 0.0
        elif optimization_type == "reliability":
            performance_improvement = 0.0
            accuracy_improvement = expected_improvement * 0.2
            efficiency_improvement = 0.0
            reliability_improvement = expected_improvement
        else:
            performance_improvement = expected_improvement * 0.5
            accuracy_improvement = expected_improvement * 0.3
            efficiency_improvement = expected_improvement * 0.4
            reliability_improvement = expected_improvement * 0.2
        
        # Estimate resource savings
        resource_savings = {}
        if "memory_optimization" in spec.name:
            resource_savings["memory"] = expected_improvement * 0.2
        if "cpu_optimization" in spec.name:
            resource_savings["cpu"] = expected_improvement * 0.15
        
        # Estimate implementation effort
        implementation_effort = 0.5  # Base effort
        if "parallel" in spec.name:
            implementation_effort += 0.3
        if "cache" in spec.name:
            implementation_effort += 0.2
        if "validation" in spec.name:
            implementation_effort += 0.4
        
        # Calculate confidence level
        confidence_level = 0.7  # Base confidence
        if spec.metadata.get("risk_level") == "low":
            confidence_level += 0.2
        elif spec.metadata.get("risk_level") == "high":
            confidence_level -= 0.3
        
        return ImpactEstimate(
            spec_name=spec.name,
            performance_improvement=performance_improvement,
            accuracy_improvement=accuracy_improvement,
            efficiency_improvement=efficiency_improvement,
            reliability_improvement=reliability_improvement,
            resource_savings=resource_savings,
            confidence_level=confidence_level,
            implementation_effort=implementation_effort
        )
    
    def _identify_bottlenecks(self, summary: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks from summary data."""
        bottlenecks = []
        
        averages = summary.get("averages", {})
        
        # Check response time
        if "response_time" in averages:
            if averages["response_time"] > 30.0:  # 30 second threshold
                bottlenecks.append("response_time")
        
        # Check throughput
        if "throughput_rps" in averages:
            if averages["throughput_rps"] < 0.1:  # Less than 0.1 queries per second
                bottlenecks.append("throughput")
        
        # Check error rate
        if "error_rate" in averages:
            if averages["error_rate"] > 0.05:  # 5% error rate threshold
                bottlenecks.append("error_rate")
        
        return bottlenecks
    
    def _identify_slow_components(self, summary: Dict[str, Any]) -> List[str]:
        """Identify slow components from summary data."""
        slow_components = []
        
        # This would typically analyze component-specific metrics
        # For now, we'll use heuristics based on overall performance
        
        averages = summary.get("averages", {})
        if "response_time" in averages and averages["response_time"] > 20.0:
            slow_components.extend(["protocol", "agents"])
        
        if "memory_usage_mb" in averages and averages["memory_usage_mb"] > 1000:
            slow_components.append("memory")
        
        return slow_components
    
    def _identify_memory_issues(self, summary: Dict[str, Any]) -> List[str]:
        """Identify memory-related issues."""
        memory_issues = []
        
        averages = summary.get("averages", {})
        if "memory_usage_mb" in averages:
            if averages["memory_usage_mb"] > 1500:  # 1.5GB threshold
                memory_issues.append("high_memory_usage")
        
        return memory_issues
    
    def _identify_accuracy_issues(self, summary: Dict[str, Any]) -> List[str]:
        """Identify accuracy-related issues."""
        accuracy_issues = []
        
        averages = summary.get("averages", {})
        if "accuracy" in averages:
            if averages["accuracy"] < 0.8:  # 80% accuracy threshold
                accuracy_issues.append("low_accuracy")
        
        return accuracy_issues
    
    def _generate_recommendations(self, bottlenecks: List[str], slow_components: List[str], 
                                memory_hogs: List[str], accuracy_issues: List[str]) -> List[str]:
        """Generate recommendations based on identified issues."""
        recommendations = []
        
        if "response_time" in bottlenecks:
            recommendations.append("Implement parallel processing for agent execution")
            recommendations.append("Add intelligent caching for frequent queries")
        
        if "throughput" in bottlenecks:
            recommendations.append("Optimize database queries and indexing")
            recommendations.append("Implement connection pooling")
        
        if "error_rate" in bottlenecks:
            recommendations.append("Improve error handling and validation")
            recommendations.append("Add circuit breaker pattern for external services")
        
        if "memory" in slow_components:
            recommendations.append("Optimize memory usage in agent execution")
            recommendations.append("Implement memory pooling for large objects")
        
        if memory_hogs:
            recommendations.append("Review and optimize data structures")
            recommendations.append("Implement lazy loading for large datasets")
        
        if accuracy_issues:
            recommendations.append("Enhance validation pipeline")
            recommendations.append("Improve reasoning algorithms")
        
        return recommendations
    
    def _generate_performance_spec(self, opportunity: OptimizationOpportunity) -> TaskSpec:
        """Generate performance optimization specification."""
        return TaskSpec(
            name=f"performance_{opportunity.name}",
            description=f"Performance optimization for {opportunity.name}",
            inputs=[
                {"name": "query", "type": "string"},
                {"name": "context", "type": "object"}
            ],
            outputs=[
                {"name": "response", "type": "string"},
                {"name": "metrics", "type": "object"}
            ],
            preconditions=[
                "query is not empty",
                "context is valid"
            ],
            postconditions=[
                "response_time < baseline * 0.7",
                "response is valid"
            ],
            implementation={
                "parallel_execution": True,
                "caching_enabled": True,
                "optimization_level": "high"
            },
            optimization_targets=["response_time", "throughput"]
        )
    
    def _generate_accuracy_spec(self, opportunity: OptimizationOpportunity) -> TaskSpec:
        """Generate accuracy improvement specification."""
        return TaskSpec(
            name=f"accuracy_{opportunity.name}",
            description=f"Accuracy improvement for {opportunity.name}",
            inputs=[
                {"name": "query", "type": "string"},
                {"name": "evidence", "type": "array"}
            ],
            outputs=[
                {"name": "response", "type": "string"},
                {"name": "confidence", "type": "float"}
            ],
            preconditions=[
                "query is not empty",
                "evidence is not empty"
            ],
            postconditions=[
                "confidence > 0.8",
                "response is factually accurate"
            ],
            implementation={
                "validation_enabled": True,
                "reasoning_depth": "deep",
                "evidence_weight": "high"
            },
            optimization_targets=["accuracy", "confidence"]
        )
    
    def _generate_efficiency_spec(self, opportunity: OptimizationOpportunity) -> TaskSpec:
        """Generate efficiency optimization specification."""
        return TaskSpec(
            name=f"efficiency_{opportunity.name}",
            description=f"Resource efficiency improvement for {opportunity.name}",
            inputs=[
                {"name": "query", "type": "string"},
                {"name": "resources", "type": "object"}
            ],
            outputs=[
                {"name": "response", "type": "string"},
                {"name": "resource_usage", "type": "object"}
            ],
            preconditions=[
                "query is not empty",
                "resources are available"
            ],
            postconditions=[
                "memory_usage < baseline * 0.8",
                "cpu_usage < baseline * 0.9"
            ],
            implementation={
                "memory_optimization": True,
                "resource_pooling": True,
                "lazy_loading": True
            },
            optimization_targets=["memory_usage", "cpu_usage"]
        )
    
    def _generate_reliability_spec(self, opportunity: OptimizationOpportunity) -> TaskSpec:
        """Generate reliability improvement specification."""
        return TaskSpec(
            name=f"reliability_{opportunity.name}",
            description=f"Reliability improvement for {opportunity.name}",
            inputs=[
                {"name": "query", "type": "string"},
                {"name": "fallback_options", "type": "array"}
            ],
            outputs=[
                {"name": "response", "type": "string"},
                {"name": "success", "type": "boolean"}
            ],
            preconditions=[
                "query is not empty",
                "fallback_options are available"
            ],
            postconditions=[
                "success = true",
                "error_rate < 0.02"
            ],
            implementation={
                "error_handling": "comprehensive",
                "circuit_breaker": True,
                "retry_mechanism": True
            },
            optimization_targets=["error_rate", "success_rate"]
        )
    
    def _generate_generic_spec(self, opportunity: OptimizationOpportunity) -> TaskSpec:
        """Generate generic specification for unknown optimization types."""
        return TaskSpec(
            name=f"generic_{opportunity.name}",
            description=f"Generic improvement for {opportunity.name}",
            inputs=[
                {"name": "input", "type": "any"}
            ],
            outputs=[
                {"name": "output", "type": "any"}
            ],
            preconditions=[
                "input is valid"
            ],
            postconditions=[
                "output is valid",
                "improvement_achieved"
            ],
            implementation={
                "optimization_enabled": True
            },
            optimization_targets=["general_improvement"]
        )
    
    def _generate_fallback_spec(self, opportunity: OptimizationOpportunity) -> TaskSpec:
        """Generate fallback specification when generation fails."""
        return TaskSpec(
            name=f"fallback_{opportunity.name}",
            description=f"Fallback improvement for {opportunity.name}",
            inputs=[{"name": "input", "type": "any"}],
            outputs=[{"name": "output", "type": "any"}],
            preconditions=["input is valid"],
            postconditions=["output is valid"],
            implementation={"fallback_mode": True},
            optimization_targets=["basic_improvement"]
        )
    
    def _generate_safety_constraints(self, opportunity: OptimizationOpportunity) -> List[str]:
        """Generate safety constraints for an opportunity."""
        constraints = [
            "maintain_data_integrity",
            "preserve_security_boundaries",
            "ensure_backward_compatibility"
        ]
        
        if opportunity.risk_level == "high":
            constraints.extend([
                "require_human_approval",
                "implement_rollback_mechanism",
                "monitor_continuously"
            ])
        
        if "memory" in opportunity.name:
            constraints.append("limit_memory_usage_increase")
        
        if "performance" in opportunity.name:
            constraints.append("maintain_response_quality")
        
        return constraints
    
    def _get_simulated_performance_data(self) -> Dict[str, Any]:
        """Get simulated performance data for testing."""
        return {
            "averages": {
                "response_time": 25.0,
                "accuracy": 0.75,
                "memory_usage_mb": 1200.0,
                "cpu_usage_percent": 45.0,
                "throughput_rps": 0.05,
                "cache_hit_rate": 0.6,
                "error_rate": 0.08
            },
            "total_queries": 100,
            "success_rate": 0.92
        }
    
    def _generate_performance_spec_from_data(self, data: Dict[str, Any]) -> TaskSpec:
        """Generate performance optimization specification from performance data."""
        response_time = data.get("response_time", 0)
        memory_usage = data.get("memory_usage_mb", 0)
        
        return TaskSpec(
            name="performance_optimization",
            description=f"Optimize response time (current: {response_time:.2f}s) and memory usage (current: {memory_usage:.2f}MB)",
            inputs=[
                {"name": "query", "type": "string"},
                {"name": "context", "type": "object"}
            ],
            outputs=[
                {"name": "response", "type": "string"},
                {"name": "performance_metrics", "type": "object"}
            ],
            preconditions=[
                "query is not empty",
                "context is valid"
            ],
            postconditions=[
                "response_time < current_response_time",
                "memory_usage < current_memory_usage",
                "response quality maintained"
            ],
            optimization_targets=["response_time", "memory_usage", "throughput"],
            safety_constraints=[
                "no data loss",
                "maintain accuracy",
                "preserve functionality"
            ],
            metadata={
                "current_response_time": response_time,
                "current_memory_usage": memory_usage,
                "optimization_type": "performance"
            }
        )
    
    def _generate_accuracy_spec_from_data(self, data: Dict[str, Any]) -> TaskSpec:
        """Generate accuracy improvement specification from performance data."""
        accuracy = data.get("accuracy", 0)
        
        return TaskSpec(
            name="accuracy_improvement",
            description=f"Improve response accuracy (current: {accuracy:.2f})",
            inputs=[
                {"name": "query", "type": "string"},
                {"name": "context", "type": "object"}
            ],
            outputs=[
                {"name": "response", "type": "string"},
                {"name": "confidence_score", "type": "float"}
            ],
            preconditions=[
                "query is not empty",
                "context is valid"
            ],
            postconditions=[
                "accuracy > current_accuracy",
                "confidence_score > 0.8"
            ],
            optimization_targets=["accuracy", "confidence", "reliability"],
            safety_constraints=[
                "no hallucination",
                "maintain factual accuracy",
                "preserve context understanding"
            ],
            metadata={
                "current_accuracy": accuracy,
                "optimization_type": "accuracy"
            }
        )
    
    def _generate_efficiency_spec_from_data(self, data: Dict[str, Any]) -> TaskSpec:
        """Generate efficiency improvement specification from performance data."""
        memory_usage = data.get("memory_usage_mb", 0)
        cpu_usage = data.get("cpu_usage_percent", 0)
        
        return TaskSpec(
            name="efficiency_optimization",
            description=f"Optimize resource efficiency (memory: {memory_usage:.2f}MB, CPU: {cpu_usage:.2f}%)",
            inputs=[
                {"name": "query", "type": "string"},
                {"name": "context", "type": "object"}
            ],
            outputs=[
                {"name": "response", "type": "string"},
                {"name": "resource_usage", "type": "object"}
            ],
            preconditions=[
                "query is not empty",
                "context is valid"
            ],
            postconditions=[
                "memory_usage < current_memory_usage",
                "cpu_usage < current_cpu_usage",
                "response quality maintained"
            ],
            optimization_targets=["memory_usage", "cpu_usage", "resource_efficiency"],
            safety_constraints=[
                "no data loss",
                "maintain performance",
                "preserve functionality"
            ],
            metadata={
                "current_memory_usage": memory_usage,
                "current_cpu_usage": cpu_usage,
                "optimization_type": "efficiency"
            }
        )
    
    def _generate_reliability_spec_from_data(self, data: Dict[str, Any]) -> TaskSpec:
        """Generate reliability improvement specification from performance data."""
        error_rate = data.get("error_rate", 0)
        success_rate = data.get("success_rate", 100)
        
        return TaskSpec(
            name="reliability_improvement",
            description=f"Improve system reliability (error rate: {error_rate:.2f}%, success rate: {success_rate:.2f}%)",
            inputs=[
                {"name": "query", "type": "string"},
                {"name": "context", "type": "object"}
            ],
            outputs=[
                {"name": "response", "type": "string"},
                {"name": "error_handling", "type": "object"}
            ],
            preconditions=[
                "query is not empty",
                "context is valid"
            ],
            postconditions=[
                "error_rate < current_error_rate",
                "success_rate > current_success_rate",
                "robust error handling"
            ],
            optimization_targets=["error_rate", "success_rate", "reliability"],
            safety_constraints=[
                "graceful error handling",
                "no system crashes",
                "maintain data integrity"
            ],
            metadata={
                "current_error_rate": error_rate,
                "current_success_rate": success_rate,
                "optimization_type": "reliability"
            }
        )


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_specification_generator():
        # Create generator
        generator = SpecificationGenerator()
        
        # Analyze performance
        analysis = await generator.analyze_system_performance()
        print("Performance Analysis:")
        print(f"  Bottlenecks: {analysis.bottlenecks}")
        print(f"  Slow Components: {analysis.slow_components}")
        print(f"  Memory Issues: {analysis.memory_hogs}")
        print(f"  Accuracy Issues: {analysis.accuracy_issues}")
        print(f"  Recommendations: {analysis.recommendations}")
        
        # Identify opportunities
        opportunities = generator.identify_optimization_opportunities(analysis)
        print(f"\nOptimization Opportunities: {len(opportunities)}")
        for opp in opportunities:
            print(f"  {opp.name}: {opp.description} (Impact: {opp.impact_score:.2f}, Effort: {opp.effort_score:.2f})")
        
        # Generate spec for first opportunity
        if opportunities:
            spec = generator.generate_improvement_spec(opportunities[0])
            print(f"\nGenerated Spec: {spec.name}")
            print(f"  Inputs: {spec.inputs}")
            print(f"  Outputs: {spec.outputs}")
            print(f"  Safety Constraints: {spec.safety_constraints}")
            
            # Validate safety
            safety = generator.validate_spec_safety(spec)
            print(f"\nSafety Validation:")
            print(f"  Passed: {safety.passed}")
            print(f"  Risk Assessment: {safety.risk_assessment}")
            print(f"  Violations: {safety.violations}")
            
            # Estimate impact
            impact = generator.estimate_improvement_impact(spec)
            print(f"\nImpact Estimate:")
            print(f"  Performance Improvement: {impact.performance_improvement:.1%}")
            print(f"  Accuracy Improvement: {impact.accuracy_improvement:.1%}")
            print(f"  Confidence Level: {impact.confidence_level:.1%}")
    
    # Run test
    asyncio.run(test_specification_generator())
