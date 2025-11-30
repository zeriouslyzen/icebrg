"""
Visual Governance Validator
Validates visual generation against constitutional rules
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import re


@dataclass
class GovernanceViolation:
    """Represents a governance violation"""
    article: str
    section: str
    rule: str
    severity: str  # "critical", "major", "minor"
    description: str
    location: Optional[str] = None
    recommendation: str = ""


@dataclass
class GovernanceValidation:
    """Result of governance validation"""
    passed: bool
    violations: List[GovernanceViolation]
    critical_violations: List[GovernanceViolation]
    major_violations: List[GovernanceViolation]
    minor_violations: List[GovernanceViolation]
    compliance_score: float  # 0.0 - 1.0


class VisualGovernanceValidator:
    """Validates visual generation against constitution"""
    
    def __init__(self, constitution_path: Optional[Path] = None):
        if constitution_path is None:
            constitution_path = Path("governance/visual_generation_constitution.md")
        
        self.constitution_path = Path(constitution_path)
        self.constitution = self._load_constitution()
    
    def _load_constitution(self) -> Dict[str, Any]:
        """Load constitution from markdown file"""
        if not self.constitution_path.exists():
            print(f"Warning: Constitution not found at {self.constitution_path}")
            return {}
        
        # Parse markdown constitution
        # In production, would parse the full document
        # For now, return placeholder
        return {
            "version": "1.0",
            "effective_date": "2025-10-14",
            "articles": {}
        }
    
    def validate_generation(
        self,
        visual_result: Any  # VisualGenerationResult
    ) -> GovernanceValidation:
        """
        Validate visual generation result against constitution
        
        Args:
            visual_result: VisualGenerationResult from VisualArchitect
            
        Returns:
            Governance validation result
        """
        violations = []
        
        # Article I: Security validation
        violations.extend(self._validate_security(visual_result))
        
        # Article II: Accessibility validation
        violations.extend(self._validate_accessibility(visual_result))
        
        # Article III: Performance validation
        violations.extend(self._validate_performance(visual_result))
        
        # Article IV: Privacy validation
        violations.extend(self._validate_privacy(visual_result))
        
        # Classify violations by severity
        critical = [v for v in violations if v.severity == "critical"]
        major = [v for v in violations if v.severity == "major"]
        minor = [v for v in violations if v.severity == "minor"]
        
        # Calculate compliance score
        total_checks = len(violations) + 50  # Assume 50 total checks
        passed_checks = total_checks - len(critical) * 3 - len(major) * 2 - len(minor)
        compliance_score = max(0.0, passed_checks / total_checks)
        
        # Pass if no critical violations
        passed = len(critical) == 0
        
        return GovernanceValidation(
            passed=passed,
            violations=violations,
            critical_violations=critical,
            major_violations=major,
            minor_violations=minor,
            compliance_score=compliance_score
        )
    
    def _validate_security(self, visual_result: Any) -> List[GovernanceViolation]:
        """Validate Article I: Security Rules"""
        violations = []
        
        # Check for eval() usage
        for backend_type, artifacts in visual_result.artifacts.items():
            if backend_type.value == "html5":
                js_code = artifacts.js
                if re.search(r'\beval\s*\(', js_code):
                    violations.append(GovernanceViolation(
                        article="I",
                        section="1.1",
                        rule="NO use of eval()",
                        severity="critical",
                        description="eval() detected in generated JavaScript",
                        location=f"{backend_type.value}/script.js",
                        recommendation="Remove eval() usage and use safe alternatives"
                    ))
        
        # Check for inline event handlers in HTML
        for backend_type, artifacts in visual_result.artifacts.items():
            if backend_type.value == "html5":
                html_code = artifacts.html
                if re.search(r'on\w+\s*=\s*["\']', html_code):
                    violations.append(GovernanceViolation(
                        article="I",
                        section="1.1",
                        rule="NO inline event handlers",
                        severity="major",
                        description="Inline event handlers detected in HTML",
                        location=f"{backend_type.value}/index.html",
                        recommendation="Use event delegation instead of inline handlers"
                    ))
        
        # Check for dangerous protocols in URLs
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:']
        for component in visual_result.ir.ui_components.values():
            for prop_name, prop_value in component.props.items():
                if prop_name in ['href', 'src']:
                    value_str = str(prop_value).lower()
                    for protocol in dangerous_protocols:
                        if protocol in value_str:
                            violations.append(GovernanceViolation(
                                article="I",
                                section="1.3",
                                rule=f"PROHIBITED protocol: {protocol}",
                                severity="critical",
                                description=f"Dangerous protocol {protocol} detected",
                                location=f"component_{component.id}",
                                recommendation="Use safe protocols (https:, mailto:, etc.)"
                            ))
        
        return violations
    
    def _validate_accessibility(self, visual_result: Any) -> List[GovernanceViolation]:
        """Validate Article II: Accessibility Rules"""
        violations = []
        
        # Check WCAG compliance
        accessibility_report = visual_result.ir.accessibility_report
        if accessibility_report:
            wcag_level = accessibility_report.get("wcag_level", "")
            if wcag_level not in ["AA", "AAA"]:
                violations.append(GovernanceViolation(
                    article="II",
                    section="2.1",
                    rule="WCAG 2.1 Level AA minimum",
                    severity="critical",
                    description=f"WCAG level {wcag_level} does not meet minimum AA standard",
                    recommendation="Ensure all components meet WCAG 2.1 Level AA"
                ))
        
        # Check for missing alt text on images
        for comp_id, component in visual_result.ir.ui_components.items():
            if component.type.value == "image":
                has_alt = (
                    'alt' in component.props or
                    'alt' in component.accessibility_attrs
                )
                if not has_alt:
                    violations.append(GovernanceViolation(
                        article="II",
                        section="2.3",
                        rule="ALL images MUST have alt text",
                        severity="major",
                        description=f"Image component {comp_id} missing alt text",
                        location=comp_id,
                        recommendation="Add descriptive alt text or empty alt for decorative images"
                    ))
        
        # Check for keyboard navigation
        interactive_types = ["button", "input", "link"]
        for comp_id, component in visual_result.ir.ui_components.items():
            if component.type.value in interactive_types:
                has_tabindex = (
                    'tabindex' in component.props or
                    'tabindex' in component.accessibility_attrs
                )
                # Native elements are keyboard accessible by default
                # Only warn if explicitly set to negative
                if 'tabindex' in component.accessibility_attrs:
                    tabindex = component.accessibility_attrs['tabindex']
                    if tabindex.startswith('-'):
                        violations.append(GovernanceViolation(
                            article="II",
                            section="2.2",
                            rule="ALL interactive elements MUST be keyboard navigable",
                            severity="major",
                            description=f"Component {comp_id} has negative tabindex",
                            location=comp_id,
                            recommendation="Remove negative tabindex or provide alternative navigation"
                        ))
        
        return violations
    
    def _validate_performance(self, visual_result: Any) -> List[GovernanceViolation]:
        """Validate Article III: Performance Rules"""
        violations = []
        
        # Check load time budget
        perf_metrics = visual_result.ir.performance_metrics
        if perf_metrics:
            estimated_load = perf_metrics.get("estimated_load_ms", 0)
            if estimated_load > 1000:  # 1 second
                violations.append(GovernanceViolation(
                    article="III",
                    section="3.1",
                    rule="Time to First Contentful Paint MUST be < 1.0 seconds",
                    severity="major",
                    description=f"Estimated load time {estimated_load}ms exceeds 1000ms",
                    recommendation="Optimize critical path resources and enable code splitting"
                ))
            
            # Check bundle size
            estimated_size = perf_metrics.get("estimated_bundle_kb", 0)
            if estimated_size > 50:
                violations.append(GovernanceViolation(
                    article="III",
                    section="3.1",
                    rule="Initial bundle size MUST be < 50KB",
                    severity="minor",
                    description=f"Estimated bundle size {estimated_size}KB exceeds 50KB",
                    recommendation="Enable tree-shaking and code splitting"
                ))
        
        # Check for images without dimensions
        for comp_id, component in visual_result.ir.ui_components.items():
            if component.type.value == "image":
                has_dimensions = (
                    'width' in component.props and 'height' in component.props
                )
                if not has_dimensions:
                    violations.append(GovernanceViolation(
                        article="III",
                        section="3.2",
                        rule="ALL images MUST have explicit width and height",
                        severity="minor",
                        description=f"Image {comp_id} missing dimensions",
                        location=comp_id,
                        recommendation="Add width and height attributes to prevent layout shift"
                    ))
        
        return violations
    
    def _validate_privacy(self, visual_result: Any) -> List[GovernanceViolation]:
        """Validate Article IV: Privacy Rules"""
        violations = []
        
        # Check for tracking scripts
        for backend_type, artifacts in visual_result.artifacts.items():
            if backend_type.value == "html5":
                js_code = artifacts.js.lower()
                tracking_keywords = ['google-analytics', 'gtag', 'facebook', 'tracking', 'pixel']
                
                for keyword in tracking_keywords:
                    if keyword in js_code:
                        violations.append(GovernanceViolation(
                            article="IV",
                            section="4.1",
                            rule="NO tracking scripts without explicit user consent",
                            severity="major",
                            description=f"Potential tracking code detected: {keyword}",
                            location=f"{backend_type.value}/script.js",
                            recommendation="Remove tracking or implement consent management"
                        ))
                        break  # Only report once per backend
        
        return violations
    
    def generate_compliance_report(self, validation: GovernanceValidation) -> str:
        """Generate human-readable compliance report"""
        report = []
        
        report.append("=" * 60)
        report.append("ICEBURG VISUAL GENERATION COMPLIANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall status
        status = "✅ PASSED" if validation.passed else "❌ FAILED"
        report.append(f"Overall Status: {status}")
        report.append(f"Compliance Score: {validation.compliance_score:.1%}")
        report.append("")
        
        # Violation summary
        report.append("Violation Summary:")
        report.append(f"  Critical: {len(validation.critical_violations)}")
        report.append(f"  Major: {len(validation.major_violations)}")
        report.append(f"  Minor: {len(validation.minor_violations)}")
        report.append("")
        
        # Detailed violations
        if validation.violations:
            report.append("Detailed Violations:")
            report.append("")
            
            for violation in validation.violations:
                severity_symbol = {
                    "critical": "🔴",
                    "major": "🟠",
                    "minor": "🟡"
                }.get(violation.severity, "⚪")
                
                report.append(f"{severity_symbol} {violation.severity.upper()} - Article {violation.article}, Section {violation.section}")
                report.append(f"   Rule: {violation.rule}")
                report.append(f"   Description: {violation.description}")
                if violation.location:
                    report.append(f"   Location: {violation.location}")
                if violation.recommendation:
                    report.append(f"   Recommendation: {violation.recommendation}")
                report.append("")
        else:
            report.append("✅ No violations found!")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    validator = VisualGovernanceValidator()
    
    print("Visual Governance Validator initialized")
    print(f"Constitution path: {validator.constitution_path}")
    print(f"Constitution version: {validator.constitution.get('version', 'unknown')}")

