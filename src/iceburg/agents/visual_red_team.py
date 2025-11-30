"""
Visual Generation Red Team Agent
Adversarial testing for visual generation security and quality
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re
from pathlib import Path

from ..config import IceburgConfig
from ..iir.visual_tsl import UISpec
from ..iir.visual_ir import VisualIRFunction
from ..iir.backends import BackendType


class AttackType(Enum):
    """Types of adversarial attacks to test"""
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    ACCESSIBILITY_BYPASS = "accessibility_bypass"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MALICIOUS_CONTENT = "malicious_content"
    PRIVACY_VIOLATION = "privacy_violation"
    SECURITY_CONTRACT_VIOLATION = "security_contract_violation"


@dataclass
class AttackVector:
    """Represents an attack vector for testing"""
    attack_type: AttackType
    description: str
    malicious_input: str
    expected_behavior: str
    severity: str  # "low", "medium", "high", "critical"


@dataclass
class RedTeamResult:
    """Result of red team testing"""
    attack_vector: AttackVector
    success: bool
    vulnerability_found: bool
    details: str
    remediation: str
    severity_score: float


class VisualRedTeam:
    """Red team agent for visual generation security testing"""
    
    def __init__(self):
        self.attack_vectors = self._initialize_attack_vectors()
        self.security_patterns = self._initialize_security_patterns()
    
    def _initialize_attack_vectors(self) -> List[AttackVector]:
        """Initialize comprehensive attack vectors"""
        return [
            # XSS Attacks
            AttackVector(
                attack_type=AttackType.XSS,
                description="Script injection in button text",
                malicious_input="<script>alert('XSS')</script>",
                expected_behavior="Text should be escaped/sanitized",
                severity="high"
            ),
            AttackVector(
                attack_type=AttackType.XSS,
                description="Event handler injection",
                malicious_input="onclick=\"alert('XSS')\"",
                expected_behavior="Event handlers should be sanitized",
                severity="high"
            ),
            
            # Accessibility Bypass
            AttackVector(
                attack_type=AttackType.ACCESSIBILITY_BYPASS,
                description="Missing ARIA labels",
                malicious_input="Create a button without aria-label",
                expected_behavior="All interactive elements should have ARIA labels",
                severity="medium"
            ),
            AttackVector(
                attack_type=AttackType.ACCESSIBILITY_BYPASS,
                description="Keyboard navigation bypass",
                malicious_input="Create non-focusable interactive elements",
                expected_behavior="All interactive elements should be keyboard accessible",
                severity="medium"
            ),
            
            # Performance Degradation
            AttackVector(
                attack_type=AttackType.PERFORMANCE_DEGRADATION,
                description="Excessive DOM complexity",
                malicious_input="Create 1000 nested divs",
                expected_behavior="DOM complexity should be optimized",
                severity="medium"
            ),
            AttackVector(
                attack_type=AttackType.PERFORMANCE_DEGRADATION,
                description="Large bundle size",
                malicious_input="Generate massive CSS/JS",
                expected_behavior="Bundle size should stay under budget",
                severity="medium"
            ),
            
            # Privacy Violations
            AttackVector(
                attack_type=AttackType.PRIVACY_VIOLATION,
                description="External tracking scripts",
                malicious_input="Include Google Analytics or tracking pixels",
                expected_behavior="No external tracking should be included",
                severity="high"
            ),
            AttackVector(
                attack_type=AttackType.PRIVACY_VIOLATION,
                description="Data collection without consent",
                malicious_input="Collect user data without explicit consent",
                expected_behavior="No data collection without consent",
                severity="high"
            ),
            
            # Security Contract Violations
            AttackVector(
                attack_type=AttackType.SECURITY_CONTRACT_VIOLATION,
                description="External script loading",
                malicious_input="Load external JavaScript files",
                expected_behavior="No external scripts should be loaded",
                severity="critical"
            ),
            AttackVector(
                attack_type=AttackType.SECURITY_CONTRACT_VIOLATION,
                description="Inline eval usage",
                malicious_input="Use eval() or Function() constructor",
                expected_behavior="No eval() or Function() usage",
                severity="critical"
            )
        ]
    
    def _initialize_security_patterns(self) -> Dict[str, List[str]]:
        """Initialize security pattern detection"""
        return {
            "xss_patterns": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"eval\s*\(",
                r"Function\s*\("
            ],
            "injection_patterns": [
                r"';.*--",
                r"<.*>.*</.*>",
                r"\{\{.*\}\}",
                r"\$\{.*\}"
            ],
            "tracking_patterns": [
                r"google-analytics",
                r"gtag",
                r"facebook\.net",
                r"doubleclick\.net",
                r"googlesyndication"
            ],
            "external_script_patterns": [
                r"src\s*=\s*['\"][^'\"]*['\"]",
                r"https?://[^'\"]*\.js",
                r"cdn\.[^'\"]*\.js"
            ]
        }
    
    def test_visual_generation(self, visual_result: Any, verbose: bool = False) -> List[RedTeamResult]:
        """Run comprehensive red team tests on visual generation result"""
        results = []
        
        if verbose:
            print("[VISUAL_RED_TEAM] Running red team tests...")
        
        # Test each attack vector
        for attack_vector in self.attack_vectors:
            result = self._test_attack_vector(attack_vector, visual_result, verbose)
            results.append(result)
        
        # Test generated artifacts
        if hasattr(visual_result, 'artifacts'):
            for backend, artifacts in visual_result.artifacts.items():
                artifact_results = self._test_artifacts(backend, artifacts, verbose)
                results.extend(artifact_results)
        
        if verbose:
            vulnerabilities = [r for r in results if r.vulnerability_found]
            print(f"[VISUAL_RED_TEAM] Found {len(vulnerabilities)} potential vulnerabilities")
        
        return results
    
    def _test_attack_vector(self, attack_vector: AttackVector, visual_result: Any, verbose: bool) -> RedTeamResult:
        """Test a specific attack vector"""
        try:
            # Simulate the attack
            if attack_vector.attack_type == AttackType.XSS:
                success = self._test_xss_attack(attack_vector, visual_result)
            elif attack_vector.attack_type == AttackType.ACCESSIBILITY_BYPASS:
                success = self._test_accessibility_bypass(attack_vector, visual_result)
            elif attack_vector.attack_type == AttackType.PERFORMANCE_DEGRADATION:
                success = self._test_performance_degradation(attack_vector, visual_result)
            elif attack_vector.attack_type == AttackType.PRIVACY_VIOLATION:
                success = self._test_privacy_violation(attack_vector, visual_result)
            elif attack_vector.attack_type == AttackType.SECURITY_CONTRACT_VIOLATION:
                success = self._test_security_contract_violation(attack_vector, visual_result)
            else:
                success = False
            
            vulnerability_found = success  # If attack succeeds, vulnerability exists
            
            return RedTeamResult(
                attack_vector=attack_vector,
                success=success,
                vulnerability_found=vulnerability_found,
                details=f"Attack {attack_vector.description}: {'SUCCESS' if success else 'BLOCKED'}",
                remediation=self._get_remediation(attack_vector),
                severity_score=self._calculate_severity_score(attack_vector, success)
            )
            
        except Exception as e:
            return RedTeamResult(
                attack_vector=attack_vector,
                success=False,
                vulnerability_found=False,
                details=f"Test error: {str(e)}",
                remediation="Fix test implementation",
                severity_score=0.0
            )
    
    def _test_xss_attack(self, attack_vector: AttackVector, visual_result: Any) -> bool:
        """Test XSS attack vectors"""
        malicious_input = attack_vector.malicious_input
        
        # Check if malicious input appears in generated code
        if hasattr(visual_result, 'artifacts'):
            for backend, artifacts in visual_result.artifacts.items():
                if hasattr(artifacts, 'html'):
                    if malicious_input in artifacts.html:
                        return True
                if hasattr(artifacts, 'components'):
                    if malicious_input in artifacts.components:
                        return True
        
        return False
    
    def _test_accessibility_bypass(self, attack_vector: AttackVector, visual_result: Any) -> bool:
        """Test accessibility bypass attacks"""
        if hasattr(visual_result, 'ir'):
            ir = visual_result.ir
            for component_id, component in ir.ui_components.items():
                # Check for missing ARIA labels on interactive elements
                if component.type.value in ['button', 'input', 'link']:
                    if not component.accessibility_attrs.get('aria-label'):
                        return True
                
                # Check for missing keyboard navigation
                if component.type.value in ['button', 'input', 'link']:
                    if 'tabindex' not in component.props:
                        return True
        
        return False
    
    def _test_performance_degradation(self, attack_vector: AttackVector, visual_result: Any) -> bool:
        """Test performance degradation attacks"""
        if hasattr(visual_result, 'ir'):
            ir = visual_result.ir
            
            # Check DOM complexity
            if len(ir.ui_components) > 100:
                return True
            
            # Check for excessive nesting
            for component_id, component in ir.ui_components.items():
                if len(component.children_refs) > 10:
                    return True
        
        return False
    
    def _test_privacy_violation(self, attack_vector: AttackVector, visual_result: Any) -> bool:
        """Test privacy violation attacks"""
        if hasattr(visual_result, 'artifacts'):
            for backend, artifacts in visual_result.artifacts.items():
                # Check for tracking patterns
                for pattern in self.security_patterns['tracking_patterns']:
                    if hasattr(artifacts, 'html') and re.search(pattern, artifacts.html, re.IGNORECASE):
                        return True
                    if hasattr(artifacts, 'js') and re.search(pattern, artifacts.js, re.IGNORECASE):
                        return True
        
        return False
    
    def _test_security_contract_violation(self, attack_vector: AttackVector, visual_result: Any) -> bool:
        """Test security contract violations"""
        if hasattr(visual_result, 'artifacts'):
            for backend, artifacts in visual_result.artifacts.items():
                # Check for external scripts
                for pattern in self.security_patterns['external_script_patterns']:
                    if hasattr(artifacts, 'html') and re.search(pattern, artifacts.html, re.IGNORECASE):
                        return True
                
                # Check for eval usage
                for pattern in self.security_patterns['xss_patterns']:
                    if hasattr(artifacts, 'js') and re.search(pattern, artifacts.js, re.IGNORECASE):
                        return True
        
        return False
    
    def _test_artifacts(self, backend: BackendType, artifacts: Any, verbose: bool) -> List[RedTeamResult]:
        """Test generated artifacts for security issues"""
        results = []
        
        # Test HTML artifacts
        if hasattr(artifacts, 'html'):
            html_results = self._test_html_content(artifacts.html, backend)
            results.extend(html_results)
        
        # Test JavaScript artifacts
        if hasattr(artifacts, 'js'):
            js_results = self._test_javascript_content(artifacts.js, backend)
            results.extend(js_results)
        
        # Test CSS artifacts
        if hasattr(artifacts, 'css'):
            css_results = self._test_css_content(artifacts.css, backend)
            results.extend(css_results)
        
        return results
    
    def _test_html_content(self, html_content: str, backend: BackendType) -> List[RedTeamResult]:
        """Test HTML content for security issues"""
        results = []
        
        # Check for XSS patterns
        for pattern in self.security_patterns['xss_patterns']:
            if re.search(pattern, html_content, re.IGNORECASE):
                results.append(RedTeamResult(
                    attack_vector=AttackVector(
                        attack_type=AttackType.XSS,
                        description=f"XSS pattern detected in {backend.value} HTML",
                        malicious_input=pattern,
                        expected_behavior="No XSS patterns should be present",
                        severity="high"
                    ),
                    success=True,
                    vulnerability_found=True,
                    details=f"XSS pattern '{pattern}' found in HTML",
                    remediation="Sanitize HTML content and escape user input",
                    severity_score=0.8
                ))
        
        return results
    
    def _test_javascript_content(self, js_content: str, backend: BackendType) -> List[RedTeamResult]:
        """Test JavaScript content for security issues"""
        results = []
        
        # Check for eval usage
        if re.search(r'eval\s*\(', js_content, re.IGNORECASE):
            results.append(RedTeamResult(
                attack_vector=AttackVector(
                    attack_type=AttackType.SECURITY_CONTRACT_VIOLATION,
                    description=f"eval() usage detected in {backend.value} JavaScript",
                    malicious_input="eval()",
                    expected_behavior="No eval() usage should be present",
                    severity="critical"
                ),
                success=True,
                vulnerability_found=True,
                details="eval() function found in JavaScript",
                remediation="Remove eval() usage and use safer alternatives",
                severity_score=1.0
            ))
        
        return results
    
    def _test_css_content(self, css_content: str, backend: BackendType) -> List[RedTeamResult]:
        """Test CSS content for security issues"""
        results = []
        
        # Check for expression() usage (IE-specific but still a concern)
        if re.search(r'expression\s*\(', css_content, re.IGNORECASE):
            results.append(RedTeamResult(
                attack_vector=AttackVector(
                    attack_type=AttackType.XSS,
                    description=f"CSS expression() detected in {backend.value} CSS",
                    malicious_input="expression()",
                    expected_behavior="No CSS expressions should be present",
                    severity="medium"
                ),
                success=True,
                vulnerability_found=True,
                details="CSS expression() found",
                remediation="Remove CSS expressions and use standard CSS",
                severity_score=0.6
            ))
        
        return results
    
    def _get_remediation(self, attack_vector: AttackVector) -> str:
        """Get remediation advice for an attack vector"""
        remediation_map = {
            AttackType.XSS: "Implement proper input sanitization and output encoding",
            AttackType.ACCESSIBILITY_BYPASS: "Add proper ARIA labels and keyboard navigation support",
            AttackType.PERFORMANCE_DEGRADATION: "Optimize DOM structure and implement performance budgets",
            AttackType.PRIVACY_VIOLATION: "Remove tracking scripts and implement privacy-first design",
            AttackType.SECURITY_CONTRACT_VIOLATION: "Enforce security contracts and remove dangerous patterns"
        }
        return remediation_map.get(attack_vector.attack_type, "Review and fix security issue")
    
    def _calculate_severity_score(self, attack_vector: AttackVector, success: bool) -> float:
        """Calculate severity score for an attack result"""
        if not success:
            return 0.0
        
        severity_map = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.8,
            "critical": 1.0
        }
        return severity_map.get(attack_vector.severity, 0.5)
    
    def generate_security_report(self, results: List[RedTeamResult]) -> str:
        """Generate a comprehensive security report"""
        vulnerabilities = [r for r in results if r.vulnerability_found]
        
        report = f"""
# Visual Generation Security Report

## Summary
- Total Tests: {len(results)}
- Vulnerabilities Found: {len(vulnerabilities)}
- Security Score: {1.0 - (sum(r.severity_score for r in vulnerabilities) / len(results)):.2f}

## Vulnerabilities
"""
        
        for result in vulnerabilities:
            report += f"""
### {result.attack_vector.description}
- **Type**: {result.attack_vector.attack_type.value}
- **Severity**: {result.attack_vector.severity}
- **Details**: {result.details}
- **Remediation**: {result.remediation}
- **Score**: {result.severity_score:.2f}
"""
        
        if not vulnerabilities:
            report += "\n✅ No vulnerabilities found! Visual generation is secure."
        
        return report


def run_visual_red_team(cfg: IceburgConfig, visual_result: Any, verbose: bool = False) -> str:
    """Run red team testing on visual generation result"""
    red_team = VisualRedTeam()
    results = red_team.test_visual_generation(visual_result, verbose)
    return red_team.generate_security_report(results)
