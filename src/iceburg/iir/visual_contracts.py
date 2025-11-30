"""
Visual Contract Language
Extends ICEBURG's contract system with visual/UI-specific validation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import re


class ContractType(Enum):
    """Types of visual contracts"""
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    PRIVACY = "privacy"
    SEMANTIC = "semantic"


@dataclass
class ContractViolation:
    """Represents a contract violation"""
    contract_type: ContractType
    contract_name: str
    severity: str  # "error", "warning", "info"
    description: str
    location: Optional[str] = None
    recommendation: Optional[str] = None


class VisualContractValidator:
    """Validates visual IR against contracts"""
    
    def __init__(self):
        # Security contracts
        self.security_contracts = {
            "no_external_scripts": self._check_no_external_scripts,
            "no_inline_eval": self._check_no_inline_eval,
            "sanitized_content": self._check_sanitized_content,
            "csp_compliant": self._check_csp_compliant,
            "no_dangerous_urls": self._check_no_dangerous_urls
        }
        
        # Accessibility contracts
        self.accessibility_contracts = {
            "wcag_aa_compliant": self._check_wcag_aa,
            "keyboard_navigable": self._check_keyboard_navigable,
            "screen_reader_compatible": self._check_screen_reader,
            "has_aria_labels": self._check_aria_labels,
            "sufficient_contrast": self._check_contrast
        }
        
        # Performance contracts
        self.performance_contracts = {
            "fast_load": self._check_fast_load,
            "small_bundle": self._check_small_bundle,
            "no_layout_shift": self._check_no_layout_shift,
            "optimized_images": self._check_optimized_images,
            "minimal_reflows": self._check_minimal_reflows
        }
        
        # Privacy contracts
        self.privacy_contracts = {
            "no_tracking": self._check_no_tracking,
            "no_third_party_embeds": self._check_no_third_party,
            "local_storage_only": self._check_local_storage,
            "no_pii_exposure": self._check_no_pii
        }
    
    def validate(self, visual_ir: Any, contracts: List[str]) -> List[ContractViolation]:
        """Validate visual IR against specified contracts"""
        violations = []
        
        for contract_name in contracts:
            # Check security contracts
            if contract_name in self.security_contracts:
                violation = self.security_contracts[contract_name](visual_ir)
                if violation:
                    violations.append(violation)
            
            # Check accessibility contracts
            elif contract_name in self.accessibility_contracts:
                violation = self.accessibility_contracts[contract_name](visual_ir)
                if violation:
                    violations.append(violation)
            
            # Check performance contracts
            elif contract_name in self.performance_contracts:
                violation = self.performance_contracts[contract_name](visual_ir)
                if violation:
                    violations.append(violation)
            
            # Check privacy contracts
            elif contract_name in self.privacy_contracts:
                violation = self.privacy_contracts[contract_name](visual_ir)
                if violation:
                    violations.append(violation)
        
        return violations
    
    # Security contract checkers
    
    def _check_no_external_scripts(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for external script loading"""
        # Scan interaction handlers for external script references
        if hasattr(visual_ir, 'interaction_graph'):
            for edge in visual_ir.interaction_graph.edges:
                handler_fn = visual_ir.interaction_graph.handler_functions.get(edge.handler_fn_id)
                if handler_fn:
                    # Check for script loading patterns
                    fn_str = str(handler_fn)
                    if any(pattern in fn_str.lower() for pattern in ['<script', 'src=', 'import(', 'require(']):
                        return ContractViolation(
                            contract_type=ContractType.SECURITY,
                            contract_name="no_external_scripts",
                            severity="error",
                            description="External script loading detected in event handler",
                            location=edge.handler_fn_id,
                            recommendation="Remove external script references or add explicit validation"
                        )
        return None
    
    def _check_no_inline_eval(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for eval() or Function() usage"""
        if hasattr(visual_ir, 'interaction_graph'):
            for edge in visual_ir.interaction_graph.edges:
                handler_fn = visual_ir.interaction_graph.handler_functions.get(edge.handler_fn_id)
                if handler_fn:
                    fn_str = str(handler_fn)
                    if re.search(r'\beval\s*\(|\bFunction\s*\(', fn_str):
                        return ContractViolation(
                            contract_type=ContractType.SECURITY,
                            contract_name="no_inline_eval",
                            severity="error",
                            description="Dangerous eval() or Function() usage detected",
                            location=edge.handler_fn_id,
                            recommendation="Replace with safe alternatives"
                        )
        return None
    
    def _check_sanitized_content(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check that user content is sanitized"""
        if hasattr(visual_ir, 'ui_components'):
            for comp_id, component in visual_ir.ui_components.items():
                # Check for innerHTML or dangerous property bindings
                if 'innerHTML' in component.props or 'dangerouslySetInnerHTML' in component.props:
                    # Check if sanitization is present
                    props_str = str(component.props)
                    if 'sanitize' not in props_str.lower():
                        return ContractViolation(
                            contract_type=ContractType.SECURITY,
                            contract_name="sanitized_content",
                            severity="error",
                            description="Unsanitized HTML content detected",
                            location=comp_id,
                            recommendation="Sanitize all user-provided HTML content"
                        )
        return None
    
    def _check_csp_compliant(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for CSP compliance"""
        # Check for inline styles and scripts
        if hasattr(visual_ir, 'ui_components'):
            for comp_id, component in visual_ir.ui_components.items():
                if 'style' in component.props and isinstance(component.props['style'], dict):
                    # Inline styles found - warning level
                    return ContractViolation(
                        contract_type=ContractType.SECURITY,
                        contract_name="csp_compliant",
                        severity="warning",
                        description="Inline styles may violate strict CSP",
                        location=comp_id,
                        recommendation="Use CSS classes instead of inline styles"
                    )
        return None
    
    def _check_no_dangerous_urls(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for dangerous URL protocols"""
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:']
        
        if hasattr(visual_ir, 'ui_components'):
            for comp_id, component in visual_ir.ui_components.items():
                # Check href, src, and other URL properties
                for prop_name, prop_value in component.props.items():
                    if prop_name in ['href', 'src', 'action', 'formaction']:
                        value_str = str(prop_value).lower()
                        for protocol in dangerous_protocols:
                            if protocol in value_str:
                                return ContractViolation(
                                    contract_type=ContractType.SECURITY,
                                    contract_name="no_dangerous_urls",
                                    severity="error",
                                    description=f"Dangerous URL protocol '{protocol}' detected",
                                    location=comp_id,
                                    recommendation="Use safe protocols (https:, mailto:, etc.)"
                                )
        return None
    
    # Accessibility contract checkers
    
    def _check_wcag_aa(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check WCAG 2.1 AA compliance"""
        violations_found = []
        
        if hasattr(visual_ir, 'ui_components'):
            for comp_id, component in visual_ir.ui_components.items():
                # Check for missing alt text on images
                if component.type.value == 'image':
                    if 'alt' not in component.props and 'alt' not in component.accessibility_attrs:
                        violations_found.append(comp_id)
                
                # Check for missing labels on inputs
                if component.type.value == 'input':
                    if 'aria-label' not in component.accessibility_attrs and 'id' not in component.props:
                        violations_found.append(comp_id)
        
        if violations_found:
            return ContractViolation(
                contract_type=ContractType.ACCESSIBILITY,
                contract_name="wcag_aa_compliant",
                severity="error",
                description=f"WCAG AA violations found in {len(violations_found)} components",
                location=", ".join(violations_found[:3]),
                recommendation="Add proper ARIA labels and alt text"
            )
        return None
    
    def _check_keyboard_navigable(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check keyboard navigation support"""
        if hasattr(visual_ir, 'ui_components'):
            interactive_components = ['button', 'input', 'link', 'form']
            
            for comp_id, component in visual_ir.ui_components.items():
                if component.type.value in interactive_components:
                    # Check for tabindex or focusable attributes
                    if 'tabindex' not in component.props and 'tabindex' not in component.accessibility_attrs:
                        # Warning level - may be implicitly focusable
                        return ContractViolation(
                            contract_type=ContractType.ACCESSIBILITY,
                            contract_name="keyboard_navigable",
                            severity="warning",
                            description="Interactive component may not be keyboard accessible",
                            location=comp_id,
                            recommendation="Ensure proper tab order with tabindex"
                        )
        return None
    
    def _check_screen_reader(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check screen reader compatibility"""
        if hasattr(visual_ir, 'ui_components'):
            for comp_id, component in visual_ir.ui_components.items():
                # Check for semantic HTML or ARIA roles
                if not component.accessibility_attrs.get('role') and component.type.value == 'container':
                    # Info level - may be fine without explicit role
                    return ContractViolation(
                        contract_type=ContractType.ACCESSIBILITY,
                        contract_name="screen_reader_compatible",
                        severity="info",
                        description="Container without explicit ARIA role",
                        location=comp_id,
                        recommendation="Consider adding appropriate ARIA role"
                    )
        return None
    
    def _check_aria_labels(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for ARIA labels"""
        missing_labels = []
        
        if hasattr(visual_ir, 'ui_components'):
            interactive_types = ['button', 'input', 'link']
            
            for comp_id, component in visual_ir.ui_components.items():
                if component.type.value in interactive_types:
                    has_label = (
                        'aria-label' in component.accessibility_attrs or
                        'aria-labelledby' in component.accessibility_attrs or
                        'text' in component.props
                    )
                    
                    if not has_label:
                        missing_labels.append(comp_id)
        
        if missing_labels:
            return ContractViolation(
                contract_type=ContractType.ACCESSIBILITY,
                contract_name="has_aria_labels",
                severity="warning",
                description=f"Missing ARIA labels on {len(missing_labels)} interactive elements",
                location=", ".join(missing_labels[:3]),
                recommendation="Add aria-label or aria-labelledby attributes"
            )
        return None
    
    def _check_contrast(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check color contrast ratios"""
        # Simplified check - in production would calculate actual contrast
        if hasattr(visual_ir, 'style_graph'):
            for style_id, style_node in visual_ir.style_graph.styles.items():
                # Check if both color and background are specified
                from .visual_ir import StyleProperty
                has_color = StyleProperty.COLOR in style_node.properties
                has_background = StyleProperty.BACKGROUND in style_node.properties
                
                if has_color and has_background:
                    # In production: calculate actual contrast ratio
                    # For now, just a placeholder check
                    pass
        return None
    
    # Performance contract checkers
    
    def _check_fast_load(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check estimated load time"""
        if hasattr(visual_ir, 'performance_metrics'):
            estimated_load = visual_ir.performance_metrics.get('estimated_load_ms', 0)
            if estimated_load > 100:
                return ContractViolation(
                    contract_type=ContractType.PERFORMANCE,
                    contract_name="fast_load",
                    severity="warning",
                    description=f"Estimated load time {estimated_load}ms exceeds 100ms budget",
                    recommendation="Optimize critical path resources"
                )
        return None
    
    def _check_small_bundle(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check bundle size"""
        if hasattr(visual_ir, 'performance_metrics'):
            estimated_size = visual_ir.performance_metrics.get('estimated_bundle_kb', 0)
            if estimated_size > 50:
                return ContractViolation(
                    contract_type=ContractType.PERFORMANCE,
                    contract_name="small_bundle",
                    severity="warning",
                    description=f"Estimated bundle size {estimated_size}KB exceeds 50KB budget",
                    recommendation="Enable code splitting and tree shaking"
                )
        return None
    
    def _check_no_layout_shift(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for potential layout shifts"""
        if hasattr(visual_ir, 'ui_components'):
            for comp_id, component in visual_ir.ui_components.items():
                # Check for images without dimensions
                if component.type.value == 'image':
                    has_dimensions = 'width' in component.props and 'height' in component.props
                    if not has_dimensions:
                        return ContractViolation(
                            contract_type=ContractType.PERFORMANCE,
                            contract_name="no_layout_shift",
                            severity="warning",
                            description="Image without dimensions may cause layout shift",
                            location=comp_id,
                            recommendation="Specify width and height attributes"
                        )
        return None
    
    def _check_optimized_images(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for image optimization"""
        if hasattr(visual_ir, 'ui_components'):
            for comp_id, component in visual_ir.ui_components.items():
                if component.type.value == 'image':
                    src = component.props.get('src', '')
                    # Check for modern formats
                    if isinstance(src, str) and not any(fmt in src.lower() for fmt in ['.webp', '.avif']):
                        return ContractViolation(
                            contract_type=ContractType.PERFORMANCE,
                            contract_name="optimized_images",
                            severity="info",
                            description="Consider using modern image formats",
                            location=comp_id,
                            recommendation="Use WebP or AVIF formats for better compression"
                        )
        return None
    
    def _check_minimal_reflows(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for operations that trigger reflows"""
        # Placeholder - would need runtime analysis
        return None
    
    # Privacy contract checkers
    
    def _check_no_tracking(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for tracking scripts"""
        if hasattr(visual_ir, 'interaction_graph'):
            tracking_patterns = ['analytics', 'gtag', 'facebook', 'tracking', 'pixel']
            
            for edge in visual_ir.interaction_graph.edges:
                handler_fn = visual_ir.interaction_graph.handler_functions.get(edge.handler_fn_id)
                if handler_fn:
                    fn_str = str(handler_fn).lower()
                    for pattern in tracking_patterns:
                        if pattern in fn_str:
                            return ContractViolation(
                                contract_type=ContractType.PRIVACY,
                                contract_name="no_tracking",
                                severity="warning",
                                description=f"Potential tracking code detected: {pattern}",
                                location=edge.handler_fn_id,
                                recommendation="Remove tracking or obtain user consent"
                            )
        return None
    
    def _check_no_third_party(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for third-party embeds"""
        if hasattr(visual_ir, 'ui_components'):
            for comp_id, component in visual_ir.ui_components.items():
                # Check for iframe or embed components
                if 'iframe' in str(component.type).lower() or 'embed' in str(component.type).lower():
                    return ContractViolation(
                        contract_type=ContractType.PRIVACY,
                        contract_name="no_third_party_embeds",
                        severity="warning",
                        description="Third-party embed detected",
                        location=comp_id,
                        recommendation="Isolate embeds or use privacy-preserving alternatives"
                    )
        return None
    
    def _check_local_storage(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for local-first storage"""
        # Placeholder - would need to analyze storage API usage
        return None
    
    def _check_no_pii(self, visual_ir: Any) -> Optional[ContractViolation]:
        """Check for PII exposure"""
        # Placeholder - would need semantic analysis
        return None


# Predefined contract sets

SECURITY_CONTRACTS = [
    "no_external_scripts",
    "no_inline_eval",
    "sanitized_content",
    "csp_compliant",
    "no_dangerous_urls"
]

ACCESSIBILITY_CONTRACTS = [
    "wcag_aa_compliant",
    "keyboard_navigable",
    "screen_reader_compatible",
    "has_aria_labels",
    "sufficient_contrast"
]

PERFORMANCE_CONTRACTS = [
    "fast_load",
    "small_bundle",
    "no_layout_shift",
    "optimized_images"
]

PRIVACY_CONTRACTS = [
    "no_tracking",
    "no_third_party_embeds",
    "local_storage_only",
    "no_pii_exposure"
]

ALL_CONTRACTS = (
    SECURITY_CONTRACTS +
    ACCESSIBILITY_CONTRACTS +
    PERFORMANCE_CONTRACTS +
    PRIVACY_CONTRACTS
)


if __name__ == "__main__":
    # Example usage
    validator = VisualContractValidator()
    print("Visual Contract Validator initialized")
    print(f"Security contracts: {len(SECURITY_CONTRACTS)}")
    print(f"Accessibility contracts: {len(ACCESSIBILITY_CONTRACTS)}")
    print(f"Performance contracts: {len(PERFORMANCE_CONTRACTS)}")
    print(f"Privacy contracts: {len(PRIVACY_CONTRACTS)}")
    print(f"Total contracts: {len(ALL_CONTRACTS)}")

