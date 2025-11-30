"""
Entitlements Generator for ICEBURG macOS Apps
Generates entitlements.plist files for macOS app security.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class EntitlementsGenerator:
    """
    Generates entitlements.plist files for macOS app security.
    
    Features:
    - Standard macOS entitlements
    - Security permissions
    - Sandbox configuration
    - Hardened runtime
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize entitlements generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_entitlements = {
            "com.apple.security.app-sandbox": True,
            "com.apple.security.files.user-selected.read-only": True,
            "com.apple.security.files.user-selected.read-write": True,
            "com.apple.security.network.client": True,
            "com.apple.security.network.server": False
        }
    
    def generate_entitlements(self,
                            bundle_id: str,
                            output_path: str,
                            app_type: str = "standard",
                            additional_entitlements: Dict[str, Any] = None,
                            verbose: bool = False) -> bool:
        """
        Generate entitlements.plist file.
        
        Args:
            bundle_id: Bundle identifier
            output_path: Output file path
            app_type: Type of app (standard, ide, network, system)
            additional_entitlements: Additional entitlements to include
            verbose: Enable verbose output
            
        Returns:
            True if generation successful
        """
        try:
            if verbose:
                logger.info(f"Generating entitlements for {bundle_id}")
            
            # Get entitlements based on app type
            entitlements = self._get_entitlements_for_type(app_type)
            
            # Add additional entitlements
            if additional_entitlements:
                entitlements.update(additional_entitlements)
            
            # Generate XML content
            xml_content = self._generate_xml_content(entitlements)
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(output_path, "w") as f:
                f.write(xml_content)
            
            if verbose:
                logger.info(f"Entitlements generated: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Entitlements generation error: {e}")
            return False
    
    def generate_for_ide_app(self,
                           bundle_id: str,
                           output_path: str,
                           verbose: bool = False) -> bool:
        """
        Generate entitlements for IDE-like app.
        
        Args:
            bundle_id: Bundle identifier
            output_path: Output file path
            verbose: Enable verbose output
            
        Returns:
            True if generation successful
        """
        ide_entitlements = {
            "com.apple.security.app-sandbox": True,
            "com.apple.security.files.user-selected.read-only": True,
            "com.apple.security.files.user-selected.read-write": True,
            "com.apple.security.files.downloads.read-write": True,
            "com.apple.security.network.client": True,
            "com.apple.security.network.server": False,
            "com.apple.security.automation.apple-events": True,
            "com.apple.security.temporary-exception.files.absolute-path.read-write": [
                "/usr/local/bin",
                "/opt/homebrew/bin",
                "/usr/bin"
            ]
        }
        
        return self.generate_entitlements(
            bundle_id=bundle_id,
            output_path=output_path,
            app_type="ide",
            additional_entitlements=ide_entitlements,
            verbose=verbose
        )
    
    def generate_for_network_app(self,
                               bundle_id: str,
                               output_path: str,
                               verbose: bool = False) -> bool:
        """
        Generate entitlements for network app.
        
        Args:
            bundle_id: Bundle identifier
            output_path: Output file path
            verbose: Enable verbose output
            
        Returns:
            True if generation successful
        """
        network_entitlements = {
            "com.apple.security.app-sandbox": True,
            "com.apple.security.network.client": True,
            "com.apple.security.network.server": True,
            "com.apple.security.files.user-selected.read-only": True,
            "com.apple.security.files.user-selected.read-write": True
        }
        
        return self.generate_entitlements(
            bundle_id=bundle_id,
            output_path=output_path,
            app_type="network",
            additional_entitlements=network_entitlements,
            verbose=verbose
        )
    
    def generate_for_system_app(self,
                              bundle_id: str,
                              output_path: str,
                              verbose: bool = False) -> bool:
        """
        Generate entitlements for system app.
        
        Args:
            bundle_id: Bundle identifier
            output_path: Output file path
            verbose: Enable verbose output
            
        Returns:
            True if generation successful
        """
        system_entitlements = {
            "com.apple.security.app-sandbox": False,
            "com.apple.security.network.client": True,
            "com.apple.security.network.server": True,
            "com.apple.security.files.user-selected.read-only": True,
            "com.apple.security.files.user-selected.read-write": True,
            "com.apple.security.automation.apple-events": True,
            "com.apple.security.temporary-exception.files.absolute-path.read-write": [
                "/usr/local/bin",
                "/opt/homebrew/bin",
                "/usr/bin",
                "/System/Library/Frameworks",
                "/System/Library/PrivateFrameworks"
            ]
        }
        
        return self.generate_entitlements(
            bundle_id=bundle_id,
            output_path=output_path,
            app_type="system",
            additional_entitlements=system_entitlements,
            verbose=verbose
        )
    
    def _get_entitlements_for_type(self, app_type: str) -> Dict[str, Any]:
        """Get entitlements based on app type."""
        entitlements = self.default_entitlements.copy()
        
        if app_type == "ide":
            entitlements.update({
                "com.apple.security.automation.apple-events": True,
                "com.apple.security.temporary-exception.files.absolute-path.read-write": [
                    "/usr/local/bin",
                    "/opt/homebrew/bin"
                ]
            })
        elif app_type == "network":
            entitlements.update({
                "com.apple.security.network.server": True
            })
        elif app_type == "system":
            entitlements.update({
                "com.apple.security.app-sandbox": False,
                "com.apple.security.automation.apple-events": True
            })
        
        return entitlements
    
    def _generate_xml_content(self, entitlements: Dict[str, Any]) -> str:
        """Generate XML content from entitlements."""
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
            '<plist version="1.0">',
            '<dict>'
        ]
        
        for key, value in entitlements.items():
            xml_lines.append(f'    <key>{key}</key>')
            xml_lines.append(self._value_to_xml(value, indent="    "))
        
        xml_lines.extend(['</dict>', '</plist>'])
        
        return '\n'.join(xml_lines)
    
    def _value_to_xml(self, value: Any, indent: str = "") -> str:
        """Convert value to XML format."""
        if isinstance(value, bool):
            return f"{indent}<{'true' if value else 'false'}/>"
        elif isinstance(value, (int, float)):
            return f"{indent}<{type(value).__name__}>{value}</{type(value).__name__}>"
        elif isinstance(value, str):
            return f"{indent}<string>{value}</string>"
        elif isinstance(value, list):
            lines = [f"{indent}<array>"]
            for item in value:
                if isinstance(item, str):
                    lines.append(f"{indent}    <string>{item}</string>")
                else:
                    lines.append(self._value_to_xml(item, f"{indent}    "))
            lines.append(f"{indent}</array>")
            return '\n'.join(lines)
        elif isinstance(value, dict):
            lines = [f"{indent}<dict>"]
            for k, v in value.items():
                lines.append(f"{indent}    <key>{k}</key>")
                lines.append(self._value_to_xml(v, f"{indent}    "))
            lines.append(f"{indent}</dict>")
            return '\n'.join(lines)
        else:
            return f"{indent}<string>{str(value)}</string>"
    
    def validate_entitlements(self, entitlements_path: str) -> bool:
        """
        Validate entitlements file.
        
        Args:
            entitlements_path: Path to entitlements file
            
        Returns:
            True if valid
        """
        try:
            if not os.path.exists(entitlements_path):
                return False
            
            # Basic XML validation
            import xml.etree.ElementTree as ET
            tree = ET.parse(entitlements_path)
            root = tree.getroot()
            
            # Check if it's a valid plist
            if root.tag != "plist":
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Entitlements validation error: {e}")
            return False
    
    def get_required_entitlements(self, app_features: List[str]) -> Dict[str, Any]:
        """
        Get required entitlements based on app features.
        
        Args:
            app_features: List of app features
            
        Returns:
            Required entitlements
        """
        entitlements = {}
        
        if "file_access" in app_features:
            entitlements.update({
                "com.apple.security.files.user-selected.read-only": True,
                "com.apple.security.files.user-selected.read-write": True
            })
        
        if "network" in app_features:
            entitlements.update({
                "com.apple.security.network.client": True
            })
        
        if "automation" in app_features:
            entitlements.update({
                "com.apple.security.automation.apple-events": True
            })
        
        if "system_access" in app_features:
            entitlements.update({
                "com.apple.security.app-sandbox": False,
                "com.apple.security.temporary-exception.files.absolute-path.read-write": [
                    "/usr/local/bin",
                    "/opt/homebrew/bin"
                ]
            })
        
        return entitlements
