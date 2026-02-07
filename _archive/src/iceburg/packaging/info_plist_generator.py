"""
Info.plist Generator for ICEBURG macOS Apps
Generates proper Info.plist files for macOS app bundles.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class InfoPlistGenerator:
    """
    Generates Info.plist files for macOS app bundles.
    
    Features:
    - Standard macOS app properties
    - Custom bundle configuration
    - Security entitlements
    - Version management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Info.plist generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_properties = {
            "CFBundleDevelopmentRegion": "en",
            "CFBundleInfoDictionaryVersion": "6.0",
            "CFBundlePackageType": "APPL",
            "LSMinimumSystemVersion": "10.15",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False
        }
    
    def generate_info_plist(self,
                          bundle_id: str,
                          app_name: str,
                          version: str,
                          output_path: str,
                          executable_name: str = None,
                          additional_properties: Dict[str, Any] = None,
                          verbose: bool = False) -> bool:
        """
        Generate Info.plist file.
        
        Args:
            bundle_id: Bundle identifier
            app_name: Application name
            version: Application version
            output_path: Output file path
            executable_name: Executable name (defaults to app_name)
            additional_properties: Additional properties to include
            verbose: Enable verbose output
            
        Returns:
            True if generation successful
        """
        try:
            if verbose:
                logger.info(f"Generating Info.plist for {app_name}")
            
            # Set executable name
            if not executable_name:
                executable_name = app_name.replace(" ", "")
            
            # Build properties
            properties = self.default_properties.copy()
            properties.update({
                "CFBundleExecutable": executable_name,
                "CFBundleIdentifier": bundle_id,
                "CFBundleName": app_name,
                "CFBundleShortVersionString": version,
                "CFBundleVersion": version
            })
            
            # Add additional properties
            if additional_properties:
                properties.update(additional_properties)
            
            # Generate XML content
            xml_content = self._generate_xml_content(properties)
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(output_path, "w") as f:
                f.write(xml_content)
            
            if verbose:
                logger.info(f"Info.plist generated: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Info.plist generation error: {e}")
            return False
    
    def generate_for_swift_app(self,
                              bundle_id: str,
                              app_name: str,
                              version: str,
                              output_path: str,
                              swift_package_name: str = None,
                              verbose: bool = False) -> bool:
        """
        Generate Info.plist for Swift app.
        
        Args:
            bundle_id: Bundle identifier
            app_name: Application name
            version: Application version
            output_path: Output file path
            swift_package_name: Swift package name
            verbose: Enable verbose output
            
        Returns:
            True if generation successful
        """
        additional_properties = {
            "CFBundleExecutable": swift_package_name or app_name.replace(" ", ""),
            "LSApplicationCategoryType": "public.app-category.developer-tools",
            "NSHumanReadableCopyright": "Copyright © 2024 ICEBURG. All rights reserved."
        }
        
        return self.generate_info_plist(
            bundle_id=bundle_id,
            app_name=app_name,
            version=version,
            output_path=output_path,
            additional_properties=additional_properties,
            verbose=verbose
        )
    
    def generate_for_ide_app(self,
                            bundle_id: str,
                            app_name: str,
                            version: str,
                            output_path: str,
                            verbose: bool = False) -> bool:
        """
        Generate Info.plist for IDE-like app.
        
        Args:
            bundle_id: Bundle identifier
            app_name: Application name
            version: Application version
            output_path: Output file path
            verbose: Enable verbose output
            
        Returns:
            True if generation successful
        """
        additional_properties = {
            "CFBundleExecutable": app_name.replace(" ", ""),
            "LSApplicationCategoryType": "public.app-category.developer-tools",
            "NSHumanReadableCopyright": "Copyright © 2024 ICEBURG. All rights reserved.",
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "Source Code",
                    "CFBundleTypeRole": "Editor",
                    "CFBundleTypeExtensions": ["swift", "py", "js", "html", "css", "json", "md", "txt"],
                    "CFBundleTypeIconFile": "DocumentIcon"
                }
            ],
            "CFBundleURLTypes": [
                {
                    "CFBundleURLName": "File URL",
                    "CFBundleURLSchemes": ["file"]
                }
            ],
            "NSAppleScriptEnabled": True,
            "NSAppleEventsUsageDescription": "This app can execute AppleScript for automation."
        }
        
        return self.generate_info_plist(
            bundle_id=bundle_id,
            app_name=app_name,
            version=version,
            output_path=output_path,
            additional_properties=additional_properties,
            verbose=verbose
        )
    
    def _generate_xml_content(self, properties: Dict[str, Any]) -> str:
        """Generate XML content from properties."""
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
            '<plist version="1.0">',
            '<dict>'
        ]
        
        for key, value in properties.items():
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
                lines.append(f"{indent}    <key>{item}</key>")
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
    
    def validate_bundle_id(self, bundle_id: str) -> bool:
        """
        Validate bundle identifier format.
        
        Args:
            bundle_id: Bundle identifier to validate
            
        Returns:
            True if valid
        """
        if not bundle_id:
            return False
        
        # Basic validation: should be reverse domain format
        parts = bundle_id.split('.')
        if len(parts) < 2:
            return False
        
        # Check each part
        for part in parts:
            if not part or not part.replace('-', '').replace('_', '').isalnum():
                return False
        
        return True
    
    def suggest_bundle_id(self, app_name: str, company: str = "iceburg") -> str:
        """
        Suggest a bundle identifier.
        
        Args:
            app_name: Application name
            company: Company name
            
        Returns:
            Suggested bundle identifier
        """
        # Clean app name
        clean_name = app_name.lower().replace(" ", "").replace("-", "")
        
        return f"com.{company}.{clean_name}"
