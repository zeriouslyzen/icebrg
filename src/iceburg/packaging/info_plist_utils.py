"""
Info.plist and Entitlements Generation Utilities

Handles:
- Dynamic Info.plist generation for different app types
- Entitlements generation for security and capabilities
- Privacy usage descriptions
- App Store metadata
"""

import plistlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class InfoPlistGenerator:
    """Generates Info.plist files for macOS apps."""
    
    def __init__(self):
        self.base_info = {
            'CFBundlePackageType': 'APPL',
            'CFBundleSignature': '????',
            'LSMinimumSystemVersion': '11.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False
        }
    
    def generate_for_ide(self, 
                        bundle_id: str,
                        app_name: str,
                        version: str = "1.0",
                        author: str = "ICEBURG",
                        features: List[str] = None) -> Dict[str, Any]:
        """
        Generate Info.plist for IDE applications.
        
        Args:
            bundle_id: Bundle identifier (e.g., com.iceburg.ide)
            app_name: Application name
            version: Version string
            author: Author name
            features: List of IDE features
            
        Returns:
            Info.plist dictionary
        """
        features = features or []
        
        info = self.base_info.copy()
        info.update({
            'CFBundleIdentifier': bundle_id,
            'CFBundleName': app_name,
            'CFBundleDisplayName': app_name,
            'CFBundleExecutable': app_name,
            'CFBundleVersion': version,
            'CFBundleShortVersionString': version,
            'CFBundleGetInfoString': f"{app_name} {version} by {author}",
            'CFBundleInfoDictionaryVersion': '6.0',
            'NSHumanReadableCopyright': f"Copyright © {datetime.now().year} {author}. All rights reserved.",
            
            # IDE-specific capabilities
            'NSAppleEventsUsageDescription': 'This IDE needs access to Apple Events for automation and scripting.',
            'NSDocumentsFolderUsageDescription': 'This IDE needs access to your Documents folder to open and save files.',
            'NSDownloadsFolderUsageDescription': 'This IDE needs access to your Downloads folder to open downloaded files.',
            'NSDesktopFolderUsageDescription': 'This IDE needs access to your Desktop folder to open and save files.',
            
            # File types supported
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'Source Code',
                    'CFBundleTypeRole': 'Editor',
                    'CFBundleTypeExtensions': ['swift', 'py', 'js', 'ts', 'html', 'css', 'json', 'xml', 'md', 'txt'],
                    'CFBundleTypeIconFile': 'DocumentIcon'
                }
            ],
            
            # URL schemes
            'CFBundleURLTypes': [
                {
                    'CFBundleURLName': f"{bundle_id}.file",
                    'CFBundleURLSchemes': ['file']
                }
            ],
            
            # Supported file operations
            'LSSupportsOpeningDocumentsInPlace': True,
            'UISupportsDocumentBrowser': True,
            
            # IDE-specific settings
            'NSAppTransportSecurity': {
                'NSAllowsArbitraryLoads': True,  # For loading external resources
                'NSExceptionDomains': {
                    'os.getenv("HOST", "localhost")': {
                        'NSExceptionAllowsInsecureHTTPLoads': True
                    }
                }
            }
        })
        
        # Add Monaco editor specific settings
        if 'monaco-editor' in features:
            info.update({
                'NSWebKitUsageDescription': 'This IDE uses WebKit to display the Monaco code editor.',
                'NSJavaScriptUsageDescription': 'This IDE uses JavaScript for the Monaco editor functionality.'
            })
        
        # Add terminal specific settings
        if 'terminal' in features:
            info.update({
                'NSTerminalUsageDescription': 'This IDE includes a terminal for command-line operations.',
                'NSProcessUsageDescription': 'This IDE needs to launch terminal processes for development tools.'
            })
        
        # Add file explorer specific settings
        if 'file-explorer' in features:
            info.update({
                'NSFileProviderUsageDescription': 'This IDE needs access to your files to provide a file explorer.',
                'NSDocumentsFolderUsageDescription': 'This IDE needs access to your Documents folder for the file explorer.'
            })
        
        return info
    
    def generate_for_calculator(self,
                               bundle_id: str,
                               app_name: str,
                               version: str = "1.0",
                               author: str = "ICEBURG") -> Dict[str, Any]:
        """Generate Info.plist for calculator applications."""
        info = self.base_info.copy()
        info.update({
            'CFBundleIdentifier': bundle_id,
            'CFBundleName': app_name,
            'CFBundleDisplayName': app_name,
            'CFBundleExecutable': app_name,
            'CFBundleVersion': version,
            'CFBundleShortVersionString': version,
            'CFBundleGetInfoString': f"{app_name} {version} by {author}",
            'CFBundleInfoDictionaryVersion': '6.0',
            'NSHumanReadableCopyright': f"Copyright © {datetime.now().year} {author}. All rights reserved.",
            
            # Calculator-specific settings
            'NSAppTransportSecurity': {
                'NSAllowsArbitraryLoads': False
            }
        })
        
        return info
    
    def generate_for_game(self,
                          bundle_id: str,
                          app_name: str,
                          version: str = "1.0",
                          author: str = "ICEBURG",
                          game_type: str = "2d") -> Dict[str, Any]:
        """Generate Info.plist for game applications."""
        info = self.base_info.copy()
        info.update({
            'CFBundleIdentifier': bundle_id,
            'CFBundleName': app_name,
            'CFBundleDisplayName': app_name,
            'CFBundleExecutable': app_name,
            'CFBundleVersion': version,
            'CFBundleShortVersionString': version,
            'CFBundleGetInfoString': f"{app_name} {version} by {author}",
            'CFBundleInfoDictionaryVersion': '6.0',
            'NSHumanReadableCopyright': f"Copyright © {datetime.now().year} {author}. All rights reserved.",
            
            # Game-specific settings
            'NSAppTransportSecurity': {
                'NSAllowsArbitraryLoads': True  # For game assets
            }
        })
        
        # Add game-specific capabilities
        if game_type == "3d":
            info.update({
                'NSOpenGLUsageDescription': 'This game uses OpenGL for 3D graphics rendering.',
                'NSMetalUsageDescription': 'This game uses Metal for high-performance graphics rendering.'
            })
        
        return info
    
    def generate_for_web_app(self,
                             bundle_id: str,
                             app_name: str,
                             version: str = "1.0",
                             author: str = "ICEBURG") -> Dict[str, Any]:
        """Generate Info.plist for web applications."""
        info = self.base_info.copy()
        info.update({
            'CFBundleIdentifier': bundle_id,
            'CFBundleName': app_name,
            'CFBundleDisplayName': app_name,
            'CFBundleExecutable': app_name,
            'CFBundleVersion': version,
            'CFBundleShortVersionString': version,
            'CFBundleGetInfoString': f"{app_name} {version} by {author}",
            'CFBundleInfoDictionaryVersion': '6.0',
            'NSHumanReadableCopyright': f"Copyright © {datetime.now().year} {author}. All rights reserved.",
            
            # Web app specific settings
            'NSAppTransportSecurity': {
                'NSAllowsArbitraryLoads': True,
                'NSExceptionDomains': {
                    'os.getenv("HOST", "localhost")': {
                        'NSExceptionAllowsInsecureHTTPLoads': True
                    }
                }
            },
            'NSWebKitUsageDescription': 'This web app uses WebKit to display web content.',
            'NSJavaScriptUsageDescription': 'This web app uses JavaScript for interactive functionality.'
        })
        
        return info
    
    def generate_custom(self,
                       bundle_id: str,
                       app_name: str,
                       version: str = "1.0",
                       author: str = "ICEBURG",
                       custom_properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate Info.plist with custom properties.
        
        Args:
            bundle_id: Bundle identifier
            app_name: Application name
            version: Version string
            author: Author name
            custom_properties: Custom properties to add
            
        Returns:
            Info.plist dictionary
        """
        info = self.base_info.copy()
        info.update({
            'CFBundleIdentifier': bundle_id,
            'CFBundleName': app_name,
            'CFBundleDisplayName': app_name,
            'CFBundleExecutable': app_name,
            'CFBundleVersion': version,
            'CFBundleShortVersionString': version,
            'CFBundleGetInfoString': f"{app_name} {version} by {author}",
            'CFBundleInfoDictionaryVersion': '6.0',
            'NSHumanReadableCopyright': f"Copyright © {datetime.now().year} {author}. All rights reserved."
        })
        
        if custom_properties:
            info.update(custom_properties)
        
        return info


class EntitlementsGenerator:
    """Generates entitlements files for macOS apps."""
    
    def __init__(self):
        self.base_entitlements = {
            'com.apple.security.app-sandbox': True,
            'com.apple.security.files.user-selected.read-write': True,
            'com.apple.security.files.downloads.read-write': True
        }
    
    def generate_for_ide(self, features: List[str] = None) -> Dict[str, Any]:
        """
        Generate entitlements for IDE applications.
        
        Args:
            features: List of IDE features
            
        Returns:
            Entitlements dictionary
        """
        features = features or []
        entitlements = self.base_entitlements.copy()
        
        # IDE-specific entitlements
        entitlements.update({
            'com.apple.security.files.user-selected.read-write': True,
            'com.apple.security.files.downloads.read-write': True,
            'com.apple.security.files.bookmarks.app-scope': True,
            'com.apple.security.network.client': True,
            'com.apple.security.network.server': True,
            'com.apple.security.automation.apple-events': True,
            'com.apple.security.temporary-exception.files.absolute-path.read-write': [
                '/usr/local/bin',
                '/opt/homebrew/bin',
                '/usr/bin'
            ]
        })
        
        # Terminal-specific entitlements
        if 'terminal' in features:
            entitlements.update({
                'com.apple.security.temporary-exception.apple-events': True,
                'com.apple.security.temporary-exception.files.absolute-path.read-write': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',
                    '/usr/bin',
                    '/bin',
                    '/sbin'
                ]
            })
        
        # Network access for Monaco editor
        if 'monaco-editor' in features:
            entitlements.update({
                'com.apple.security.network.client': True,
                'com.apple.security.network.server': True
            })
        
        return entitlements
    
    def generate_for_calculator(self) -> Dict[str, Any]:
        """Generate entitlements for calculator applications."""
        entitlements = self.base_entitlements.copy()
        entitlements.update({
            'com.apple.security.app-sandbox': True,
            'com.apple.security.files.user-selected.read-write': False,
            'com.apple.security.files.downloads.read-write': False,
            'com.apple.security.network.client': False,
            'com.apple.security.network.server': False
        })
        
        return entitlements
    
    def generate_for_game(self, game_type: str = "2d") -> Dict[str, Any]:
        """Generate entitlements for game applications."""
        entitlements = self.base_entitlements.copy()
        entitlements.update({
            'com.apple.security.app-sandbox': True,
            'com.apple.security.files.user-selected.read-write': True,
            'com.apple.security.files.downloads.read-write': True,
            'com.apple.security.network.client': True
        })
        
        # 3D games may need additional entitlements
        if game_type == "3d":
            entitlements.update({
                'com.apple.security.device.audio-input': True,
                'com.apple.security.device.camera': True
            })
        
        return entitlements
    
    def generate_for_web_app(self) -> Dict[str, Any]:
        """Generate entitlements for web applications."""
        entitlements = self.base_entitlements.copy()
        entitlements.update({
            'com.apple.security.app-sandbox': True,
            'com.apple.security.files.user-selected.read-write': True,
            'com.apple.security.files.downloads.read-write': True,
            'com.apple.security.network.client': True,
            'com.apple.security.network.server': True,
            'com.apple.security.automation.apple-events': True
        })
        
        return entitlements
    
    def generate_custom(self, custom_entitlements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate entitlements with custom properties.
        
        Args:
            custom_entitlements: Custom entitlements to add
            
        Returns:
            Entitlements dictionary
        """
        entitlements = self.base_entitlements.copy()
        
        if custom_entitlements:
            entitlements.update(custom_entitlements)
        
        return entitlements


class PrivacyUsageGenerator:
    """Generates privacy usage descriptions for Info.plist."""
    
    @staticmethod
    def get_usage_descriptions(features: List[str]) -> Dict[str, str]:
        """
        Get privacy usage descriptions based on app features.
        
        Args:
            features: List of app features
            
        Returns:
            Dictionary of privacy usage descriptions
        """
        descriptions = {}
        
        feature_descriptions = {
            'monaco-editor': {
                'NSWebKitUsageDescription': 'This app uses WebKit to display the Monaco code editor.',
                'NSJavaScriptUsageDescription': 'This app uses JavaScript for the Monaco editor functionality.'
            },
            'terminal': {
                'NSTerminalUsageDescription': 'This app includes a terminal for command-line operations.',
                'NSProcessUsageDescription': 'This app needs to launch terminal processes for development tools.'
            },
            'file-explorer': {
                'NSFileProviderUsageDescription': 'This app needs access to your files to provide a file explorer.',
                'NSDocumentsFolderUsageDescription': 'This app needs access to your Documents folder for the file explorer.'
            },
            'camera': {
                'NSCameraUsageDescription': 'This app needs access to your camera for video features.'
            },
            'microphone': {
                'NSMicrophoneUsageDescription': 'This app needs access to your microphone for voice features.'
            },
            'location': {
                'NSLocationUsageDescription': 'This app needs access to your location for location-based features.'
            },
            'contacts': {
                'NSContactsUsageDescription': 'This app needs access to your contacts for contact-based features.'
            },
            'calendar': {
                'NSCalendarsUsageDescription': 'This app needs access to your calendar for scheduling features.'
            }
        }
        
        for feature in features:
            if feature in feature_descriptions:
                descriptions.update(feature_descriptions[feature])
        
        return descriptions
