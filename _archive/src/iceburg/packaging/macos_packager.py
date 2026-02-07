"""
macOS App Packager for ICEBURG
Handles app bundling, code signing, notarization, and DMG creation.
"""

import os
import subprocess
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MacOSPackager:
    """
    macOS app packager with full build/sign/notarize/DMG pipeline.
    
    Features:
    - App bundling
    - Code signing
    - Notarization
    - DMG creation
    - Entitlements management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize macOS packager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.developer_id = self.config.get("developer_id")
        self.team_id = self.config.get("team_id")
        self.apple_id = self.config.get("apple_id")
        self.app_specific_password = self.config.get("app_specific_password")
        
        # Default paths
        self.codesign_identity = self.config.get("codesign_identity", "Developer ID Application")
        self.notarization_provider = self.config.get("notarization_provider")
    
    def create_app_bundle(self, 
                         app_name: str,
                         executable_path: str,
                         output_dir: str,
                         bundle_id: str = None,
                         version: str = "1.0.0",
                         verbose: bool = False) -> str:
        """
        Create macOS app bundle.
        
        Args:
            app_name: Name of the app
            executable_path: Path to the executable
            output_dir: Output directory
            bundle_id: Bundle identifier
            version: App version
            verbose: Enable verbose output
            
        Returns:
            Path to created app bundle
        """
        if verbose:
            logger.info(f"Creating app bundle for {app_name}")
        
        # Create app bundle structure
        app_path = Path(output_dir) / f"{app_name}.app"
        contents_path = app_path / "Contents"
        macos_path = contents_path / "MacOS"
        resources_path = contents_path / "Resources"
        
        # Create directories
        macos_path.mkdir(parents=True, exist_ok=True)
        resources_path.mkdir(parents=True, exist_ok=True)
        
        # Copy executable
        executable_name = Path(executable_path).name
        target_executable = macos_path / executable_name
        self._copy_file(executable_path, str(target_executable))
        
        # Make executable
        os.chmod(target_executable, 0o755)
        
        # Create Info.plist
        info_plist_content = self._generate_info_plist(
            app_name=app_name,
            bundle_id=bundle_id or f"com.iceburg.{app_name.lower()}",
            version=version,
            executable_name=executable_name
        )
        
        info_plist_path = contents_path / "Info.plist"
        with open(info_plist_path, "w") as f:
            f.write(info_plist_content)
        
        if verbose:
            logger.info(f"Created app bundle: {app_path}")
        
        return str(app_path)
    
    def sign_app(self, 
                app_path: str,
                team_id: str = None,
                identity: str = None,
                entitlements_path: str = None,
                verbose: bool = False) -> bool:
        """
        Code sign the app.
        
        Args:
            app_path: Path to the app bundle
            team_id: Apple Developer Team ID
            identity: Code signing identity
            entitlements_path: Path to entitlements file
            verbose: Enable verbose output
            
        Returns:
            True if signing successful
        """
        try:
            if verbose:
                logger.info(f"Code signing app: {app_path}")
            
            # Build codesign command
            cmd = ["codesign", "--force", "--sign"]
            
            if identity:
                cmd.append(identity)
            elif team_id:
                cmd.append(f"Developer ID Application ({team_id})")
            else:
                cmd.append(self.codesign_identity)
            
            if entitlements_path and os.path.exists(entitlements_path):
                cmd.extend(["--entitlements", entitlements_path])
            
            cmd.extend(["--options", "runtime", app_path])
            
            # Execute codesign
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if verbose:
                    logger.info("Code signing successful")
                return True
            else:
                logger.error(f"Code signing failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Code signing error: {e}")
            return False
    
    def notarize_app(self, 
                    app_path: str,
                    team_id: str = None,
                    apple_id: str = None,
                    password: str = None,
                    verbose: bool = False) -> bool:
        """
        Notarize the app with Apple.
        
        Args:
            app_path: Path to the app bundle
            team_id: Apple Developer Team ID
            apple_id: Apple ID for notarization
            password: App-specific password
            verbose: Enable verbose output
            
        Returns:
            True if notarization successful
        """
        try:
            if verbose:
                logger.info(f"Notarizing app: {app_path}")
            
            # Use provided credentials or config defaults
            apple_id = apple_id or self.apple_id
            password = password or self.app_specific_password
            team_id = team_id or self.team_id
            
            if not apple_id or not password:
                logger.error("Apple ID and app-specific password required for notarization")
                return False
            
            # Create zip for notarization
            zip_path = f"{app_path}.zip"
            subprocess.run(["ditto", "-c", "-k", "--keepParent", app_path, zip_path], check=True)
            
            # Submit for notarization
            cmd = ["xcrun", "notarytool", "submit", zip_path]
            cmd.extend(["--apple-id", apple_id])
            cmd.extend(["--password", password])
            
            if team_id:
                cmd.extend(["--team-id", team_id])
            
            cmd.extend(["--wait"])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if verbose:
                    logger.info("Notarization successful")
                
                # Staple the notarization
                self._staple_notarization(app_path, verbose)
                
                # Clean up zip
                os.remove(zip_path)
                return True
            else:
                logger.error(f"Notarization failed: {result.stderr}")
                os.remove(zip_path)
                return False
                
        except Exception as e:
            logger.error(f"Notarization error: {e}")
            return False
    
    def create_dmg(self, 
                  app_path: str,
                  output_dir: str,
                  dmg_name: str = None,
                  verbose: bool = False) -> Optional[str]:
        """
        Create DMG for distribution.
        
        Args:
            app_path: Path to the app bundle
            output_dir: Output directory for DMG
            dmg_name: Name for the DMG file
            verbose: Enable verbose output
            
        Returns:
            Path to created DMG or None if failed
        """
        try:
            if verbose:
                logger.info(f"Creating DMG for: {app_path}")
            
            app_name = Path(app_path).stem
            dmg_name = dmg_name or f"{app_name}.dmg"
            dmg_path = Path(output_dir) / dmg_name
            
            # Create temporary directory for DMG contents
            temp_dir = Path(output_dir) / "dmg_temp"
            temp_dir.mkdir(exist_ok=True)
            
            # Copy app to temp directory
            temp_app_path = temp_dir / Path(app_path).name
            self._copy_directory(app_path, str(temp_app_path))
            
            # Create DMG
            cmd = [
                "hdiutil", "create",
                "-volname", app_name,
                "-srcfolder", str(temp_dir),
                "-ov", "-format", "UDZO",
                str(dmg_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)
            
            if result.returncode == 0:
                if verbose:
                    logger.info(f"DMG created: {dmg_path}")
                return str(dmg_path)
            else:
                logger.error(f"DMG creation failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"DMG creation error: {e}")
            return None
    
    def _generate_info_plist(self, 
                           app_name: str,
                           bundle_id: str,
                           version: str,
                           executable_name: str) -> str:
        """Generate Info.plist content."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>{executable_name}</string>
    <key>CFBundleIdentifier</key>
    <string>{bundle_id}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>{app_name}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>{version}</string>
    <key>CFBundleVersion</key>
    <string>{version}</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSRequiresAquaSystemAppearance</key>
    <false/>
</dict>
</plist>"""
    
    def _staple_notarization(self, app_path: str, verbose: bool = False):
        """Staple the notarization to the app."""
        try:
            cmd = ["xcrun", "stapler", "staple", app_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if verbose:
                    logger.info("Notarization stapled successfully")
            else:
                logger.warning(f"Stapling failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Stapling error: {e}")
    
    def _copy_file(self, src: str, dst: str):
        """Copy a file."""
        import shutil
        shutil.copy2(src, dst)
    
    def _copy_directory(self, src: str, dst: str):
        """Copy a directory."""
        import shutil
        shutil.copytree(src, dst, dirs_exist_ok=True)
    
    def verify_app(self, app_path: str, verbose: bool = False) -> bool:
        """
        Verify the app bundle.
        
        Args:
            app_path: Path to the app bundle
            verbose: Enable verbose output
            
        Returns:
            True if verification successful
        """
        try:
            if verbose:
                logger.info(f"Verifying app: {app_path}")
            
            # Check app structure
            if not os.path.exists(app_path):
                logger.error("App bundle does not exist")
                return False
            
            contents_path = Path(app_path) / "Contents"
            if not contents_path.exists():
                logger.error("Contents directory missing")
                return False
            
            info_plist = contents_path / "Info.plist"
            if not info_plist.exists():
                logger.error("Info.plist missing")
                return False
            
            # Verify code signature
            cmd = ["codesign", "--verify", "--verbose", app_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if verbose:
                    logger.info("App verification successful")
                return True
            else:
                logger.error(f"App verification failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"App verification error: {e}")
            return False
    
    def get_signing_identities(self) -> list:
        """Get available code signing identities."""
        try:
            cmd = ["security", "find-identity", "-v", "-p", "codesigning"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            identities = []
            for line in result.stdout.split('\n'):
                if 'Developer ID' in line or 'Mac Developer' in line:
                    identities.append(line.strip())
            
            return identities
            
        except Exception as e:
            logger.error(f"Error getting signing identities: {e}")
            return []
    
    def get_team_ids(self) -> list:
        """Get available Apple Developer Team IDs."""
        try:
            cmd = ["xcrun", "altool", "--list-providers", "-u", self.apple_id, "-p", self.app_specific_password]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            team_ids = []
            for line in result.stdout.split('\n'):
                if 'Team ID:' in line:
                    team_id = line.split('Team ID:')[1].strip()
                    team_ids.append(team_id)
            
            return team_ids
            
        except Exception as e:
            logger.error(f"Error getting team IDs: {e}")
            return []