"""
Visual Generation Deployment System
Handles deployment of generated UIs to various platforms
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pathlib import Path
import json
import subprocess
import shutil
import tempfile
import zipfile
from datetime import datetime

from ..iir.backends import BackendType
from ..storage.visual_storage import VisualArtifactStorage
from ..build.packaging.macos import create_app_bundle
from ..packaging.macos_packager import MacOSPackager
from ..packaging.info_plist_utils import InfoPlistGenerator, EntitlementsGenerator


class DeploymentTarget(Enum):
    """Deployment targets for generated UIs"""
    LOCAL_SERVER = "local_server"
    GITHUB_PAGES = "github_pages"
    NETLIFY = "netlify"
    VERCEL = "vercel"
    HEROKU = "heroku"
    AWS_S3 = "aws_s3"
    DOCKER = "docker"
    MACOS_APP = "macos_app"
    IOS_APP = "ios_app"
    ANDROID_APP = "android_app"


@dataclass
class DeploymentConfig:
    """Configuration for deployment"""
    target: DeploymentTarget
    artifact_id: str
    project_id: str = "default"
    custom_domain: Optional[str] = None
    environment_vars: Dict[str, str] = None
    build_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.environment_vars is None:
            self.environment_vars = {}
        if self.build_settings is None:
            self.build_settings = {}


@dataclass
class DeploymentResult:
    """Result of deployment operation"""
    success: bool
    deployment_url: Optional[str] = None
    deployment_id: Optional[str] = None
    logs: List[str] = None
    error: Optional[str] = None
    deployment_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.deployment_time is None:
            self.deployment_time = datetime.now()


class VisualDeployer:
    """Deploys generated visual artifacts to various platforms"""
    
    def __init__(self):
        self.storage = VisualArtifactStorage()
        self.deployment_handlers = {
            DeploymentTarget.LOCAL_SERVER: self._deploy_local_server,
            DeploymentTarget.GITHUB_PAGES: self._deploy_github_pages,
            DeploymentTarget.NETLIFY: self._deploy_netlify,
            DeploymentTarget.VERCEL: self._deploy_vercel,
            DeploymentTarget.DOCKER: self._deploy_docker,
            DeploymentTarget.MACOS_APP: self._deploy_macos_app,
        }
    
    def deploy(self, config: DeploymentConfig, verbose: bool = False) -> DeploymentResult:
        """Deploy visual artifacts to specified target"""
        try:
            if verbose:
                print(f"[DEPLOYER] Starting deployment to {config.target.value}")
            
            # Load artifacts
            artifacts = self.storage.load_artifact(config.artifact_id, config.project_id)
            if not artifacts:
                return DeploymentResult(
                    success=False,
                    error=f"Artifact {config.artifact_id} not found"
                )
            
            # Get deployment handler
            handler = self.deployment_handlers.get(config.target)
            if not handler:
                return DeploymentResult(
                    success=False,
                    error=f"Deployment target {config.target.value} not supported"
                )
            
            # Deploy
            result = handler(config, artifacts, verbose)
            
            if verbose:
                print(f"[DEPLOYER] Deployment {'successful' if result.success else 'failed'}")
                if result.deployment_url:
                    print(f"[DEPLOYER] URL: {result.deployment_url}")
            
            return result
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=f"Deployment error: {str(e)}"
            )
    
    def _deploy_local_server(self, config: DeploymentConfig, artifacts: Dict[str, Any], verbose: bool) -> DeploymentResult:
        """Deploy to local development server"""
        try:
            # Create deployment directory
            deploy_dir = Path(f"./deployments/{config.artifact_id}")
            deploy_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy HTML5 artifacts
            if "html5" in artifacts.get("backends", {}):
                html5_path = artifacts["backends"]["html5"]
                if Path(html5_path).exists():
                    shutil.copytree(html5_path, deploy_dir / "html5", dirs_exist_ok=True)
            
            # Create simple server script
            server_script = f"""#!/usr/bin/env python3
import http.server
import socketserver
import os
from pathlib import Path

PORT = {config.build_settings.get('port', 8000)}
DIRECTORY = Path(__file__).parent / "html5"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"Server running at http://os.getenv("HOST", "localhost"):{{PORT}}")
        httpd.serve_forever()
"""
            
            (deploy_dir / "server.py").write_text(server_script)
            (deploy_dir / "server.py").chmod(0o755)
            
            # Start server in background
            port = config.build_settings.get('port', 8000)
            subprocess.Popen([
                "python3", str(deploy_dir / "server.py")
            ], cwd=deploy_dir)
            
            return DeploymentResult(
                success=True,
                deployment_url=f"http://os.getenv("HOST", "localhost"):{port}",
                deployment_id=f"local_{config.artifact_id}",
                logs=[f"Local server started on port {port}"]
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=f"Local server deployment failed: {str(e)}"
            )
    
    def _deploy_github_pages(self, config: DeploymentConfig, artifacts: Dict[str, Any], verbose: bool) -> DeploymentResult:
        """Deploy to GitHub Pages"""
        try:
            # Create temporary directory for GitHub Pages
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy HTML5 artifacts
                if "html5" in artifacts.get("backends", {}):
                    html5_path = artifacts["backends"]["html5"]
                    if Path(html5_path).exists():
                        shutil.copytree(html5_path, temp_path / "html5")
                
                # Create GitHub Pages configuration
                pages_config = {
                    "name": f"icberg-ui-{config.artifact_id}",
                    "version": "1.0.0",
                    "description": "ICEBURG Generated UI",
                    "scripts": {
                        "build": "echo 'No build step required'",
                        "deploy": "gh-pages -d html5"
                    },
                    "devDependencies": {
                        "gh-pages": "^4.0.0"
                    }
                }
                
                (temp_path / "package.json").write_text(json.dumps(pages_config, indent=2))
                
                # Create deployment script
                deploy_script = """#!/bin/bash
# Install gh-pages if not present
npm install -g gh-pages

# Deploy to GitHub Pages
npx gh-pages -d html5 -b gh-pages

echo "Deployed to GitHub Pages"
"""
                
                (temp_path / "deploy.sh").write_text(deploy_script)
                (temp_path / "deploy.sh").chmod(0o755)
                
                # Execute deployment
                result = subprocess.run(
                    ["bash", str(temp_path / "deploy.sh")],
                    capture_output=True,
                    text=True,
                    cwd=temp_path
                )
                
                if result.returncode == 0:
                    # GitHub Pages URL (would need repository info)
                    github_url = f"https://username.github.io/repository-name"  # Placeholder
                    
                    return DeploymentResult(
                        success=True,
                        deployment_url=github_url,
                        deployment_id=f"github_{config.artifact_id}",
                        logs=result.stdout.split('\n')
                    )
                else:
                    return DeploymentResult(
                        success=False,
                        error=f"GitHub Pages deployment failed: {result.stderr}"
                    )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=f"GitHub Pages deployment failed: {str(e)}"
            )
    
    def _deploy_netlify(self, config: DeploymentConfig, artifacts: Dict[str, Any], verbose: bool) -> DeploymentResult:
        """Deploy to Netlify"""
        try:
            # Create Netlify configuration
            netlify_config = {
                "build": {
                    "publish": "html5",
                    "command": "echo 'No build required'"
                },
                "redirects": [
                    {
                        "from": "/*",
                        "to": "/index.html",
                        "status": 200
                    }
                ]
            }
            
            # Create deployment package
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy HTML5 artifacts
                if "html5" in artifacts.get("backends", {}):
                    html5_path = artifacts["backends"]["html5"]
                    if Path(html5_path).exists():
                        shutil.copytree(html5_path, temp_path / "html5")
                
                # Add Netlify configuration
                (temp_path / "netlify.toml").write_text(f"""
[build]
  publish = "html5"
  command = "echo 'No build required'"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
""")
                
                # Create deployment script
                deploy_script = f"""#!/bin/bash
# Install Netlify CLI if not present
npm install -g netlify-cli

# Deploy to Netlify
netlify deploy --prod --dir=html5

echo "Deployed to Netlify"
"""
                
                (temp_path / "deploy.sh").write_text(deploy_script)
                (temp_path / "deploy.sh").chmod(0o755)
                
                # Execute deployment
                result = subprocess.run(
                    ["bash", str(temp_path / "deploy.sh")],
                    capture_output=True,
                    text=True,
                    cwd=temp_path
                )
                
                if result.returncode == 0:
                    # Extract URL from Netlify output
                    netlify_url = "https://your-site.netlify.app"  # Would extract from output
                    
                    return DeploymentResult(
                        success=True,
                        deployment_url=netlify_url,
                        deployment_id=f"netlify_{config.artifact_id}",
                        logs=result.stdout.split('\n')
                    )
                else:
                    return DeploymentResult(
                        success=False,
                        error=f"Netlify deployment failed: {result.stderr}"
                    )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=f"Netlify deployment failed: {str(e)}"
            )
    
    def _deploy_vercel(self, config: DeploymentConfig, artifacts: Dict[str, Any], verbose: bool) -> DeploymentResult:
        """Deploy to Vercel"""
        try:
            # Create Vercel configuration
            vercel_config = {
                "version": 2,
                "builds": [
                    {
                        "src": "html5/**/*",
                        "use": "@vercel/static"
                    }
                ],
                "routes": [
                    {
                        "src": "/(.*)",
                        "dest": "/html5/$1"
                    }
                ]
            }
            
            # Create deployment package
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy HTML5 artifacts
                if "html5" in artifacts.get("backends", {}):
                    html5_path = artifacts["backends"]["html5"]
                    if Path(html5_path).exists():
                        shutil.copytree(html5_path, temp_path / "html5")
                
                # Add Vercel configuration
                (temp_path / "vercel.json").write_text(json.dumps(vercel_config, indent=2))
                
                # Create deployment script
                deploy_script = """#!/bin/bash
# Install Vercel CLI if not present
npm install -g vercel

# Deploy to Vercel
vercel --prod

echo "Deployed to Vercel"
"""
                
                (temp_path / "deploy.sh").write_text(deploy_script)
                (temp_path / "deploy.sh").chmod(0o755)
                
                # Execute deployment
                result = subprocess.run(
                    ["bash", str(temp_path / "deploy.sh")],
                    capture_output=True,
                    text=True,
                    cwd=temp_path
                )
                
                if result.returncode == 0:
                    # Extract URL from Vercel output
                    vercel_url = "https://your-site.vercel.app"  # Would extract from output
                    
                    return DeploymentResult(
                        success=True,
                        deployment_url=vercel_url,
                        deployment_id=f"vercel_{config.artifact_id}",
                        logs=result.stdout.split('\n')
                    )
                else:
                    return DeploymentResult(
                        success=False,
                        error=f"Vercel deployment failed: {result.stderr}"
                    )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=f"Vercel deployment failed: {str(e)}"
            )
    
    def _deploy_docker(self, config: DeploymentConfig, artifacts: Dict[str, Any], verbose: bool) -> DeploymentResult:
        """Deploy as Docker container"""
        try:
            # Create Dockerfile
            dockerfile = """FROM nginx:alpine

# Copy HTML5 artifacts
COPY html5/ /usr/share/nginx/html/

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
"""
            
            # Create nginx configuration
            nginx_config = """events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    server {
        listen 80;
        server_name os.getenv("HOST", "localhost");
        root /usr/share/nginx/html;
        index index.html;
        
        location / {
            try_files $uri $uri/ /index.html;
        }
    }
}
"""
            
            # Create deployment package
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy HTML5 artifacts
                if "html5" in artifacts.get("backends", {}):
                    html5_path = artifacts["backends"]["html5"]
                    if Path(html5_path).exists():
                        shutil.copytree(html5_path, temp_path / "html5")
                
                # Add Docker files
                (temp_path / "Dockerfile").write_text(dockerfile)
                (temp_path / "nginx.conf").write_text(nginx_config)
                
                # Build Docker image
                image_name = f"icberg-ui-{config.artifact_id}"
                build_result = subprocess.run([
                    "docker", "build", "-t", image_name, "."
                ], capture_output=True, text=True, cwd=temp_path)
                
                if build_result.returncode == 0:
                    # Run Docker container
                    container_name = f"icberg-ui-{config.artifact_id}"
                    run_result = subprocess.run([
                        "docker", "run", "-d", "-p", "8080:80", "--name", container_name, image_name
                    ], capture_output=True, text=True)
                    
                    if run_result.returncode == 0:
                        return DeploymentResult(
                            success=True,
                            deployment_url="http://os.getenv("HOST", "localhost"):8080",
                            deployment_id=f"docker_{config.artifact_id}",
                            logs=[f"Docker container {container_name} running on port 8080"]
                        )
                    else:
                        return DeploymentResult(
                            success=False,
                            error=f"Failed to run Docker container: {run_result.stderr}"
                        )
                else:
                    return DeploymentResult(
                        success=False,
                        error=f"Failed to build Docker image: {build_result.stderr}"
                    )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=f"Docker deployment failed: {str(e)}"
            )
    
    def _deploy_macos_app(self, config: DeploymentConfig, artifacts: Dict[str, Any], verbose: bool) -> DeploymentResult:
        """Deploy as native macOS app using enhanced packaging system"""
        try:
            # This would use the SwiftUI backend artifacts
            if "swiftui" not in artifacts.get("backends", {}):
                return DeploymentResult(
                    success=False,
                    error="SwiftUI backend not available for macOS app deployment"
                )
            
            swiftui_path = artifacts["backends"]["swiftui"]
            if not Path(swiftui_path).exists():
                return DeploymentResult(
                    success=False,
                    error="SwiftUI artifacts not found"
                )
            
            # Create Xcode project structure
            app_name = f"ICEBURGUI_{config.artifact_id}"
            app_dir = Path(f"./deployments/{config.artifact_id}/macos_app")
            app_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy SwiftUI artifacts
            shutil.copytree(swiftui_path, app_dir / "Sources", dirs_exist_ok=True)
            
            # Create Package.swift with enhanced configuration
            package_swift = f"""// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "{app_name}",
    platforms: [
        .macOS(.v11)
    ],
    products: [
        .executable(
            name: "{app_name}",
            targets: ["{app_name}"]
        )
    ],
    targets: [
        .executableTarget(
            name: "{app_name}",
            dependencies: [],
            path: "Sources/{app_name}",
            linkerSettings: [
                .linkedFramework("SwiftUI"),
                .linkedFramework("WebKit"),
                .linkedFramework("AppKit"),
                .linkedFramework("Foundation")
            ]
        )
    ]
)
"""
            
            (app_dir / "Package.swift").write_text(package_swift)
            
            # Build macOS app
            build_result = subprocess.run([
                "swift", "build", "--configuration", "release"
            ], capture_output=True, text=True, cwd=app_dir)
            
            if build_result.returncode == 0:
                # Use enhanced packaging system
                dist_dir = app_dir / "dist"
                dist_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Initialize enhanced packager
                    packager = MacOSPackager()
                    
                    # Generate Info.plist
                    info_generator = InfoPlistGenerator()
                    bundle_id = f"com.iceburg.ui.{config.artifact_id}"
                    info_plist = info_generator.generate_for_web_app(
                        bundle_id=bundle_id,
                        app_name=app_name,
                        version="1.0",
                        author="ICEBURG"
                    )
                    
                    # Generate entitlements
                    entitlements_generator = EntitlementsGenerator()
                    entitlements = entitlements_generator.generate_for_web_app()
                    
                    # Create app bundle with enhanced features
                    executable_path = app_dir / ".build" / "release" / app_name
                    app_bundle = packager.create_app_bundle(
                        executable_path=executable_path,
                        bundle_name=app_name,
                        info_plist=info_plist,
                        resources={
                            "AppIcon.icns": Path("placeholder.icns")  # Placeholder icon
                        },
                        verbose=verbose
                    )
                    
                    # Move to dist directory
                    final_bundle = dist_dir / f"{app_name}.app"
                    if final_bundle.exists():
                        shutil.rmtree(final_bundle)
                    shutil.move(str(app_bundle), str(final_bundle))
                    
                    # Sign the app bundle
                    if packager.developer_id:
                        packager.sign_app(final_bundle, verbose=verbose)
                    
                    # Create DMG if requested
                    dmg_path = None
                    if config.build_settings and config.build_settings.get("create_dmg", False):
                        dmg_path = packager.create_dmg(final_bundle, verbose=verbose)
                    
                    logs = [
                        f"macOS app built at {app_dir}",
                        f"App bundle created: {final_bundle}",
                        f"Enhanced packaging applied",
                    ]
                    
                    if dmg_path:
                        logs.append(f"DMG created: {dmg_path}")
                    
                    return DeploymentResult(
                        success=True,
                        deployment_url=str(final_bundle),
                        deployment_id=f"macos_{config.artifact_id}",
                        logs=logs
                    )
                    
                except Exception as e:
                    return DeploymentResult(
                        success=False,
                        error=f"Failed to create enhanced app bundle: {str(e)}"
                    )
            else:
                return DeploymentResult(
                    success=False,
                    error=f"Failed to build macOS app: {build_result.stderr}"
                )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=f"macOS app deployment failed: {str(e)}"
            )
    
    def list_deployment_targets(self) -> List[DeploymentTarget]:
        """List available deployment targets"""
        return list(self.deployment_handlers.keys())
    
    def get_deployment_info(self, target: DeploymentTarget) -> Dict[str, Any]:
        """Get information about a deployment target"""
        info_map = {
            DeploymentTarget.LOCAL_SERVER: {
                "name": "Local Development Server",
                "description": "Run locally for development and testing",
                "requirements": ["Python 3.7+"],
                "cost": "Free",
                "setup_time": "Instant"
            },
            DeploymentTarget.GITHUB_PAGES: {
                "name": "GitHub Pages",
                "description": "Static site hosting via GitHub",
                "requirements": ["GitHub account", "Git repository"],
                "cost": "Free",
                "setup_time": "2-5 minutes"
            },
            DeploymentTarget.NETLIFY: {
                "name": "Netlify",
                "description": "Static site hosting with CI/CD",
                "requirements": ["Netlify account"],
                "cost": "Free tier available",
                "setup_time": "1-3 minutes"
            },
            DeploymentTarget.VERCEL: {
                "name": "Vercel",
                "description": "Static site hosting with edge functions",
                "requirements": ["Vercel account"],
                "cost": "Free tier available",
                "setup_time": "1-3 minutes"
            },
            DeploymentTarget.DOCKER: {
                "name": "Docker Container",
                "description": "Containerized deployment",
                "requirements": ["Docker"],
                "cost": "Free",
                "setup_time": "2-5 minutes"
            },
            DeploymentTarget.MACOS_APP: {
                "name": "Native macOS App",
                "description": "Native macOS application",
                "requirements": ["Xcode", "macOS"],
                "cost": "Free",
                "setup_time": "5-10 minutes"
            }
        }
        return info_map.get(target, {"name": "Unknown", "description": "Unknown target"})


def deploy_visual_ui(artifact_id: str, target: DeploymentTarget, project_id: str = "default", verbose: bool = False) -> DeploymentResult:
    """Deploy a visual UI artifact to specified target"""
    config = DeploymentConfig(
        target=target,
        artifact_id=artifact_id,
        project_id=project_id
    )
    
    deployer = VisualDeployer()
    return deployer.deploy(config, verbose)
