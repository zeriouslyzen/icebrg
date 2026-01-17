"""
Computer Capability Discovery Engine
Dynamically discovers computer resources without hardcoding
"""

from __future__ import annotations

import os
import sys
import subprocess
import platform
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import logging
import json

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredTool:
    """Represents a discovered tool/capability"""
    name: str
    tool_type: str  # "python_package", "system_command", "api", "file", "process"
    location: Optional[str] = None
    description: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityRegistry:
    """Registry of discovered capabilities"""
    tools: Dict[str, DiscoveredTool] = field(default_factory=dict)
    processes: List[Dict[str, Any]] = field(default_factory=list)
    network_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    file_system_structure: Dict[str, Any] = field(default_factory=dict)
    apis_and_services: List[Dict[str, Any]] = field(default_factory=list)
    discovered_at: float = field(default_factory=lambda: __import__('time').time())


class ComputerCapabilityDiscovery:
    """
    Dynamically discovers computer capabilities without hardcoding.
    
    Discovers:
    - Installed tools (Python packages, system commands, APIs)
    - Running processes and network interfaces
    - File system structure and data locations
    - Available APIs and services
    - Builds capability registry dynamically
    """
    
    def __init__(self, root_path: Optional[Path] = None):
        """
        Initialize computer capability discovery.
        
        Args:
            root_path: Root path to scan (default: user's home directory)
        """
        self.root_path = root_path or Path.home()
        self.registry: Optional[CapabilityRegistry] = None
        self._python_packages: Dict[str, Any] = {}
        self._system_commands: Set[str] = set()
        self._discovered_files: Dict[str, List[Path]] = {}
        
        logger.info(f"Computer Capability Discovery initialized (root: {self.root_path})")
    
    def discover_installed_tools(self) -> Dict[str, Any]:
        """
        Discover installed tools (Python packages, system commands, APIs).
        
        Returns:
            Dictionary of discovered tools by type
        """
        tools = {
            "python_packages": {},
            "system_commands": [],
            "apis": [],
            "file_tools": []
        }
        
        # Discover Python packages
        try:
            import pkg_resources
            installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
            tools["python_packages"] = installed_packages
            self._python_packages = installed_packages
            logger.info(f"Discovered {len(installed_packages)} Python packages")
        except Exception as e:
            logger.warning(f"Could not discover Python packages: {e}")
        
        # Discover system commands (common tools)
        common_commands = [
            "curl", "wget", "git", "python", "python3", "node", "npm",
            "docker", "kubectl", "aws", "gcloud", "terraform",
            "ffmpeg", "imagemagick", "pandoc", "latex",
            "jupyter", "notebook", "ipython"
        ]
        
        for cmd in common_commands:
            if self._command_exists(cmd):
                tools["system_commands"].append(cmd)
                self._system_commands.add(cmd)
        
        logger.info(f"Discovered {len(tools['system_commands'])} system commands")
        
        # Discover file-based tools (scripts, executables)
        file_tools = self._discover_file_tools()
        tools["file_tools"] = file_tools
        
        return tools
    
    def discover_running_processes(self) -> List[Dict[str, Any]]:
        """
        Discover running processes.
        
        Returns:
            List of running processes with metadata
        """
        processes = []
        
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, cannot discover processes")
            return processes
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    proc_info = proc.info
                    processes.append({
                        "pid": proc_info.get('pid'),
                        "name": proc_info.get('name'),
                        "cmdline": ' '.join(proc_info.get('cmdline') or []),
                        "cpu_percent": proc_info.get('cpu_percent', 0.0),
                        "memory_mb": proc_info.get('memory_info', {}).get('rss', 0) / (1024 * 1024) if proc_info.get('memory_info') else 0.0
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            logger.info(f"Discovered {len(processes)} running processes")
        except Exception as e:
            logger.warning(f"Error discovering processes: {e}")
        
        return processes
    
    def discover_network_interfaces(self) -> List[Dict[str, Any]]:
        """
        Discover network interfaces.
        
        Returns:
            List of network interfaces with metadata
        """
        interfaces = []
        
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, cannot discover network interfaces")
            return interfaces
        
        try:
            net_if_addrs = psutil.net_if_addrs()
            net_if_stats = psutil.net_if_stats()
            
            for interface_name, addrs in net_if_addrs.items():
                interface_info = {
                    "name": interface_name,
                    "addresses": [],
                    "is_up": net_if_stats.get(interface_name, {}).isup if interface_name in net_if_stats else False,
                    "speed": net_if_stats.get(interface_name, {}).speed if interface_name in net_if_stats else 0
                }
                
                for addr in addrs:
                    interface_info["addresses"].append({
                        "family": str(addr.family),
                        "address": addr.address,
                        "netmask": addr.netmask if hasattr(addr, 'netmask') else None,
                        "broadcast": addr.broadcast if hasattr(addr, 'broadcast') else None
                    })
                
                interfaces.append(interface_info)
            
            logger.info(f"Discovered {len(interfaces)} network interfaces")
        except Exception as e:
            logger.warning(f"Error discovering network interfaces: {e}")
        
        return interfaces
    
    def discover_file_system_structure(self, root_path: Optional[Path] = None, max_depth: int = 3) -> Dict[str, Any]:
        """
        Discover file system structure and data locations.
        
        Args:
            root_path: Root path to scan (default: self.root_path)
            max_depth: Maximum depth to scan (default: 3)
            
        Returns:
            Dictionary of file system structure
        """
        root = root_path or self.root_path
        structure = {
            "root": str(root),
            "directories": [],
            "data_files": [],
            "code_files": [],
            "config_files": [],
            "documentation_files": []
        }
        
        try:
            root_depth = len(root.parts)
            for path in root.rglob("*"):
                # Calculate depth relative to root
                path_depth = len(path.parts) - root_depth
                if path.is_dir():
                    if path_depth <= max_depth:
                        structure["directories"].append({
                            "path": str(path),
                            "depth": path_depth,
                            "name": path.name
                        })
                elif path.is_file():
                    # Categorize files
                    suffix = path.suffix.lower()
                    file_info = {
                        "path": str(path),
                        "name": path.name,
                        "size": path.stat().st_size,
                        "suffix": suffix
                    }
                    
                    if suffix in ['.json', '.csv', '.tsv', '.parquet', '.db', '.sqlite']:
                        structure["data_files"].append(file_info)
                    elif suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.go', '.rs']:
                        structure["code_files"].append(file_info)
                    elif suffix in ['.yaml', '.yml', '.json', '.toml', '.ini', '.conf', '.env']:
                        structure["config_files"].append(file_info)
                    elif suffix in ['.md', '.txt', '.rst', '.pdf', '.doc', '.docx']:
                        structure["documentation_files"].append(file_info)
            
            logger.info(f"Discovered file system structure: {len(structure['directories'])} dirs, "
                       f"{len(structure['data_files'])} data files, {len(structure['code_files'])} code files")
        except Exception as e:
            logger.warning(f"Error discovering file system structure: {e}")
        
        return structure
    
    def discover_apis_and_services(self) -> List[Dict[str, Any]]:
        """
        Discover available APIs and services.
        
        Returns:
            List of discovered APIs and services
        """
        apis = []
        
        # Check for common API endpoints
        common_ports = [8000, 8080, 3000, 5000, 9000, 5432, 3306, 6379, 27017]
        
        if PSUTIL_AVAILABLE:
            try:
                connections = psutil.net_connections(kind='inet')
                active_ports = {conn.laddr.port for conn in connections if conn.status == 'LISTEN'}
                
                for port in common_ports:
                    if port in active_ports:
                        apis.append({
                            "type": "local_service",
                            "port": port,
                            "status": "listening",
                            "description": self._guess_service_type(port)
                        })
            except Exception as e:
                logger.warning(f"Error discovering API ports: {e}")
        
        # Check for environment variables that indicate APIs
        api_env_vars = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "AWS_ACCESS_KEY_ID", "AZURE_API_KEY", "GCP_API_KEY"
        ]
        
        for env_var in api_env_vars:
            if os.getenv(env_var):
                apis.append({
                    "type": "api_key",
                    "name": env_var,
                    "status": "configured",
                    "description": f"API key for {env_var.replace('_API_KEY', '').replace('_KEY', '')}"
                })
        
        logger.info(f"Discovered {len(apis)} APIs and services")
        return apis
    
    def build_capability_registry(self) -> CapabilityRegistry:
        """
        Build complete capability registry.
        
        Returns:
            CapabilityRegistry with all discovered capabilities
        """
        logger.info("Building capability registry...")
        
        tools = self.discover_installed_tools()
        processes = self.discover_running_processes()
        network_interfaces = self.discover_network_interfaces()
        file_system = self.discover_file_system_structure()
        apis = self.discover_apis_and_services()
        
        # Build tool registry
        discovered_tools = {}
        
        # Add Python packages as tools
        for pkg_name, version in tools.get("python_packages", {}).items():
            discovered_tools[f"python_{pkg_name}"] = DiscoveredTool(
                name=pkg_name,
                tool_type="python_package",
                location=f"python://{pkg_name}",
                description=f"Python package {pkg_name} version {version}",
                capabilities=["import", "use_in_code"],
                metadata={"version": version}
            )
        
        # Add system commands as tools
        for cmd in tools.get("system_commands", []):
            discovered_tools[f"command_{cmd}"] = DiscoveredTool(
                name=cmd,
                tool_type="system_command",
                location=f"command://{cmd}",
                description=f"System command: {cmd}",
                capabilities=["execute", "run_command"],
                metadata={}
            )
        
        registry = CapabilityRegistry(
            tools=discovered_tools,
            processes=processes,
            network_interfaces=network_interfaces,
            file_system_structure=file_system,
            apis_and_services=apis
        )
        
        self.registry = registry
        logger.info(f"Capability registry built: {len(discovered_tools)} tools, "
                   f"{len(processes)} processes, {len(network_interfaces)} interfaces")
        
        return registry
    
    def _command_exists(self, command: str) -> bool:
        """Check if a system command exists."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["where", command],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
            else:
                result = subprocess.run(
                    ["which", command],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
            return result.returncode == 0
        except Exception:
            return False
    
    def _discover_file_tools(self) -> List[Dict[str, Any]]:
        """Discover file-based tools (scripts, executables)."""
        tools = []
        
        # Common tool locations
        tool_paths = [
            Path.home() / ".local" / "bin",
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path.home() / "bin"
        ]
        
        for tool_path in tool_paths:
            if tool_path.exists() and tool_path.is_dir():
                try:
                    for file in tool_path.iterdir():
                        if file.is_file() and os.access(file, os.X_OK):
                            tools.append({
                                "name": file.name,
                                "path": str(file),
                                "type": "executable",
                                "location": str(tool_path)
                            })
                except Exception:
                    continue
        
        return tools
    
    def _guess_service_type(self, port: int) -> str:
        """Guess service type based on port number."""
        port_services = {
            8000: "Development server",
            8080: "Web server",
            3000: "Node.js server",
            5000: "Flask server",
            9000: "SonarQube",
            5432: "PostgreSQL",
            3306: "MySQL",
            6379: "Redis",
            27017: "MongoDB"
        }
        return port_services.get(port, "Unknown service")
    
    def get_registry(self) -> Optional[CapabilityRegistry]:
        """Get current capability registry (builds if not exists)."""
        if self.registry is None:
            self.build_capability_registry()
        return self.registry
    
    def refresh_registry(self) -> CapabilityRegistry:
        """Refresh capability registry by rediscovering everything."""
        self.registry = None
        return self.build_capability_registry()

