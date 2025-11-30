"""
Dynamic Tool Usage System
Uses discovered tools dynamically based on query
"""

from __future__ import annotations

import os
import subprocess
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging

from .computer_capability_discovery import (
    ComputerCapabilityDiscovery,
    DiscoveredTool,
    CapabilityRegistry
)

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Represents a tool that can be used"""
    name: str
    tool_type: str
    location: str
    description: str
    execute: Callable[[str], Any]
    capabilities: List[str]
    metadata: Dict[str, Any]


class DynamicToolUsage:
    """
    Dynamically uses discovered tools based on query.
    
    For marketing: discovers marketing tools/data and uses them
    For astrology: discovers astrology tools/data and uses them
    For any query: discovers relevant tools and uses them
    """
    
    def __init__(self, discovery: Optional[ComputerCapabilityDiscovery] = None):
        """
        Initialize dynamic tool usage.
        
        Args:
            discovery: ComputerCapabilityDiscovery instance (creates new if None)
        """
        self.discovery = discovery or ComputerCapabilityDiscovery()
        self.registry: Optional[CapabilityRegistry] = None
        self._tool_cache: Dict[str, Tool] = {}
        
        logger.info("Dynamic Tool Usage initialized")
    
    def discover_tools_for_query(self, query: str) -> List[Tool]:
        """
        Discover tools relevant to the query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant tools
        """
        # Ensure registry is built
        if self.registry is None:
            self.registry = self.discovery.build_capability_registry()
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        relevant_tools = []
        
        # Match tools based on query keywords
        for tool_id, discovered_tool in self.registry.tools.items():
            # Check if tool matches query
            tool_text = f"{discovered_tool.name} {discovered_tool.description or ''} {' '.join(discovered_tool.capabilities)}".lower()
            
            # Simple keyword matching
            if any(word in tool_text for word in query_words):
                # Convert DiscoveredTool to Tool with execute function
                tool = self._create_executable_tool(discovered_tool)
                if tool:
                    relevant_tools.append(tool)
        
        # Also check for domain-specific tools
        domain_tools = self._discover_domain_tools(query)
        relevant_tools.extend(domain_tools)
        
        logger.info(f"Discovered {len(relevant_tools)} tools for query: {query[:50]}...")
        return relevant_tools
    
    def use_computer_to_find_info(self, query: str) -> Dict[str, Any]:
        """
        Use computer resources to find information for the query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with found information
        """
        results = {
            "query": query,
            "tools_used": [],
            "data_found": [],
            "files_found": [],
            "processes_found": [],
            "errors": []
        }
        
        # Discover relevant tools
        tools = self.discover_tools_for_query(query)
        
        for tool in tools:
            try:
                # Execute tool
                tool_result = self.execute_discovered_tool(tool, query)
                
                if tool_result:
                    results["tools_used"].append({
                        "tool": tool.name,
                        "type": tool.tool_type,
                        "result": tool_result
                    })
                    
                    # Categorize results
                    if isinstance(tool_result, dict):
                        if "data" in tool_result:
                            results["data_found"].append(tool_result["data"])
                        if "files" in tool_result:
                            results["files_found"].extend(tool_result["files"])
                        if "processes" in tool_result:
                            results["processes_found"].extend(tool_result["processes"])
            except Exception as e:
                logger.warning(f"Error executing tool {tool.name}: {e}")
                results["errors"].append({
                    "tool": tool.name,
                    "error": str(e)
                })
        
        # Also search file system for relevant files
        file_results = self._search_file_system(query)
        results["files_found"].extend(file_results)
        
        logger.info(f"Found {len(results['tools_used'])} tools, "
                   f"{len(results['data_found'])} data items, "
                   f"{len(results['files_found'])} files")
        
        return results
    
    def execute_discovered_tool(self, tool: Tool, query: str) -> Any:
        """
        Execute a discovered tool with the query.
        
        Args:
            tool: Tool to execute
            query: Query to execute with
            
        Returns:
            Tool execution result
        """
        try:
            result = tool.execute(query)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool.name}: {e}")
            raise
    
    def _create_executable_tool(self, discovered_tool: DiscoveredTool) -> Optional[Tool]:
        """Convert DiscoveredTool to executable Tool."""
        tool_id = f"{discovered_tool.tool_type}_{discovered_tool.name}"
        
        # Check cache
        if tool_id in self._tool_cache:
            return self._tool_cache[tool_id]
        
        # Create execute function based on tool type
        if discovered_tool.tool_type == "python_package":
            execute_func = self._create_python_package_executor(discovered_tool)
        elif discovered_tool.tool_type == "system_command":
            execute_func = self._create_system_command_executor(discovered_tool)
        elif discovered_tool.tool_type == "file":
            execute_func = self._create_file_executor(discovered_tool)
        else:
            # Generic executor
            execute_func = self._create_generic_executor(discovered_tool)
        
        if execute_func is None:
            return None
        
        tool = Tool(
            name=discovered_tool.name,
            tool_type=discovered_tool.tool_type,
            location=discovered_tool.location or "",
            description=discovered_tool.description or "",
            execute=execute_func,
            capabilities=discovered_tool.capabilities,
            metadata=discovered_tool.metadata
        )
        
        # Cache tool
        self._tool_cache[tool_id] = tool
        
        return tool
    
    def _create_python_package_executor(self, discovered_tool: DiscoveredTool) -> Optional[Callable[[str], Any]]:
        """Create executor for Python package."""
        package_name = discovered_tool.name
        
        def execute(query: str) -> Dict[str, Any]:
            try:
                # Try to import package
                module = importlib.import_module(package_name)
                
                # Check if package has relevant functions
                result = {
                    "package": package_name,
                    "available": True,
                    "module": str(module),
                    "attributes": dir(module)[:20]  # First 20 attributes
                }
                
                # Try to find relevant functions
                query_lower = query.lower()
                relevant_attrs = [
                    attr for attr in dir(module)
                    if query_lower in attr.lower() or any(word in attr.lower() for word in query_lower.split())
                ]
                
                if relevant_attrs:
                    result["relevant_attributes"] = relevant_attrs
                
                return result
            except ImportError:
                return {
                    "package": package_name,
                    "available": False,
                    "error": "Package not importable"
                }
            except Exception as e:
                return {
                    "package": package_name,
                    "available": False,
                    "error": str(e)
                }
        
        return execute
    
    def _create_system_command_executor(self, discovered_tool: DiscoveredTool) -> Optional[Callable[[str], Any]]:
        """Create executor for system command."""
        command = discovered_tool.name
        
        def execute(query: str) -> Dict[str, Any]:
            try:
                # Try to execute command (safely)
                # Only execute read-only commands
                safe_commands = ["git", "python", "python3", "node", "npm", "docker", "kubectl"]
                
                if command not in safe_commands:
                    return {
                        "command": command,
                        "available": True,
                        "note": "Command available but not executed for safety"
                    }
                
                # For safe commands, try to get version or help
                try:
                    result = subprocess.run(
                        [command, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    return {
                        "command": command,
                        "available": True,
                        "version": result.stdout.strip() if result.returncode == 0 else None,
                        "error": result.stderr.strip() if result.stderr else None
                    }
                except subprocess.TimeoutExpired:
                    return {
                        "command": command,
                        "available": True,
                        "note": "Command timed out"
                    }
                except Exception as e:
                    return {
                        "command": command,
                        "available": True,
                        "error": str(e)
                    }
            except Exception as e:
                return {
                    "command": command,
                    "available": False,
                    "error": str(e)
                }
        
        return execute
    
    def _create_file_executor(self, discovered_tool: DiscoveredTool) -> Optional[Callable[[str], Any]]:
        """Create executor for file-based tool."""
        file_path = discovered_tool.location
        
        def execute(query: str) -> Dict[str, Any]:
            try:
                path = Path(file_path)
                if path.exists():
                    if path.is_file():
                        # Read file content (limited size)
                        if path.stat().st_size < 1024 * 1024:  # 1MB limit
                            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            
                            return {
                                "file": str(path),
                                "available": True,
                                "content_preview": content[:1000],
                                "size": path.stat().st_size
                            }
                        else:
                            return {
                                "file": str(path),
                                "available": True,
                                "size": path.stat().st_size,
                                "note": "File too large to read"
                            }
                    else:
                        return {
                            "file": str(path),
                            "available": True,
                            "type": "directory",
                            "items": [str(p) for p in path.iterdir()][:20]
                        }
                else:
                    return {
                        "file": str(path),
                        "available": False,
                        "error": "File not found"
                    }
            except Exception as e:
                return {
                    "file": file_path,
                    "available": False,
                    "error": str(e)
                }
        
        return execute
    
    def _create_generic_executor(self, discovered_tool: DiscoveredTool) -> Optional[Callable[[str], Any]]:
        """Create generic executor for unknown tool types."""
        def execute(query: str) -> Dict[str, Any]:
            return {
                "tool": discovered_tool.name,
                "type": discovered_tool.tool_type,
                "location": discovered_tool.location,
                "description": discovered_tool.description,
                "capabilities": discovered_tool.capabilities,
                "metadata": discovered_tool.metadata,
                "note": "Generic tool - execution not implemented"
            }
        
        return execute
    
    def _discover_domain_tools(self, query: str) -> List[Tool]:
        """Discover domain-specific tools (marketing, astrology, etc.)."""
        query_lower = query.lower()
        domain_tools = []
        
        # Marketing tools
        if any(word in query_lower for word in ["marketing", "advertising", "campaign", "brand", "promotion"]):
            domain_tools.extend(self._discover_marketing_tools())
        
        # Astrology tools
        if any(word in query_lower for word in ["astrology", "horoscope", "zodiac", "planet", "celestial"]):
            domain_tools.extend(self._discover_astrology_tools())
        
        # Data analysis tools
        if any(word in query_lower for word in ["data", "analysis", "analytics", "statistics", "chart", "graph"]):
            domain_tools.extend(self._discover_data_analysis_tools())
        
        return domain_tools
    
    def _discover_marketing_tools(self) -> List[Tool]:
        """Discover marketing-related tools."""
        tools = []
        
        # Check for marketing data files
        marketing_dirs = [
            Path.home() / "Documents" / "Marketing",
            Path.home() / "Desktop" / "Marketing",
            Path.cwd() / "data" / "marketing"
        ]
        
        for dir_path in marketing_dirs:
            if dir_path.exists() and dir_path.is_dir():
                for file in dir_path.rglob("*.{csv,json,xlsx,xls}"):
                    tool = Tool(
                        name=f"marketing_data_{file.stem}",
                        tool_type="marketing_data",
                        location=str(file),
                        description=f"Marketing data file: {file.name}",
                        execute=lambda f=file: self._read_marketing_file(f),
                        capabilities=["read", "analyze"],
                        metadata={"file": str(file)}
                    )
                    tools.append(tool)
        
        return tools
    
    def _discover_astrology_tools(self) -> List[Tool]:
        """Discover astrology-related tools."""
        tools = []
        
        # Check for astrology data files
        astrology_dirs = [
            Path.home() / "Documents" / "Astrology",
            Path.home() / "Desktop" / "Astrology",
            Path.cwd() / "data" / "astrology"
        ]
        
        for dir_path in astrology_dirs:
            if dir_path.exists() and dir_path.is_dir():
                for file in dir_path.rglob("*.{csv,json,txt,md}"):
                    tool = Tool(
                        name=f"astrology_data_{file.stem}",
                        tool_type="astrology_data",
                        location=str(file),
                        description=f"Astrology data file: {file.name}",
                        execute=lambda f=file: self._read_astrology_file(f),
                        capabilities=["read", "analyze"],
                        metadata={"file": str(file)}
                    )
                    tools.append(tool)
        
        return tools
    
    def _discover_data_analysis_tools(self) -> List[Tool]:
        """Discover data analysis tools."""
        tools = []
        
        # Check for Python data analysis packages
        data_packages = ["pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy", "sklearn"]
        
        for package in data_packages:
            try:
                importlib.import_module(package)
                tool = Tool(
                    name=f"data_analysis_{package}",
                    tool_type="python_package",
                    location=f"python://{package}",
                    description=f"Data analysis package: {package}",
                    execute=lambda p=package: {"package": p, "available": True},
                    capabilities=["analyze", "visualize", "process"],
                    metadata={"package": package}
                )
                tools.append(tool)
            except ImportError:
                continue
        
        return tools
    
    def _read_marketing_file(self, file_path: Path) -> Dict[str, Any]:
        """Read marketing data file."""
        try:
            if file_path.suffix == ".csv":
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)[:10]  # First 10 rows
                    return {
                        "file": str(file_path),
                        "type": "csv",
                        "data": rows,
                        "columns": list(rows[0].keys()) if rows else []
                    }
            elif file_path.suffix == ".json":
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        "file": str(file_path),
                        "type": "json",
                        "data": data
                    }
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:1000]
                    return {
                        "file": str(file_path),
                        "type": "text",
                        "content": content
                    }
        except Exception as e:
            return {
                "file": str(file_path),
                "error": str(e)
            }
    
    def _read_astrology_file(self, file_path: Path) -> Dict[str, Any]:
        """Read astrology data file."""
        return self._read_marketing_file(file_path)  # Same implementation
    
    def _search_file_system(self, query: str) -> List[Dict[str, Any]]:
        """Search file system for relevant files."""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Search in common locations
        search_dirs = [
            Path.home() / "Documents",
            Path.home() / "Desktop",
            Path.cwd() / "data"
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            try:
                for file in search_dir.rglob("*"):
                    if file.is_file():
                        # Check if filename or path matches query
                        file_text = f"{file.name} {file.parent}".lower()
                        if any(word in file_text for word in query_words):
                            results.append({
                                "path": str(file),
                                "name": file.name,
                                "size": file.stat().st_size,
                                "type": file.suffix
                            })
                            
                            if len(results) >= 20:  # Limit results
                                break
            except Exception:
                continue
            
            if len(results) >= 20:
                break
        
        return results

