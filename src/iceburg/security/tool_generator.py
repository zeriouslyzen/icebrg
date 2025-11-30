"""
Tool Generator
Generates penetration testing tools
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from .exploit_generator import ExploitGenerator
from .tool_inventory import ToolInventory


class ToolGenerator:
    """Generates penetration testing tools"""
    
    def __init__(self, inventory: Optional[ToolInventory] = None):
        self.exploit_generator = ExploitGenerator()
        self.inventory = inventory or ToolInventory()
        # Keep backward compatibility
        self.generated_tools: List[Dict[str, Any]] = []
    
    def generate_penetration_tool(
        self,
        tool_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate penetration testing tool"""
        if tool_type == "exploit_script":
            return self._generate_exploit_script(parameters)
        elif tool_type == "scanner":
            return self._generate_scanner(parameters)
        elif tool_type == "payload_generator":
            return self._generate_payload_generator(parameters)
        else:
            return {
                "error": f"Unknown tool type: {tool_type}"
            }
    
    def _generate_exploit_script(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate exploit script"""
        vulnerability_type = parameters.get("vulnerability_type")
        target = parameters.get("target")
        language = parameters.get("language", "python")
        
        exploit = self.exploit_generator.generate_exploit(
            vulnerability_type,
            target,
            parameters
        )
        
        script = self.exploit_generator.create_exploit_script(exploit, language)
        
        tool = {
            "type": "exploit_script",
            "vulnerability_type": vulnerability_type,
            "target": target,
            "language": language,
            "script": script,
            "generated_at": datetime.now().isoformat()
        }
        
        # Store in persistent inventory
        tool_id = self.inventory.add_tool(tool)
        tool["id"] = tool_id
        
        # Keep backward compatibility
        self.generated_tools.append(tool)
        return tool
    
    def _generate_scanner(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate vulnerability scanner"""
        scan_type = parameters.get("scan_type", "network")
        target = parameters.get("target")
        
        scanner_script = f"""#!/usr/bin/env python3
# Vulnerability Scanner
# Type: {scan_type}
# Target: {target}

import socket
import requests
import sys

target = "{target}"
scan_type = "{scan_type}"

def scan_network():
    ports = [22, 80, 443, 8080, 3306, 5432]
    open_ports = []
    
    for port in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((target, port))
            sock.close()
            
            if result == 0:
                open_ports.append(port)
        except Exception:
            pass
    
    print(f"Open ports: {{open_ports}}")
    return open_ports

def scan_web():
    try:
        response = requests.get(target, timeout=5)
        print(f"Status: {{response.status_code}}")
        print(f"Headers: {{response.headers}}")
    except Exception as e:
        print(f"Error: {{e}}")

if scan_type == "network":
    scan_network()
elif scan_type == "web":
    scan_web()
"""
        
        tool = {
            "type": "scanner",
            "scan_type": scan_type,
            "target": target,
            "script": scanner_script,
            "generated_at": datetime.now().isoformat()
        }
        
        # Store in persistent inventory
        tool_id = self.inventory.add_tool(tool)
        tool["id"] = tool_id
        
        # Keep backward compatibility
        self.generated_tools.append(tool)
        return tool
    
    def _generate_payload_generator(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate payload generator"""
        payload_type = parameters.get("payload_type", "sql_injection")
        
        generator_script = f"""#!/usr/bin/env python3
# Payload Generator
# Type: {payload_type}

payloads = {{
    "sql_injection": [
        "' OR '1'='1",
        "' OR 1=1--",
        "admin'--"
    ],
    "xss": [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>"
    ],
    "command_injection": [
        "; ls",
        "| cat /etc/passwd"
    ]
}}

selected_payloads = payloads.get("{payload_type}", [])
for payload in selected_payloads:
    print(payload)
"""
        
        tool = {
            "type": "payload_generator",
            "payload_type": payload_type,
            "script": generator_script,
            "generated_at": datetime.now().isoformat()
        }
        
        # Store in persistent inventory
        tool_id = self.inventory.add_tool(tool)
        tool["id"] = tool_id
        
        # Keep backward compatibility
        self.generated_tools.append(tool)
        return tool
    
    def get_generated_tools(self) -> List[Dict[str, Any]]:
        """Get all generated tools"""
        # Get from persistent inventory
        return self.inventory.get_all_tools()
    
    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get specific tool"""
        return self.inventory.get_tool(tool_id)
    
    def search_tools(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search tools by query using semantic search"""
        return self.inventory.search_tools(query, k=k)

