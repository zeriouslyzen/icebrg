"""
Autonomous Red Team
Autonomous red team agent for continuous security testing
"""

from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime
from .penetration_tester import PenetrationTester
from .vulnerability_scanner import VulnerabilityScanner
from .exploit_generator import ExploitGenerator


class AutonomousRedTeam:
    """Autonomous red team agent"""
    
    def __init__(self):
        self.penetration_tester = PenetrationTester()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.exploit_generator = ExploitGenerator()
        self.is_running = False
        self.test_results: List[Dict[str, Any]] = []
        self.discovered_vulnerabilities: List[Dict[str, Any]] = []
    
    async def start_autonomous_testing(
        self,
        targets: List[str],
        test_interval: int = 3600
    ) -> bool:
        """Start autonomous red team testing"""
        if self.is_running:
            return False
        
        self.is_running = True
        asyncio.create_task(self._autonomous_testing_loop(targets, test_interval))
        return True
    
    async def stop_autonomous_testing(self) -> bool:
        """Stop autonomous red team testing"""
        self.is_running = False
        return True
    
    async def _autonomous_testing_loop(
        self,
        targets: List[str],
        test_interval: int
    ):
        """Autonomous testing loop"""
        while self.is_running:
            for target in targets:
                if not self.is_running:
                    break
                
                # Perform tests
                test_result = await self._perform_comprehensive_test(target)
                self.test_results.append(test_result)
                
                # Store discovered vulnerabilities
                if test_result.get("vulnerabilities"):
                    self.discovered_vulnerabilities.extend(test_result["vulnerabilities"])
            
            # Wait for next interval
            await asyncio.sleep(test_interval)
    
    async def _perform_comprehensive_test(self, target: str) -> Dict[str, Any]:
        """Perform comprehensive security test"""
        result = {
            "target": target,
            "timestamp": datetime.now().isoformat(),
            "tests_performed": [],
            "vulnerabilities": [],
            "exploits_generated": []
        }
        
        # Network penetration test
        network_test = self.penetration_tester.network_penetration_test(target)
        result["tests_performed"].append("network_penetration")
        if network_test.get("vulnerabilities"):
            result["vulnerabilities"].extend(network_test["vulnerabilities"])
        
        # Web application test
        if target.startswith("http"):
            web_test = self.penetration_tester.web_application_test(target)
            result["tests_performed"].append("web_application")
            if web_test.get("vulnerabilities"):
                result["vulnerabilities"].extend(web_test["vulnerabilities"])
        
        # API security test
        if "/api/" in target:
            api_test = self.penetration_tester.api_security_test(target)
            result["tests_performed"].append("api_security")
            if api_test.get("vulnerabilities"):
                result["vulnerabilities"].extend(api_test["vulnerabilities"])
        
        # Generate exploits for discovered vulnerabilities
        for vuln in result["vulnerabilities"]:
            vuln_type = vuln.get("type")
            if vuln_type:
                exploit = self.exploit_generator.generate_exploit(vuln_type, target)
                result["exploits_generated"].append(exploit)
        
        return result
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.test_results),
            "total_vulnerabilities": len(self.discovered_vulnerabilities),
            "vulnerabilities_by_type": {},
            "vulnerabilities_by_severity": {},
            "recommendations": []
        }
        
        # Group vulnerabilities by type
        for vuln in self.discovered_vulnerabilities:
            vuln_type = vuln.get("type", "unknown")
            report["vulnerabilities_by_type"][vuln_type] = \
                report["vulnerabilities_by_type"].get(vuln_type, 0) + 1
        
        # Group vulnerabilities by severity
        for vuln in self.discovered_vulnerabilities:
            severity = vuln.get("severity", "unknown")
            report["vulnerabilities_by_severity"][severity] = \
                report["vulnerabilities_by_severity"].get(severity, 0) + 1
        
        # Generate recommendations
        if report["vulnerabilities_by_type"].get("sql_injection", 0) > 0:
            report["recommendations"].append(
                "Implement parameterized queries to prevent SQL injection"
            )
        
        if report["vulnerabilities_by_type"].get("xss", 0) > 0:
            report["recommendations"].append(
                "Implement input validation and output encoding to prevent XSS"
            )
        
        if report["vulnerabilities_by_severity"].get("high", 0) > 0:
            report["recommendations"].append(
                "Address high-severity vulnerabilities immediately"
            )
        
        return report
    
    def get_status(self) -> Dict[str, Any]:
        """Get red team status"""
        return {
            "is_running": self.is_running,
            "total_tests": len(self.test_results),
            "total_vulnerabilities": len(self.discovered_vulnerabilities),
            "last_test": self.test_results[-1] if self.test_results else None
        }

