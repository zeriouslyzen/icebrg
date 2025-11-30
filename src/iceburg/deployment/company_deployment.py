"""
Company Deployment
Company deployment framework
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime
from ..autonomous.tenant_manager import TenantManager
from ..autonomous.company_integration import CompanyIntegration


class CompanyDeployment:
    """Company deployment framework"""
    
    def __init__(self):
        self.tenant_manager = TenantManager()
        self.deployments: Dict[str, CompanyIntegration] = {}
    
    def deploy_for_company(
        self,
        company_id: str,
        company_name: str,
        domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deploy ICEBURG for company"""
        # Create tenant
        tenant = self.tenant_manager.create_tenant(
            company_id=company_id,
            company_name=company_name,
            metadata=metadata
        )
        
        # Create company integration
        company_integration = CompanyIntegration(company_id)
        self.deployments[company_id] = company_integration
        
        # Start autonomous learning if domain provided
        if domain:
            import asyncio
            asyncio.create_task(
                company_integration.start_autonomous_learning(domain=domain)
            )
        
        return {
            "company_id": company_id,
            "company_name": company_name,
            "tenant": tenant,
            "deployment_status": "active",
            "deployed_at": datetime.now().isoformat()
        }
    
    def get_company_deployment(self, company_id: str) -> Optional[CompanyIntegration]:
        """Get company deployment"""
        return self.deployments.get(company_id)
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all company deployments"""
        deployments = []
        
        for company_id, integration in self.deployments.items():
            status = integration.get_learning_status()
            deployments.append({
                "company_id": company_id,
                "status": status
            })
        
        return deployments
    
    def stop_company_deployment(self, company_id: str) -> bool:
        """Stop company deployment"""
        if company_id in self.deployments:
            integration = self.deployments[company_id]
            import asyncio
            asyncio.create_task(integration.stop_autonomous_learning())
            return True
        return False
    
    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics"""
        return {
            "total_deployments": len(self.deployments),
            "active_deployments": sum(
                1 for d in self.deployments.values()
                if d.get_learning_status().get("is_running", False)
            ),
            "companies": list(self.deployments.keys())
        }

