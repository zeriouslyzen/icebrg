"""
Tenant Manager
Multi-tenant company deployment management
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path
import json


class TenantManager:
    """Manages multi-tenant company deployments"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or "data/tenants")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tenants: Dict[str, Dict[str, Any]] = {}
        self._load_tenants()
    
    def _load_tenants(self):
        """Load tenants from storage"""
        tenants_file = self.data_dir / "tenants.json"
        if tenants_file.exists():
            try:
                with open(tenants_file, 'r') as f:
                    self.tenants = json.load(f)
            except Exception:
                self.tenants = {}
    
    def _save_tenants(self):
        """Save tenants to storage"""
        tenants_file = self.data_dir / "tenants.json"
        try:
            with open(tenants_file, 'w') as f:
                json.dump(self.tenants, f, indent=2)
        except Exception:
            pass
    
    def create_tenant(
        self,
        company_id: str,
        company_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create new tenant"""
        tenant = {
            "company_id": company_id,
            "company_name": company_name,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "data_isolation": True,
            "access_control": {
                "roles": [],
                "permissions": {}
            },
            "usage_analytics": {
                "queries": 0,
                "storage_used": 0,
                "last_activity": None
            }
        }
        
        self.tenants[company_id] = tenant
        self._save_tenants()
        
        # Create tenant data directory
        tenant_dir = self.data_dir / company_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        
        return tenant
    
    def get_tenant(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant by ID"""
        return self.tenants.get(company_id)
    
    def list_tenants(self) -> List[Dict[str, Any]]:
        """List all tenants"""
        return list(self.tenants.values())
    
    def update_tenant(
        self,
        company_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update tenant"""
        if company_id not in self.tenants:
            return False
        
        self.tenants[company_id].update(updates)
        self.tenants[company_id]["updated_at"] = datetime.now().isoformat()
        self._save_tenants()
        return True
    
    def delete_tenant(self, company_id: str) -> bool:
        """Delete tenant"""
        if company_id not in self.tenants:
            return False
        
        del self.tenants[company_id]
        self._save_tenants()
        
        # Delete tenant data directory
        tenant_dir = self.data_dir / company_id
        if tenant_dir.exists():
            import shutil
            shutil.rmtree(tenant_dir)
        
        return True
    
    def set_access_control(
        self,
        company_id: str,
        role: str,
        permissions: List[str]
    ) -> bool:
        """Set access control for tenant"""
        if company_id not in self.tenants:
            return False
        
        tenant = self.tenants[company_id]
        if "access_control" not in tenant:
            tenant["access_control"] = {"roles": [], "permissions": {}}
        
        tenant["access_control"]["roles"].append(role)
        tenant["access_control"]["permissions"][role] = permissions
        self._save_tenants()
        return True
    
    def check_permission(
        self,
        company_id: str,
        role: str,
        permission: str
    ) -> bool:
        """Check if role has permission"""
        tenant = self.get_tenant(company_id)
        if not tenant:
            return False
        
        permissions = tenant.get("access_control", {}).get("permissions", {})
        role_permissions = permissions.get(role, [])
        
        return permission in role_permissions or "admin" in role_permissions
    
    def update_usage_analytics(
        self,
        company_id: str,
        analytics: Dict[str, Any]
    ) -> bool:
        """Update usage analytics for tenant"""
        if company_id not in self.tenants:
            return False
        
        tenant = self.tenants[company_id]
        if "usage_analytics" not in tenant:
            tenant["usage_analytics"] = {}
        
        tenant["usage_analytics"].update(analytics)
        tenant["usage_analytics"]["last_activity"] = datetime.now().isoformat()
        self._save_tenants()
        return True
    
    def get_tenant_data_path(self, company_id: str) -> Path:
        """Get tenant data path"""
        return self.data_dir / company_id
    
    def isolate_data(self, company_id: str) -> bool:
        """Ensure data isolation for tenant"""
        tenant = self.get_tenant(company_id)
        if not tenant:
            return False
        
        tenant["data_isolation"] = True
        self._save_tenants()
        return True

