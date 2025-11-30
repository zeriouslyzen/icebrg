"""
Copyright Vault
Manages copyright status and data vault for web-scraped content
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import uuid
import hashlib


class CopyrightVault:
    """Manages copyright status and data vault for web-scraped content"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.vault_dir = self.data_dir / "copyright_vault"
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.vault_file = self.vault_dir / "vault.jsonl"
        self.copyright_registry_file = self.vault_dir / "copyright_registry.json"
        self.vault: Dict[str, Dict[str, Any]] = {}
        self.copyright_registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def store_content(
        self,
        content: str,
        url: str,
        source_type: str = "web_scraped",
        copyright_status: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store content in copyright vault"""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        vault_id = str(uuid.uuid4())
        
        vault_entry = {
            "vault_id": vault_id,
            "content_hash": content_hash,
            "url": url,
            "source_type": source_type,
            "copyright_status": copyright_status,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(content),
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "metadata": metadata or {}
        }
        
        # Store in vault
        self.vault[vault_id] = vault_entry
        self._save_vault_entry(vault_entry)
        
        # Register copyright status
        self._register_copyright(url, copyright_status, metadata)
        
        return vault_id
    
    def _register_copyright(
        self,
        url: str,
        copyright_status: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register copyright status for a URL"""
        if url not in self.copyright_registry:
            self.copyright_registry[url] = {
                "url": url,
                "copyright_status": copyright_status,
                "first_seen": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "access_count": 0,
                "metadata": metadata or {}
            }
        else:
            self.copyright_registry[url]["last_updated"] = datetime.now().isoformat()
            if metadata:
                self.copyright_registry[url]["metadata"].update(metadata)
        
        self.copyright_registry[url]["access_count"] += 1
        self._save_registry()
    
    def check_copyright_status(self, url: str) -> str:
        """Check copyright status for a URL"""
        if url in self.copyright_registry:
            return self.copyright_registry[url].get("copyright_status", "unknown")
        return "unknown"
    
    def get_copyright_compliant_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Get content if copyright-compliant"""
        status = self.check_copyright_status(url)
        
        # Only return if copyright-compliant
        if status in ["public_domain", "fair_use", "licensed", "open_access"]:
            vault_entry = next((v for v in self.vault.values() if v.get("url") == url), None)
            return vault_entry
        return None
    
    def _save_vault_entry(self, entry: Dict[str, Any]):
        """Save vault entry to JSONL file"""
        try:
            with open(self.vault_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Error saving vault entry: {e}")
    
    def _load_registry(self):
        """Load copyright registry"""
        try:
            if self.copyright_registry_file.exists():
                with open(self.copyright_registry_file, 'r', encoding='utf-8') as f:
                    self.copyright_registry = json.load(f)
        except Exception as e:
            print(f"Error loading copyright registry: {e}")
            self.copyright_registry = {}
    
    def _save_registry(self):
        """Save copyright registry"""
        try:
            with open(self.copyright_registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.copyright_registry, f, indent=2)
        except Exception as e:
            print(f"Error saving copyright registry: {e}")
    
    def get_vault_stats(self) -> Dict[str, Any]:
        """Get statistics about the copyright vault"""
        total_entries = len(self.vault)
        total_registered = len(self.copyright_registry)
        
        copyright_statuses = {}
        source_types = {}
        
        for entry in self.vault.values():
            status = entry.get("copyright_status", "unknown")
            copyright_statuses[status] = copyright_statuses.get(status, 0) + 1
            
            source_type = entry.get("source_type", "unknown")
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        return {
            "total_entries": total_entries,
            "total_registered_urls": total_registered,
            "copyright_statuses": copyright_statuses,
            "source_types": source_types,
            "total_content_size": sum(e.get("content_length", 0) for e in self.vault.values())
        }

