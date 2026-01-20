"""
OpenCorporates API Client
Access corporate data from the world's largest open database of companies.
https://api.opencorporates.com/
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import requests
import os

logger = logging.getLogger(__name__)


@dataclass
class Company:
    """Company data from OpenCorporates."""
    name: str
    company_number: str
    jurisdiction_code: str
    incorporation_date: Optional[str] = None
    company_type: Optional[str] = None
    registry_url: Optional[str] = None
    status: Optional[str] = None
    officers: List[Dict[str, Any]] = field(default_factory=list)
    filings: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "company_number": self.company_number,
            "jurisdiction_code": self.jurisdiction_code,
            "incorporation_date": self.incorporation_date,
            "company_type": self.company_type,
            "registry_url": self.registry_url,
            "status": self.status,
            "officers": self.officers,
            "filings": self.filings,
            "metadata": self.metadata
        }


@dataclass
class Officer:
    """Officer/director data from OpenCorporates."""
    name: str
    position: str
    company_name: str
    company_number: str
    jurisdiction_code: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    current: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position,
            "company_name": self.company_name,
            "company_number": self.company_number,
            "jurisdiction_code": self.jurisdiction_code,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "current": self.current,
            "metadata": self.metadata
        }


class OpenCorporatesClient:
    """
    OpenCorporates API client for corporate research.
    
    Free tier: 500 requests/month, limited data
    API key: Set OPENCORPORATES_API_KEY env var for full access
    
    Docs: https://api.opencorporates.com/documentation/API-Reference
    """
    
    BASE_URL = "https://api.opencorporates.com/v0.4"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenCorporates client.
        
        Args:
            api_key: API key (or uses OPENCORPORATES_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENCORPORATES_API_KEY")
        self.session = requests.Session()
        
        if self.api_key:
            logger.info("OpenCorporates client initialized with API key")
        else:
            logger.warning("OpenCorporates API key not found - using free tier (limited)")
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Make API request."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        if params is None:
            params = {}
        
        if self.api_key:
            params["api_token"] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenCorporates API error: {e}")
            return None
    
    def search_companies(
        self,
        query: str,
        jurisdiction_code: Optional[str] = None,
        limit: int = 10
    ) -> List[Company]:
        """
        Search for companies.
        
        Args:
            query: Company name search query
            jurisdiction_code: Optional jurisdiction filter (e.g., 'us_de', 'gb')
            limit: Maximum results
            
        Returns:
            List of Company objects
        """
        params = {
            "q": query,
            "per_page": min(limit, 30)  # API max is 30
        }
        
        if jurisdiction_code:
            params["jurisdiction_code"] = jurisdiction_code
        
        data = self._request("companies/search", params)
        
        if not data:
            return []
        
        companies = []
        for item in data.get("results", {}).get("companies", []):
            company_data = item.get("company", {})
            companies.append(Company(
                name=company_data.get("name", ""),
                company_number=company_data.get("company_number", ""),
                jurisdiction_code=company_data.get("jurisdiction_code", ""),
                incorporation_date=company_data.get("incorporation_date"),
                company_type=company_data.get("company_type"),
                registry_url=company_data.get("registry_url"),
                status=company_data.get("current_status"),
                metadata={
                    "opencorporates_url": company_data.get("opencorporates_url"),
                    "source": company_data.get("source", {})
                }
            ))
        
        logger.info(f"OpenCorporates: Found {len(companies)} companies for '{query}'")
        return companies
    
    def get_company(
        self,
        jurisdiction_code: str,
        company_number: str
    ) -> Optional[Company]:
        """
        Get detailed company information.
        
        Args:
            jurisdiction_code: Jurisdiction (e.g., 'us_de')
            company_number: Company registration number
            
        Returns:
            Company object with full details or None
        """
        endpoint = f"companies/{jurisdiction_code}/{company_number}"
        data = self._request(endpoint)
        
        if not data:
            return None
        
        company_data = data.get("results", {}).get("company", {})
        
        if not company_data:
            return None
        
        return Company(
            name=company_data.get("name", ""),
            company_number=company_data.get("company_number", ""),
            jurisdiction_code=company_data.get("jurisdiction_code", ""),
            incorporation_date=company_data.get("incorporation_date"),
            company_type=company_data.get("company_type"),
            registry_url=company_data.get("registry_url"),
            status=company_data.get("current_status"),
            officers=[
                {
                    "name": o.get("officer", {}).get("name"),
                    "position": o.get("officer", {}).get("position"),
                    "start_date": o.get("officer", {}).get("start_date"),
                    "end_date": o.get("officer", {}).get("end_date")
                }
                for o in company_data.get("officers", [])
            ],
            filings=[
                {
                    "title": f.get("filing", {}).get("title"),
                    "date": f.get("filing", {}).get("date"),
                    "url": f.get("filing", {}).get("url")
                }
                for f in company_data.get("filings", [])
            ],
            metadata={
                "opencorporates_url": company_data.get("opencorporates_url"),
                "registered_address": company_data.get("registered_address"),
                "agent": company_data.get("agent"),
                "previous_names": company_data.get("previous_names", [])
            }
        )
    
    def search_officers(
        self,
        query: str,
        jurisdiction_code: Optional[str] = None,
        limit: int = 10
    ) -> List[Officer]:
        """
        Search for corporate officers/directors.
        
        Args:
            query: Officer name search
            jurisdiction_code: Optional jurisdiction filter
            limit: Maximum results
            
        Returns:
            List of Officer objects
        """
        params = {
            "q": query,
            "per_page": min(limit, 30)
        }
        
        if jurisdiction_code:
            params["jurisdiction_code"] = jurisdiction_code
        
        data = self._request("officers/search", params)
        
        if not data:
            return []
        
        officers = []
        for item in data.get("results", {}).get("officers", []):
            officer_data = item.get("officer", {})
            company = officer_data.get("company", {})
            
            officers.append(Officer(
                name=officer_data.get("name", ""),
                position=officer_data.get("position", ""),
                company_name=company.get("name", ""),
                company_number=company.get("company_number", ""),
                jurisdiction_code=company.get("jurisdiction_code", ""),
                start_date=officer_data.get("start_date"),
                end_date=officer_data.get("end_date"),
                current=officer_data.get("end_date") is None,
                metadata={
                    "opencorporates_url": officer_data.get("opencorporates_url")
                }
            ))
        
        logger.info(f"OpenCorporates: Found {len(officers)} officers for '{query}'")
        return officers
    
    def get_officer_companies(self, officer_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find all companies an officer is associated with.
        
        Useful for mapping someone's corporate network.
        
        Args:
            officer_name: Name of the officer
            limit: Maximum companies to return
            
        Returns:
            List of company associations
        """
        officers = self.search_officers(officer_name, limit=limit)
        
        # Deduplicate by company
        companies = {}
        for officer in officers:
            key = f"{officer.jurisdiction_code}/{officer.company_number}"
            if key not in companies:
                companies[key] = {
                    "company_name": officer.company_name,
                    "company_number": officer.company_number,
                    "jurisdiction_code": officer.jurisdiction_code,
                    "positions": []
                }
            companies[key]["positions"].append({
                "position": officer.position,
                "start_date": officer.start_date,
                "end_date": officer.end_date,
                "current": officer.current
            })
        
        return list(companies.values())
