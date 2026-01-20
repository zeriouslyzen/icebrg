"""
OpenSecrets API Client
Access political donations and lobbying data from OpenSecrets.org.
https://www.opensecrets.org/api/
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import requests
import os

logger = logging.getLogger(__name__)


@dataclass
class Politician:
    """Politician data from OpenSecrets."""
    cid: str  # OpenSecrets candidate ID
    name: str
    party: str
    state: str
    chamber: str  # 'House' or 'Senate'
    total_raised: float = 0.0
    top_industries: List[Dict[str, Any]] = field(default_factory=list)
    top_donors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "name": self.name,
            "party": self.party,
            "state": self.state,
            "chamber": self.chamber,
            "total_raised": self.total_raised,
            "top_industries": self.top_industries,
            "top_donors": self.top_donors,
            "metadata": self.metadata
        }


@dataclass
class Organization:
    """Organization lobbying/donation data."""
    name: str
    cycle: str
    total_contributions: float = 0.0
    soft_money: float = 0.0
    to_democrats: float = 0.0
    to_republicans: float = 0.0
    lobbying_total: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cycle": self.cycle,
            "total_contributions": self.total_contributions,
            "soft_money": self.soft_money,
            "to_democrats": self.to_democrats,
            "to_republicans": self.to_republicans,
            "lobbying_total": self.lobbying_total,
            "metadata": self.metadata
        }


class OpenSecretsClient:
    """
    OpenSecrets API client for political money data.
    
    Requires API key: https://www.opensecrets.org/api/admin/index.php
    Set OPENSECRETS_API_KEY environment variable.
    
    Docs: https://www.opensecrets.org/api/?method=candContrib
    """
    
    BASE_URL = "https://www.opensecrets.org/api/"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenSecrets client.
        
        Args:
            api_key: API key (or uses OPENSECRETS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENSECRETS_API_KEY")
        self.session = requests.Session()
        
        if self.api_key:
            logger.info("OpenSecrets client initialized with API key")
        else:
            logger.warning("OpenSecrets API key not found - API calls will fail")
    
    def _request(self, method: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Make API request."""
        if not self.api_key:
            logger.error("OpenSecrets API key required")
            return None
        
        if params is None:
            params = {}
        
        params["apikey"] = self.api_key
        params["method"] = method
        params["output"] = "json"
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenSecrets API error: {e}")
            return None
    
    def get_legislators(self, state: str) -> List[Politician]:
        """
        Get legislators for a state.
        
        Args:
            state: Two-letter state code (e.g., 'CA', 'NY')
            
        Returns:
            List of Politician objects
        """
        data = self._request("getLegislators", {"id": state})
        
        if not data:
            return []
        
        politicians = []
        legislators = data.get("response", {}).get("legislator", [])
        
        # Handle single result (not a list)
        if isinstance(legislators, dict):
            legislators = [legislators]
        
        for leg in legislators:
            attrs = leg.get("@attributes", {})
            politicians.append(Politician(
                cid=attrs.get("cid", ""),
                name=attrs.get("firstlast", ""),
                party=attrs.get("party", ""),
                state=state,
                chamber=attrs.get("office", "").split()[0] if attrs.get("office") else ""
            ))
        
        logger.info(f"OpenSecrets: Found {len(politicians)} legislators for {state}")
        return politicians
    
    def get_candidate_summary(self, cid: str, cycle: str = "2024") -> Optional[Politician]:
        """
        Get detailed fundraising summary for a candidate.
        
        Args:
            cid: OpenSecrets candidate ID
            cycle: Election cycle year
            
        Returns:
            Politician with summary data or None
        """
        data = self._request("candSummary", {"cid": cid, "cycle": cycle})
        
        if not data:
            return None
        
        summary = data.get("response", {}).get("summary", {})
        attrs = summary.get("@attributes", {}) if isinstance(summary, dict) else {}
        
        if not attrs:
            return None
        
        return Politician(
            cid=cid,
            name=attrs.get("cand_name", ""),
            party=attrs.get("party", ""),
            state=attrs.get("state", ""),
            chamber=attrs.get("chamber", ""),
            total_raised=float(attrs.get("total", 0)),
            metadata={
                "cycle": cycle,
                "spent": float(attrs.get("spent", 0)),
                "cash_on_hand": float(attrs.get("cash_on_hand", 0)),
                "debt": float(attrs.get("debt", 0)),
                "source": attrs.get("source", ""),
                "last_updated": attrs.get("last_updated", "")
            }
        )
    
    def get_candidate_contributors(self, cid: str, cycle: str = "2024") -> List[Dict[str, Any]]:
        """
        Get top contributors to a candidate.
        
        Args:
            cid: OpenSecrets candidate ID
            cycle: Election cycle year
            
        Returns:
            List of contributor dictionaries
        """
        data = self._request("candContrib", {"cid": cid, "cycle": cycle})
        
        if not data:
            return []
        
        contributors = data.get("response", {}).get("contributors", {}).get("contributor", [])
        
        if isinstance(contributors, dict):
            contributors = [contributors]
        
        result = []
        for contrib in contributors:
            attrs = contrib.get("@attributes", {})
            result.append({
                "name": attrs.get("org_name", ""),
                "total": float(attrs.get("total", 0)),
                "pacs": float(attrs.get("pacs", 0)),
                "individuals": float(attrs.get("indivs", 0))
            })
        
        logger.info(f"OpenSecrets: Found {len(result)} contributors for {cid}")
        return result
    
    def get_candidate_industries(self, cid: str, cycle: str = "2024") -> List[Dict[str, Any]]:
        """
        Get top industries contributing to a candidate.
        
        Args:
            cid: OpenSecrets candidate ID
            cycle: Election cycle year
            
        Returns:
            List of industry dictionaries
        """
        data = self._request("candIndustry", {"cid": cid, "cycle": cycle})
        
        if not data:
            return []
        
        industries = data.get("response", {}).get("industries", {}).get("industry", [])
        
        if isinstance(industries, dict):
            industries = [industries]
        
        result = []
        for ind in industries:
            attrs = ind.get("@attributes", {})
            result.append({
                "industry_code": attrs.get("industry_code", ""),
                "industry_name": attrs.get("industry_name", ""),
                "total": float(attrs.get("total", 0)),
                "individuals": float(attrs.get("indivs", 0)),
                "pacs": float(attrs.get("pacs", 0))
            })
        
        return result
    
    def get_org_summary(self, org_name: str, cycle: str = "2024") -> Optional[Organization]:
        """
        Get summary of an organization's political spending.
        
        Args:
            org_name: Organization name to search
            cycle: Election cycle year
            
        Returns:
            Organization summary or None
        """
        data = self._request("getOrgs", {"org": org_name})
        
        if not data:
            return None
        
        orgs = data.get("response", {}).get("organization", [])
        
        if isinstance(orgs, dict):
            orgs = [orgs]
        
        if not orgs:
            return None
        
        # Get first matching org
        org = orgs[0]
        attrs = org.get("@attributes", {})
        org_id = attrs.get("orgid", "")
        
        if not org_id:
            return None
        
        # Get detailed summary
        summary_data = self._request("orgSummary", {"id": org_id})
        
        if not summary_data:
            return Organization(
                name=attrs.get("orgname", org_name),
                cycle=cycle,
                metadata={"orgid": org_id}
            )
        
        summary = summary_data.get("response", {}).get("organization", {})
        summary_attrs = summary.get("@attributes", {})
        
        return Organization(
            name=summary_attrs.get("orgname", org_name),
            cycle=cycle,
            total_contributions=float(summary_attrs.get("total", 0)),
            to_democrats=float(summary_attrs.get("dems", 0)),
            to_republicans=float(summary_attrs.get("repubs", 0)),
            lobbying_total=float(summary_attrs.get("lobbying", 0)),
            metadata={
                "orgid": org_id,
                "giving_pct": summary_attrs.get("giving", ""),
                "source": summary_attrs.get("source", "")
            }
        )
    
    def get_independent_expenditures(self) -> List[Dict[str, Any]]:
        """
        Get recent independent expenditures (Super PAC spending).
        
        Returns:
            List of independent expenditure records
        """
        data = self._request("independentExpend")
        
        if not data:
            return []
        
        records = data.get("response", {}).get("indexp", [])
        
        if isinstance(records, dict):
            records = [records]
        
        result = []
        for rec in records:
            attrs = rec.get("@attributes", {})
            result.append({
                "committee": attrs.get("cmteid", ""),
                "committee_name": attrs.get("pacshort", ""),
                "candidate": attrs.get("candname", ""),
                "amount": float(attrs.get("amount", 0)),
                "support_or_oppose": attrs.get("suppopp", ""),
                "date": attrs.get("date", "")
            })
        
        return result
