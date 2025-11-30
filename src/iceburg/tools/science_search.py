from __future__ import annotations
from typing import List, Dict
import os
import urllib.parse
import xml.etree.ElementTree as ET

try:
    import httpx  # provided transitively by chromadb deps
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


def search_scientific_literature(query: str, max_results: int = 5, timeout_s: float = 8.0) -> List[Dict[str, str]]:
    """Best-effort literature search.
    - If ICEBURG_ENABLE_WEB=1 and httpx is available, query arXiv API.
    - Returns a list of {title, url, summary} dicts.
    """
    enable_web = os.getenv("ICEBURG_ENABLE_WEB", "0").strip() in {"1", "true", "TRUE"}
    if not enable_web or httpx is None:
        return []
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max_results}"
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(url)
            resp.raise_for_status()
            text = resp.text
        feed = ET.fromstring(text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results: List[Dict[str, str]] = []
        for entry in feed.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            link_el = None
            for link in entry.findall("atom:link", ns):
                if link.get("rel") == "alternate":
                    link_el = link
                    break
            title = (title_el.text or "").strip() if title_el is not None else ""
            summary = (summary_el.text or "").strip() if summary_el is not None else ""
            url_out = link_el.get("href") if link_el is not None else ""
            if title or url_out:
                results.append({"title": title, "url": url_out, "summary": summary})
        return results
    except Exception:
        return []
