"""
Birth Data Extraction Utility
Handles extraction and parsing of birth date and location from various input formats
"""

import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_birth_data_from_message(message: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
    """
    Extract birth data from WebSocket message and/or query text.
    
    Priority:
    1. Natural language extraction from query (user explicitly typed it)
    2. Structured data from message['data'] (cached/previous)
    3. Return None if not found
    
    Args:
        message: WebSocket message dictionary
        query: User query text
        
    Returns:
        Dictionary with 'birth_datetime' and 'location' (lat, lng tuple) or None
    """
    birth_data = None
    
    # Priority 1: Try natural language extraction from query first (user explicitly typed it)
    # This ensures fresh user input takes precedence over cached data
    query_birth_data = extract_birth_data_from_query(query)
    if query_birth_data and query_birth_data.get("birth_datetime"):
        birth_data = query_birth_data
        logger.info(f"🌌 Extracted birth data from query text: {birth_data['birth_datetime']}, {birth_data.get('location')}")
        return birth_data
    
    # Priority 2: Fall back to structured data from message (cached/previous)
    data = message.get("data")
    if data and isinstance(data, dict):
        birth_date_str = data.get("birth_date") or data.get("birthDate")
        location_data = data.get("location") or data.get("location")
        
        if birth_date_str:
            try:
                birth_datetime = parse_birth_date(birth_date_str)
                location = None
                
                if location_data:
                    location = parse_location(location_data)
                
                if birth_datetime:
                    birth_data = {
                        "birth_datetime": birth_datetime,
                        "location": location
                    }
                    logger.info(f"🌌 Extracted birth data from structured input: {birth_datetime}, {location}")
            except Exception as e:
                logger.warning(f"Error parsing structured birth data: {e}")
    
    return birth_data


def extract_birth_data_from_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Extract birth date and location from natural language query.
    
    Patterns:
    - "I was born on March 15, 1990"
    - "Born: 1990-03-15"
    - "Birth date: 15/03/1990"
    - "I was born in New York"
    - "Born in 40.7128, -74.0060"
    
    Returns:
        Dictionary with 'birth_datetime' and 'location' or None
    """
    query_lower = query.lower()
    birth_datetime = None
    location = None
    
    # Extract birth date patterns (order matters - more specific first)
    date_patterns = [
        # Month name format with time: "December 26, 1991 at 7:20 AM" or "december 26, 1991 at 7:20 am"
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})\s+at\s+(\d{1,2}):(\d{2})\s+(am|pm)\b',
        # Month abbreviation format with time: "Dec 26, 1991 at 7:20 AM" or "dec 26, 1991 at 7:20 am"
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2}),?\s+(\d{4})\s+at\s+(\d{1,2}):(\d{2})\s+(am|pm)\b',
        # Numeric format with time: "12/26/1991 7:20 AM" or "12-26-1991 7:20 AM" (must come before ISO)
        r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s+(\d{1,2}):(\d{2})\s+(am|pm)\b',
        # ISO format: 1990-03-15 or 1990-03-15T14:30:00
        r'\b(\d{4})-(\d{1,2})-(\d{1,2})(?:T(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?\b',
        # US format: March 15, 1990 or 03/15/1990
        r'\b(?:born|birth).*?(?:on|date|:)?\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',
        # Month name format (full): December 26, 1991 or december 26 1991 (comma optional)
        r'\b(?:born|birth|my\s+birthdate|birthdate|birthday).*?(?:on|date|:)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})\b',
        # Month abbreviation format: Dec 26, 1991 or dec 26 1991 (comma optional)
        r'\b(?:born|birth|my\s+birthdate|birthdate|birthday).*?(?:on|date|:)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2}),?\s+(\d{4})\b',
        # Year only: born in 1990
        r'\b(?:born|birth).*?(?:in|year|:)?\s*(\d{4})\b',
    ]
    
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    month_abbrevs = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    for pattern in date_patterns:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            try:
                groups = match.groups()
                
                # Numeric format with time: "12/26/1991 7:20 AM" (6 groups: month, day, year, hour, minute, am/pm)
                # Check if this pattern matched the numeric-with-time pattern (has 6 groups and am/pm indicator)
                if len(groups) == 6 and groups[5] and ('/' in match.group(0) or '-' in match.group(0)):
                    # This is numeric format with time
                    month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                    hour = int(groups[3])
                    minute = int(groups[4])
                    am_pm = groups[5].lower()
                    # Convert to 24-hour format
                    if am_pm == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    # Validate month/day order (try both)
                    if month > 12:  # Likely DD/MM/YYYY
                        day, month = month, day
                    birth_datetime = datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
                # ISO format: "1990-03-15" or "1990-03-15T14:30:00" (3-6 groups, starts with year)
                elif len(groups) >= 3 and groups[0].isdigit() and len(groups[0]) == 4 and '-' in match.group(0) and not ('/' in match.group(0)):
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    hour = int(groups[3]) if len(groups) > 3 and groups[3] else 12
                    minute = int(groups[4]) if len(groups) > 4 and groups[4] else 0
                    second = int(groups[5]) if len(groups) > 5 and groups[5] else 0
                    birth_datetime = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                
                # Month name format with time (full): "December 26, 1991 at 7:20 AM"
                elif len(groups) >= 6 and groups[0].lower() in month_names:
                    month = month_names[groups[0].lower()]
                    day = int(groups[1])
                    year = int(groups[2])
                    hour = int(groups[3])
                    minute = int(groups[4])
                    am_pm = groups[5].lower() if len(groups) > 5 else None
                    # Convert to 24-hour format
                    if am_pm == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    birth_datetime = datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
                # Month abbreviation format with time: "Dec 26, 1991 at 7:20 AM"
                elif len(groups) >= 6 and groups[0].lower() in month_abbrevs:
                    month = month_abbrevs[groups[0].lower()]
                    day = int(groups[1])
                    year = int(groups[2])
                    hour = int(groups[3])
                    minute = int(groups[4])
                    am_pm = groups[5].lower() if len(groups) > 5 else None
                    # Convert to 24-hour format
                    if am_pm == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    birth_datetime = datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
                # Month name format (full or abbreviated) without time
                elif groups[0].lower() in month_names:
                    month = month_names[groups[0].lower()]
                    day = int(groups[1])
                    year = int(groups[2])
                    # Default to noon if no time found
                    hour, minute = 12, 0
                    birth_datetime = datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
                elif groups[0].lower() in month_abbrevs:
                    month = month_abbrevs[groups[0].lower()]
                    day = int(groups[1])
                    year = int(groups[2])
                    # Default to noon if no time found
                    hour, minute = 12, 0
                    birth_datetime = datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
                
                # Numeric format (MM/DD/YYYY or DD/MM/YYYY)
                elif len(groups) >= 3:
                    # Try both formats
                    try:
                        month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                        if month > 12:  # Likely DD/MM/YYYY
                            day, month = month, day
                        birth_datetime = datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc)
                    except ValueError:
                        pass
                
                # Year only
                elif len(groups) == 1:
                    year = int(groups[0])
                    birth_datetime = datetime(year, 6, 15, 12, 0, 0, tzinfo=timezone.utc)  # Default to mid-year
                
                if birth_datetime:
                    break
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing date from pattern {pattern}: {e}")
                continue
    
    # Extract time separately if not captured in date (e.g., "7:20pm", "7:20 pm", "7:20am")
    if birth_datetime:
        time_patterns = [
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)\b',
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)\b',
            r'\bat\s+(\d{1,2}):(\d{2})\s*(am|pm)?\b',
        ]
        for pattern in time_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                try:
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                    am_pm = match.group(3).lower() if len(match.groups()) > 2 and match.group(3) else None
                    
                    # Convert to 24-hour format
                    if am_pm == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    
                    # Update birth_datetime with extracted time
                    birth_datetime = birth_datetime.replace(hour=hour, minute=minute)
                    logger.info(f"Extracted time from query: {hour}:{minute:02d}")
                    break
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing time from pattern {pattern}: {e}")
                    continue
    
    # Extract location patterns
    location_patterns = [
        # Coordinates: 40.7128, -74.0060 or lat: 40.7128, lng: -74.0060
        r'(?:lat|latitude)[:\s]*(-?\d+\.?\d*)[,\s]+(?:lng|longitude|lon)[:\s]*(-?\d+\.?\d*)',
        r'(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)',  # Simple coordinate pair
        # City names after keywords: "in New York", "born in London"
        r'(?:born|birth|in)\s+(?:the\s+)?(?:city\s+of\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        # City names anywhere: "dallas texas", "Dallas, Texas" (common city patterns)
        r'\b(dallas|new york|london|los angeles|chicago|san francisco|paris|tokyo|sydney|toronto|miami|boston|seattle)(?:\s*,\s*(?:texas|tx|california|ca|new york|ny|illinois|il|florida|fl|massachusetts|ma|washington|wa))?\b',
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            groups = match.groups()
            
            # Coordinate format
            if len(groups) == 2:
                try:
                    lat = float(groups[0])
                    lng = float(groups[1])
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        location = (lat, lng)
                        break
                except ValueError:
                    pass
            
            # City name - would need geocoding service (placeholder for now)
            elif len(groups) == 1:
                city_name = groups[0]
                # Common cities mapping (simplified - in production would use geocoding API)
                city_coords = {
                    'new york': (40.7128, -74.0060),
                    'london': (51.5074, -0.1278),
                    'los angeles': (34.0522, -118.2437),
                    'chicago': (41.8781, -87.6298),
                    'san francisco': (37.7749, -122.4194),
                    'paris': (48.8566, 2.3522),
                    'tokyo': (35.6762, 139.6503),
                    'sydney': (-33.8688, 151.2093),
                    'dallas': (32.7767, -96.7970),
                    'dallas texas': (32.7767, -96.7970),
                    'dallas, texas': (32.7767, -96.7970),
                    'dallas tx': (32.7767, -96.7970),
                    'dallas, tx': (32.7767, -96.7970),
                }
                city_lower = city_name.lower().strip()
                # Also try matching with state/country suffix removed
                city_base = city_lower.split(',')[0].strip().split()[0]  # Get first word (e.g., "dallas" from "dallas texas")
                if city_lower in city_coords:
                    location = city_coords[city_lower]
                    logger.info(f"Matched city: {city_lower} -> {location}")
                    break
                elif city_base in city_coords:
                    location = city_coords[city_base]
                    logger.info(f"Matched city base: {city_base} -> {location}")
                    break
    
    if birth_datetime or location:
        return {
            "birth_datetime": birth_datetime,
            "location": location
        }
    
    return None


def parse_birth_date(date_str: str) -> Optional[datetime]:
    """
    Parse birth date string into datetime object.
    
    Supports formats:
    - ISO: 1990-03-15 or 1990-03-15T14:30:00Z
    - US: 03/15/1990
    - European: 15/03/1990
    - Natural: March 15, 1990
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        datetime object with UTC timezone or None if parsing fails
    """
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Try ISO format first
    iso_formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in iso_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    
    # Try US format (MM/DD/YYYY)
    try:
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    
    # Try European format (DD/MM/YYYY)
    try:
        dt = datetime.strptime(date_str, "%d/%m/%Y")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    
    # Try natural language month names
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    for month_name, month_num in month_names.items():
        pattern = rf'{month_name}\s+(\d{{1,2}}),?\s+(\d{{4}})'
        match = re.search(pattern, date_str.lower())
        if match:
            day, year = int(match.group(1)), int(match.group(2))
            try:
                return datetime(year, month_num, day, 12, 0, 0, tzinfo=timezone.utc)
            except ValueError:
                continue
    
    logger.warning(f"Could not parse birth date: {date_str}")
    return None


def parse_location(location_input: Any) -> Optional[Tuple[float, float]]:
    """
    Parse location input into (latitude, longitude) tuple.
    
    Supports:
    - Tuple/list: (40.7128, -74.0060) or [40.7128, -74.0060]
    - String: "40.7128,-74.0060" or "40.7128, -74.0060"
    - Dict: {"lat": 40.7128, "lng": -74.0060}
    - City name: "New York" (uses common city mapping)
    
    Args:
        location_input: Location in various formats
        
    Returns:
        (latitude, longitude) tuple or None if parsing fails
    """
    if not location_input:
        return None
    
    # If already a tuple/list
    if isinstance(location_input, (tuple, list)):
        if len(location_input) >= 2:
            try:
                lat, lng = float(location_input[0]), float(location_input[1])
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    return (lat, lng)
            except (ValueError, TypeError):
                pass
    
    # If dict with lat/lng keys
    if isinstance(location_input, dict):
        lat = location_input.get("lat") or location_input.get("latitude")
        lng = location_input.get("lng") or location_input.get("longitude") or location_input.get("lon")
        if lat is not None and lng is not None:
            try:
                lat, lng = float(lat), float(lng)
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    return (lat, lng)
            except (ValueError, TypeError):
                pass
    
    # If string, try parsing as coordinates
    if isinstance(location_input, str):
        # Try coordinate format: "40.7128,-74.0060"
        coord_match = re.match(r'(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)', location_input.strip())
        if coord_match:
            try:
                lat, lng = float(coord_match.group(1)), float(coord_match.group(2))
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    return (lat, lng)
            except ValueError:
                pass
        
        # Try city name mapping
        city_coords = {
            'new york': (40.7128, -74.0060),
            'london': (51.5074, -0.1278),
            'los angeles': (34.0522, -118.2437),
            'chicago': (41.8781, -87.6298),
            'san francisco': (37.7749, -122.4194),
            'paris': (48.8566, 2.3522),
            'tokyo': (35.6762, 139.6503),
            'sydney': (-33.8688, 151.2093),
            'toronto': (43.6532, -79.3832),
            'miami': (25.7617, -80.1918),
            'boston': (42.3601, -71.0589),
            'seattle': (47.6062, -122.3321),
            'dallas': (32.7767, -96.7970),
            'dallas texas': (32.7767, -96.7970),
            'dallas, texas': (32.7767, -96.7970),
            'dallas tx': (32.7767, -96.7970),
            'dallas, tx': (32.7767, -96.7970),
        }
        
        city_lower = location_input.lower().strip()
        if city_lower in city_coords:
            return city_coords[city_lower]
    
    logger.warning(f"Could not parse location: {location_input}")
    return None

