"""
Error message formatting for user-friendly error display.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def format_error_for_user(error: Exception, context: str = "general") -> str:
    """
    Format an error for user-friendly display.
    
    Args:
        error: The exception that occurred
        context: The context in which the error occurred (e.g., "secretary_failure", "provider_initialization")
    
    Returns:
        A user-friendly error message string
    """
    error_str = str(error)
    error_type = type(error).__name__
    
    # Log the full error for debugging
    logger.error(f"Error in context '{context}': {error_type}: {error_str}")
    
    # Context-specific messages
    context_messages = {
        "secretary_failure": "I encountered an issue processing your request.",
        "provider_initialization": "There was a problem connecting to the AI provider.",
        "chat_mode_error": "An error occurred in chat mode.",
        "websocket_error": "Connection issue detected.",
        "general": "Something unexpected happened."
    }
    
    base_message = context_messages.get(context, context_messages["general"])
    
    # Check for common error types and provide helpful messages
    if "quota" in error_str.lower() or "rate limit" in error_str.lower():
        return f"{base_message} The API rate limit has been reached. Please try again later."
    
    if "api key" in error_str.lower() or "authentication" in error_str.lower():
        return f"{base_message} There may be an issue with the API key configuration."
    
    if "connection" in error_str.lower() or "timeout" in error_str.lower():
        return f"{base_message} Unable to connect to the service. Please check your network connection."
    
    if "not found" in error_str.lower():
        return f"{base_message} The requested resource was not found."
    
    # For other errors, provide a generic but informative message
    # Truncate very long error messages
    if len(error_str) > 200:
        error_str = error_str[:200] + "..."
    
    return f"{base_message} Error details: {error_str}"

