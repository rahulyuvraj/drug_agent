"""
Text Utilities - Helper functions for text processing.
"""

import hashlib
import re


def generate_doc_id(*parts) -> str:
    """Generate a unique document ID from parts."""
    combined = "_".join(str(p) for p in parts if p)
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def truncate_text(text: str, max_length: int = 512) -> str:
    """Truncate text to max length at word boundary."""
    if not text or len(text) <= max_length:
        return text or ""
    truncated = text[:max_length].rsplit(" ", 1)[0]
    return truncated + "..."


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    return text