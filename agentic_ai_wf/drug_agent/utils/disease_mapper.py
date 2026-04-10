"""
Disease Mapper Utility (Dynamic)
================================

Maps disease name variations to canonical forms.
Fully dynamic - learns from data and external sources.
No hardcoded disease information.
"""

import re
import logging
from typing import List, Dict, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class DiseaseMapper:
    """
    Maps disease name variations to canonical forms.
    
    Fully dynamic - can load mappings from:
    - Knowledge base during ingestion
    - External mapping files
    - APIs (if configured)
    """
    
    def __init__(self, mappings: Optional[Dict[str, str]] = None):
        """
        Initialize disease mapper.
        
        Args:
            mappings: Optional initial mappings (alias -> canonical).
        """
        # Mappings: lowercase alias -> canonical name
        self.mappings: Dict[str, str] = {}
        
        # Reverse mappings: canonical -> set of aliases
        self.canonical_to_aliases: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.seen_diseases: Set[str] = set()
        
        # Load initial mappings if provided
        if mappings:
            self.load_mappings(mappings)
    
    def load_mappings(self, mappings: Dict[str, str]):
        """
        Load disease mappings from a dictionary.
        
        Args:
            mappings: Dictionary of alias -> canonical name
        """
        for alias, canonical in mappings.items():
            self.add_mapping(alias, canonical)
    
    def load_mappings_from_file(self, filepath: str):
        """
        Load mappings from a file (JSON or CSV).
        
        Args:
            filepath: Path to mapping file
        """
        import json
        from pathlib import Path
        
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Mapping file not found: {filepath}")
            return
        
        try:
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    mappings = json.load(f)
                    self.load_mappings(mappings)
            elif path.suffix == '.csv':
                import csv
                with open(path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        alias = row.get('alias', row.get('Alias', ''))
                        canonical = row.get('canonical', row.get('Canonical', ''))
                        if alias and canonical:
                            self.add_mapping(alias, canonical)
            
            logger.info(f"Loaded disease mappings from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
    
    def add_mapping(self, alias: str, canonical: str):
        """Add a single mapping."""
        if not alias or not canonical:
            return
        
        alias_lower = alias.strip().lower()
        canonical_clean = canonical.strip()
        
        self.mappings[alias_lower] = canonical_clean
        self.canonical_to_aliases[canonical_clean].add(alias_lower)
    
    def learn_from_data(self, disease_name: str, aliases: List[str] = None):
        """
        Learn disease names and aliases from data during ingestion.
        
        Args:
            disease_name: Primary disease name
            aliases: Optional list of aliases
        """
        if not disease_name:
            return
        
        canonical = self.normalize(disease_name)
        self.seen_diseases.add(canonical)
        
        if aliases:
            for alias in aliases:
                if alias:
                    self.add_mapping(alias, canonical)
    
    def normalize(self, disease_name: str) -> str:
        """
        Normalize a disease name to its canonical form.
        
        Args:
            disease_name: Input disease name.
            
        Returns:
            Canonical disease name.
        """
        if not disease_name:
            return ""
        
        # Clean and prepare for lookup
        cleaned = disease_name.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        lookup_key = cleaned.lower()
        
        # Check for exact mapping
        if lookup_key in self.mappings:
            return self.mappings[lookup_key]
        
        # Check for partial matches (substring)
        for alias, canonical in self.mappings.items():
            if alias in lookup_key or lookup_key in alias:
                return canonical
        
        # Default: Title case the original
        normalized = cleaned.title()
        self.seen_diseases.add(normalized)
        return normalized
    
    def get_aliases(self, canonical_name: str) -> List[str]:
        """Get all aliases for a canonical disease name."""
        return list(self.canonical_to_aliases.get(canonical_name, set()))
    
    def get_search_terms(self, disease_name: str) -> List[str]:
        """Get all search terms for a disease (canonical + aliases)."""
        canonical = self.normalize(disease_name)
        terms = [canonical]
        terms.extend(self.get_aliases(canonical))
        return list(set(terms))
    
    def is_same_disease(self, disease1: str, disease2: str) -> bool:
        """Check if two disease names refer to the same disease."""
        return self.normalize(disease1) == self.normalize(disease2)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get mapper statistics."""
        return {
            "total_mappings": len(self.mappings),
            "unique_canonical": len(self.canonical_to_aliases),
            "diseases_seen": len(self.seen_diseases),
        }
    
    def export_mappings(self) -> Dict[str, List[str]]:
        """Export all mappings for persistence."""
        return {
            canonical: list(aliases)
            for canonical, aliases in self.canonical_to_aliases.items()
        }
