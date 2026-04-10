"""
Gene Resolver Utility (Dynamic)
===============================

Resolves gene aliases to official symbols.
Fully dynamic - learns from data and external sources.
No hardcoded gene information.
"""

import logging
from typing import List, Dict, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class GeneResolver:
    """
    Resolves gene aliases to official HGNC symbols.
    
    Fully dynamic - can load mappings from:
    - Knowledge base during ingestion
    - External mapping files (e.g., HGNC)
    - APIs (if configured)
    """
    
    def __init__(self, mappings: Optional[Dict[str, str]] = None):
        """
        Initialize gene resolver.
        
        Args:
            mappings: Optional initial mappings (alias -> official symbol).
        """
        # Mappings: uppercase alias -> official symbol
        self.alias_to_symbol: Dict[str, str] = {}
        
        # Reverse mappings: symbol -> set of aliases
        self.symbol_to_aliases: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.seen_genes: Set[str] = set()
        
        # Load initial mappings if provided
        if mappings:
            self.load_mappings(mappings)
    
    def load_mappings(self, mappings: Dict[str, str]):
        """
        Load gene mappings from a dictionary.
        
        Args:
            mappings: Dictionary of alias -> official symbol
        """
        for alias, symbol in mappings.items():
            self.add_mapping(alias, symbol)
    
    def load_mappings_from_file(self, filepath: str):
        """
        Load mappings from a file (JSON or CSV).
        
        Supports HGNC-style files with columns:
        - symbol, alias (or Symbol, Alias)
        
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
                    # Handle both flat dict and nested format
                    if isinstance(mappings, dict):
                        first_value = next(iter(mappings.values()), None)
                        if isinstance(first_value, list):
                            # Format: {symbol: [aliases]}
                            for symbol, aliases in mappings.items():
                                for alias in aliases:
                                    self.add_mapping(alias, symbol)
                        else:
                            # Format: {alias: symbol}
                            self.load_mappings(mappings)
            
            elif path.suffix == '.csv':
                import csv
                with open(path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        symbol = row.get('symbol', row.get('Symbol', row.get('HGNC Symbol', '')))
                        # Aliases might be comma-separated
                        aliases_str = row.get('alias', row.get('Alias', row.get('Previous Symbols', '')))
                        if symbol:
                            symbol = symbol.strip().upper()
                            if aliases_str:
                                for alias in aliases_str.split(','):
                                    alias = alias.strip()
                                    if alias:
                                        self.add_mapping(alias, symbol)
            
            logger.info(f"Loaded gene mappings from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
    
    def add_mapping(self, alias: str, symbol: str):
        """Add a single alias to symbol mapping."""
        if not alias or not symbol:
            return
        
        alias_upper = alias.strip().upper()
        symbol_upper = symbol.strip().upper()
        
        self.alias_to_symbol[alias_upper] = symbol_upper
        self.symbol_to_aliases[symbol_upper].add(alias_upper)
    
    def learn_from_data(self, symbol: str, aliases: List[str] = None):
        """
        Learn gene symbols and aliases from data during ingestion.
        
        Args:
            symbol: Official gene symbol
            aliases: Optional list of aliases
        """
        if not symbol:
            return
        
        symbol_upper = symbol.strip().upper()
        self.seen_genes.add(symbol_upper)
        self.symbol_to_aliases[symbol_upper].add(symbol_upper)
        
        if aliases:
            for alias in aliases:
                if alias:
                    self.add_mapping(alias, symbol_upper)
    
    def resolve(self, gene_name: str) -> str:
        """
        Resolve a gene name to its official symbol.
        
        Args:
            gene_name: Gene name (may be alias or symbol).
            
        Returns:
            Official gene symbol (uppercase).
        """
        if not gene_name:
            return ""
        
        upper = gene_name.strip().upper()
        
        # Check if it's a known alias
        if upper in self.alias_to_symbol:
            return self.alias_to_symbol[upper]
        
        # Return as-is (uppercase)
        self.seen_genes.add(upper)
        return upper
    
    def get_aliases(self, gene_symbol: str) -> List[str]:
        """Get all aliases for a gene symbol."""
        symbol = gene_symbol.strip().upper()
        aliases = list(self.symbol_to_aliases.get(symbol, set()))
        # Always include the symbol itself
        if symbol not in aliases:
            aliases.insert(0, symbol)
        return aliases
    
    def get_all_names(self, gene_name: str) -> List[str]:
        """Get all names (symbol + aliases) for a gene."""
        symbol = self.resolve(gene_name)
        return self.get_aliases(symbol)
    
    def are_same_gene(self, gene1: str, gene2: str) -> bool:
        """Check if two gene names refer to the same gene."""
        return self.resolve(gene1) == self.resolve(gene2)
    
    def resolve_list(self, gene_names: List[str]) -> List[str]:
        """Resolve a list of gene names to official symbols."""
        resolved = []
        seen = set()
        for name in gene_names:
            symbol = self.resolve(name)
            if symbol and symbol not in seen:
                resolved.append(symbol)
                seen.add(symbol)
        return resolved
    
    def expand_gene_list(self, gene_symbols: List[str]) -> Set[str]:
        """Expand gene symbols to include all aliases."""
        expanded = set()
        for symbol in gene_symbols:
            expanded.update(self.get_all_names(symbol))
        return expanded
    
    def get_statistics(self) -> Dict[str, int]:
        """Get resolver statistics."""
        return {
            "total_aliases": len(self.alias_to_symbol),
            "unique_symbols": len(self.symbol_to_aliases),
            "genes_seen": len(self.seen_genes),
        }
    
    def export_mappings(self) -> Dict[str, List[str]]:
        """Export all mappings for persistence."""
        return {
            symbol: list(aliases)
            for symbol, aliases in self.symbol_to_aliases.items()
        }
