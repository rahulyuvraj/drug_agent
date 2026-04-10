"""
JSON Parser Module
==================

Parses gene JSON files and extracts relevant drug, disease, and pathway information.
Handles file walking, schema validation, and error recovery.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedGeneData:
    """Container for parsed gene data from JSON file."""
    
    # File metadata
    file_path: str
    gene_symbol: str
    
    # Gene information
    gene_name: str = ""
    gene_category: str = ""
    gene_aliases: List[str] = field(default_factory=list)
    gene_summary: str = ""
    
    # Drug/Compound data
    unified_drugs: List[Dict[str, Any]] = field(default_factory=list)
    unified_compounds: List[Dict[str, Any]] = field(default_factory=list)
    compounds: List[Dict[str, Any]] = field(default_factory=list)
    
    # Disease associations
    malacards_disorders: List[Dict[str, Any]] = field(default_factory=list)
    malacards_inferred_disorders: List[Dict[str, Any]] = field(default_factory=list)
    uniprot_disorders: List[Dict[str, Any]] = field(default_factory=list)
    gwas_phenotypes: List[Dict[str, Any]] = field(default_factory=list)
    hpo_terms: List[Dict[str, Any]] = field(default_factory=list)
    phenotypes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pathway data
    pathways: List[Dict[str, Any]] = field(default_factory=list)
    super_pathways: List[Dict[str, Any]] = field(default_factory=list)
    
    # Functional annotations
    molecular_functions: List[Dict[str, Any]] = field(default_factory=list)
    biological_processes: List[Dict[str, Any]] = field(default_factory=list)
    cellular_components: List[Dict[str, Any]] = field(default_factory=list)
    
    # Protein interactions
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Expression data
    tissue_expression: List[Dict[str, Any]] = field(default_factory=list)
    differential_expression: List[Dict[str, Any]] = field(default_factory=list)
    
    # External identifiers
    external_ids: Dict[str, str] = field(default_factory=dict)
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def has_drug_data(self) -> bool:
        """Check if gene has any drug/compound data."""
        return bool(self.unified_drugs or self.unified_compounds or self.compounds)
    
    def has_disease_data(self) -> bool:
        """Check if gene has any disease association data."""
        return bool(
            self.malacards_disorders or self.malacards_inferred_disorders or
            self.uniprot_disorders or self.gwas_phenotypes
        )
    
    def has_pathway_data(self) -> bool:
        """Check if gene has any pathway data."""
        return bool(self.pathways or self.super_pathways)
    
    def get_all_drugs(self) -> List[Dict[str, Any]]:
        """Get all drugs/compounds combined."""
        all_drugs = []
        for drug in self.unified_drugs:
            drug["source"] = "UnifiedDrugs"
            all_drugs.append(drug)
        for compound in self.unified_compounds:
            compound["source"] = "UnifiedCompounds"
            all_drugs.append(compound)
        for compound in self.compounds:
            compound["source"] = "Compounds"
            all_drugs.append(compound)
        return all_drugs
    
    def get_all_diseases(self) -> List[Dict[str, Any]]:
        """Get all disease associations combined."""
        all_diseases = []
        for disorder in self.malacards_disorders:
            disorder["source"] = "MalaCardsDisorders"
            disorder["association_type"] = "Direct"
            all_diseases.append(disorder)
        for disorder in self.malacards_inferred_disorders:
            disorder["source"] = "MalaCardsInferredDisorders"
            disorder["association_type"] = "Inferred"
            all_diseases.append(disorder)
        for disorder in self.uniprot_disorders:
            disorder["source"] = "UniProtDisorders"
            disorder["association_type"] = "UniProt"
            all_diseases.append(disorder)
        for phenotype in self.gwas_phenotypes:
            phenotype["source"] = "GWASPhenotypes"
            phenotype["association_type"] = "GWAS"
            all_diseases.append(phenotype)
        return all_diseases


class JSONParser:
    """Parser for gene JSON files."""
    
    def __init__(self, validate_schema: bool = True, strict_mode: bool = False, skip_empty: bool = True):
        self.validate_schema = validate_schema
        self.strict_mode = strict_mode
        self.skip_empty = skip_empty
        self.stats = {"files_processed": 0, "files_with_drugs": 0, "files_with_diseases": 0,
                      "files_with_pathways": 0, "files_skipped": 0, "files_errored": 0}
    
    def walk_directory(self, base_directory: str) -> Generator[Path, None, None]:
        """Walk through gene directories and yield JSON file paths."""
        base_path = Path(base_directory)
        if not base_path.exists():
            raise FileNotFoundError(f"Directory not found: {base_directory}")
        
        has_az_folders = any((base_path / letter).is_dir() for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        if has_az_folders:
            for letter in sorted(os.listdir(base_path)):
                letter_path = base_path / letter
                if letter_path.is_dir():
                    for json_file in sorted(letter_path.glob("*.json")):
                        yield json_file
        else:
            for json_file in sorted(base_path.glob("**/*.json")):
                yield json_file
    
    def parse_file(self, file_path: Path) -> Optional[ParsedGeneData]:
        """Parse a single JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.stats["files_processed"] += 1
            gene_symbol = self._extract_gene_symbol(file_path, data)
            
            if not gene_symbol:
                self.stats["files_skipped"] += 1
                return None
            
            parsed = self._parse_gene_data(data, file_path, gene_symbol)
            
            if self.skip_empty and not (parsed.has_drug_data() or parsed.has_disease_data() or parsed.has_pathway_data()):
                self.stats["files_skipped"] += 1
                return None
            
            if parsed.has_drug_data():
                self.stats["files_with_drugs"] += 1
            if parsed.has_disease_data():
                self.stats["files_with_diseases"] += 1
            if parsed.has_pathway_data():
                self.stats["files_with_pathways"] += 1
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            self.stats["files_errored"] += 1
            return None
    
    def _extract_gene_symbol(self, file_path: Path, data: Dict[str, Any]) -> Optional[str]:
        """Extract gene symbol from file or data."""
        gene_field = data.get("Gene", [])
        if gene_field and isinstance(gene_field, list) and len(gene_field) > 0:
            symbol = gene_field[0].get("Symbol")
            if symbol:
                return symbol
        return file_path.stem
    
    def _parse_gene_data(self, data: Dict[str, Any], file_path: Path, gene_symbol: str) -> ParsedGeneData:
        """Parse gene data from JSON dictionary."""
        gene_field = data.get("Gene", [{}])
        gene_info = gene_field[0] if gene_field else {}
        aliases = [a.get("Value", "") for a in data.get("Aliases", []) if a.get("Value")]
        summaries = data.get("Summaries", [])
        summary = summaries[0].get("Summary", "") if summaries else ""
        
        ext_ids = {}
        for ext in data.get("ExternalIdentifiers", []):
            if ext.get("Source") and ext.get("Value"):
                ext_ids[ext["Source"]] = ext["Value"]
        
        return ParsedGeneData(
            file_path=str(file_path), gene_symbol=gene_symbol,
            gene_name=gene_info.get("Name", ""), gene_category=gene_info.get("Category", ""),
            gene_aliases=aliases, gene_summary=summary,
            unified_drugs=data.get("UnifiedDrugs", []),
            unified_compounds=data.get("UnifiedCompounds", []),
            compounds=data.get("Compounds", []),
            malacards_disorders=data.get("MalaCardsDisorders", []),
            malacards_inferred_disorders=data.get("MalaCardsInferredDisorders", []),
            uniprot_disorders=data.get("UniProtDisorders", []),
            gwas_phenotypes=data.get("GWASPhenotypes", []),
            hpo_terms=data.get("HumanPhenotypeOntology", []),
            phenotypes=data.get("Phenotypes", []),
            pathways=data.get("Pathways", []),
            super_pathways=data.get("SuperPathway", []),
            molecular_functions=data.get("MolecularFunctions", []),
            biological_processes=data.get("BiologicalProcesses", []),
            cellular_components=data.get("CellularComponents", []),
            interactions=data.get("Interactions", []),
            tissue_expression=data.get("TissueExpression", []),
            differential_expression=data.get("DifferentialExpression", []),
            external_ids=ext_ids,
        )
    
    def parse_directory(self, base_directory: str, progress_callback=None) -> Generator[ParsedGeneData, None, None]:
        """Parse all JSON files in a directory."""
        file_count = 0
        for file_path in self.walk_directory(base_directory):
            parsed = self.parse_file(file_path)
            if parsed is not None:
                yield parsed
            file_count += 1
            if progress_callback and file_count % 100 == 0:
                progress_callback(file_count, self.stats)
    
    def get_statistics(self) -> Dict[str, int]:
        return dict(self.stats)
