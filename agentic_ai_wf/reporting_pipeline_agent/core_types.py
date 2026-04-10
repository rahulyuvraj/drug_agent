#!/usr/bin/env python3
"""
=============================================================================
REPORTING PIPELINE AGENT - Core Types Module
=============================================================================
Foundation layer containing all enums, dataclasses, constants, and shared types.

This module has ZERO internal dependencies to prevent circular imports.
All other modules import from this one.

Contents:
    - Enums (DeconvolutionMethod, ContradictionSeverity)
    - Status classes (BiomarkerStatus, GeneCategory)
    - Dataclasses (DeconvolutionConfig, DeconvolutionResult, GeneMapping, etc.)
    - Global constants and thresholds
    - Utility functions for column detection and gene classification
=============================================================================
"""

import re
import math
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# =============================================================================
# GLOBAL LLM CONFIGURATION
# =============================================================================
# Temperature setting for all LLM calls in the pipeline.
# 0.0 = deterministic, reproducible outputs
# Adjust this single value to change temperature across the entire pipeline.
LLM_TEMPERATURE = 0.0

# =============================================================================
# DEG FILTERING THRESHOLDS
# =============================================================================
# These thresholds define what constitutes a meaningful differential expression.
# Genes passing both significance AND effect-size thresholds are "Significant DEGs".
# Genes with p<0.05 but low |log2FC| are labeled as "Trend DEGs".
# 
# METHODOLOGY DOCUMENTATION:
# - log2FC >= 0.58 corresponds to 1.5-fold change (2^0.58 ≈ 1.5)
# - log2FC >= 1.0 corresponds to 2-fold change (2^1.0 = 2.0)
# - Reference: Standard practice in differential expression analysis
# =============================================================================

DEG_ADJ_PVALUE_THRESHOLD = 0.05  # Statistical significance threshold (FDR-corrected)
DEG_LOG2FC_THRESHOLD = 0.58     # Minimum |log2FC| for "Significant DEG" (1.5-fold)
DEG_HIGH_CONFIDENCE_LOG2FC = 1.0  # High confidence effect size threshold (2-fold)
DEG_TREND_THRESHOLD = 0.3       # Minimum |log2FC| for "Trend DEG" (below this = noise)

# =============================================================================
# DRUG PRIORITY SCORE THRESHOLDS
# =============================================================================
# These thresholds determine drug priority classification
# Adjust these values to change what constitutes High/Moderate/Low priority
# =============================================================================
DRUG_HIGH_PRIORITY_THRESHOLD = 55   # Score >= this = High Priority
DRUG_MODERATE_PRIORITY_THRESHOLD = 30  # Score >= this (but < high) = Moderate Priority

# =============================================================================
# DECONVOLUTION THRESHOLDS
# =============================================================================
# METHODOLOGY DOCUMENTATION:
# These are DEFAULT thresholds; disease-specific calibration is recommended.
# Thresholds can be dynamically adjusted based on disease and sample type.
# 
# Cell fraction interpretation:
# - High: Cell type is dominant in the composition (clinically meaningful)
# - Moderate: Cell type is present at notable levels
# - Low: Cell type detected but at background levels
# 
# CIBERSORT: Fractions sum to 1.0 (100%), thresholds are absolute fractions.
# xCell: Enrichment scores are unbounded, thresholds are relative.
# BisqueRNA: Proportions similar to CIBERSORT.
# =============================================================================
XCELL_LOW_SIGNAL_THRESHOLD = 0.01  # xCell enrichment score threshold for "low signal"
CIBERSORT_LOW_SIGNAL_THRESHOLD = 0.02  # CIBERSORT fraction threshold (2%)
CIBERSORT_HIGH_THRESHOLD = 0.10  # CIBERSORT: >10% = "High" enrichment
CIBERSORT_MODERATE_THRESHOLD = 0.02  # CIBERSORT: >2% = "Moderate" enrichment
BISQUE_LOW_SIGNAL_THRESHOLD = 0.02  # BisqueRNA proportion threshold (2%)
BISQUE_HIGH_THRESHOLD = 0.10  # BisqueRNA: >10% = "High"
BISQUE_MODERATE_THRESHOLD = 0.02  # BisqueRNA: >2% = "Moderate"
XCELL_MIN_HIGH_ENRICHED = 1  # Minimum number of "High" enriched cell types

# =============================================================================
# DRUG VALIDATION FLAGS (FIX 9)
# =============================================================================
# Drug Agent Service is now integrated — validation active via 15 Qdrant collections.
# =============================================================================
DRUG_VALIDATION_PENDING = False
DRUG_VALIDATION_DISCLAIMER = (
    "Note: Drug recommendations are pending validation by pharmacological database integration. "
    "Some listed compounds may be investigational, incorrectly identified, or placeholder entities. "
    "Do not use for prescribing decisions without independent verification."
)

# =============================================================================
# CLINICAL LANGUAGE TEMPLATES
# =============================================================================
# RNA expression CANNOT determine drug eligibility. These templates ensure
# appropriate clinical hedging in all report outputs.
#
# REQUIREMENTS:
# - Do NOT treat RNA up/down as "eligibility"
# - Never write "Eligible / Not eligible" based on RNA expression alone
# - Use language like: "suggestive," "supports hypothesis," "requires confirmatory testing"

CLINICAL_LANGUAGE_TEMPLATES = {
    # When RNA expression supports a finding (use these instead of "eligible")
    'supportive': [
        "suggestive of",
        "supports the hypothesis of",
        "consistent with",
        "may indicate",
        "expression pattern aligns with",
        "transcriptomic evidence supports",
    ],
    # When RNA expression is discordant
    'not_supportive': [
        "does not support",
        "expression pattern inconsistent with",
        "transcriptomic data does not suggest",
        "limited RNA-level evidence for",
    ],
    'confirmation_required': [
        "requires confirmatory testing", "clinical confirmation required",
        "confirmatory testing recommended", "protein-level assessment needed",
    ],
    'forbidden_phrases': [
        "eligible for", "not eligible for", "qualifies for", "disqualifies from",
        "indicated for", "contraindicated", "approved for", "should receive",
        "should not receive", "will respond to", "will not respond to",
    ]
}

# DEPRECATED: Validation requirements are now fetched dynamically via LLM
# This empty dict is kept for backward compatibility only
# Use get_biomarker_confirmation_requirements_dynamic() instead
BIOMARKER_CONFIRMATION_REQUIREMENTS = {}

# =============================================================================
# ARTIFACT GENE PATTERNS
# =============================================================================
# These genes may appear with high fold-change but should be:
# - Labeled as "biologically interesting but clinically uncertain"
# - Visually separated from canonical drivers in figures
# - Not prioritized as therapeutic targets
# =============================================================================

# Patterns to match artifact/non-clinically-actionable genes
ARTIFACT_GENE_PATTERNS = [
    # T-cell receptor pseudogenes and variable genes
    "TRD-GTC2",    # T-cell receptor delta pseudogenes
    "TRD-GTC1",
    "TRA-",        # T-cell receptor alpha
    "TRB-",        # T-cell receptor beta  
    "TRG-",        # T-cell receptor gamma
    "TRDV",        # T-cell receptor delta variable
    "TRAV",        # T-cell receptor alpha variable
    "TRBV",        # T-cell receptor beta variable
    "TRGV",        # T-cell receptor gamma variable
    "TRDC",        # T-cell receptor delta constant
    "TRAC",        # T-cell receptor alpha constant
    
    # Histone variants - chromatin remodeling, not established cancer drivers
    "H2BC",        # Histone H2B cluster (e.g., H2BC1, H2BC21)
    "H2AC",        # Histone H2A cluster
    "H1-",         # Histone H1 linker
    "H3C",         # Histone H3 cluster
    "H4C",         # Histone H4 cluster
    
    # Olfactory receptors - sensory genes, uncertain clinical relevance
    "OR1",         # Olfactory receptor family 1
    "OR2",         # Olfactory receptor family 2
    "OR3",         # Olfactory receptor family 3
    "OR4",         # Olfactory receptor family 4
    "OR5",         # Olfactory receptor family 5
    "OR6",         # Olfactory receptor family 6
    "OR7",         # Olfactory receptor family 7
    "OR8",         # Olfactory receptor family 8
    "OR9",         # Olfactory receptor family 9
    "OR10",        # Olfactory receptor family 10
    "OR11",        # Olfactory receptor family 11
    "OR12",        # Olfactory receptor family 12
    "OR13",        # Olfactory receptor family 13
    "OR14",        # Olfactory receptor family 14
    "OR51",        # Olfactory receptor family 51
    "OR52",        # Olfactory receptor family 52
    
    # Uncharacterized loci and pseudogenes
    "LOC",         # Uncharacterized loci
    "LINC",        # Long intergenic non-coding RNAs
]

# Known artifact genes (explicit list)
KNOWN_ARTIFACT_GENES = {
    # TRD-GTC2 family
    "TRD-GTC2-11", "TRD-GTC2-10", "TRD-GTC2-9", "TRD-GTC2-8",
    "TRD-GTC2-7", "TRD-GTC2-6", "TRD-GTC2-5", "TRD-GTC2-4",
    "TRD-GTC2-3", "TRD-GTC2-2", "TRD-GTC2-1", "TRD-GTC2-12",
    "TRD-GTC1-1", "TRD-GTC1-2",
    # Common histone variants with high expression but uncertain clinical role
    "H2BC1", "H2BC21", "H2BC12", "H2BC3", "H2BC4", "H2BC5",
    "H2AC4", "H2AC6", "H2AC7", "H2AC8", "H2AC11", "H2AC12",
}

# Backward compatibility alias
ARTIFACT_GENES = KNOWN_ARTIFACT_GENES

# DEPRECATED: These empty dictionaries are kept for backward compatibility.
# All data is now fetched dynamically via LLM (see DynamicKnowledgeManager).
# If LLM is unavailable, the pipeline gracefully degrades with minimal fallbacks.
BASE_SIGNATURE_GENES = {}  # DEPRECATED: Use build_signatures_from_pathways()
DRUGGABLE_GENES = {}      # DEPRECATED: Use get_druggable_gene_info_dynamic()
DISEASE_BIOMARKERS = {}   # DEPRECATED: Use get_disease_biomarkers_dynamic()
CELL_TYPE_MARKERS = {}    # DEPRECATED: Use get_cell_type_markers_dynamic()
CELL_TYPE_PATHWAYS = {}   # DEPRECATED: Use get_cell_type_pathways_dynamic()

# Drug mechanism information - kept as minimal fallback
DRUG_MECHANISM_INFO = {}  # DEPRECATED: Use DynamicKnowledgeManager.get_drug_mechanism_info()


# =============================================================================
# SAMPLE TYPE CONTEXT SYSTEM (FIX 1)
# =============================================================================
# Sample type affects interpretation of cell composition findings.
# Peripheral blood cannot infer tissue infiltration; tissue samples cannot
# infer systemic immune state. This system provides context-appropriate terminology.
# =============================================================================

@dataclass
class SampleTypeContext:
    """Context for sample type-appropriate interpretation."""
    category: str  # "peripheral", "tissue", "fluid"
    sample_type: str  # Original sample type string
    valid_inferences: List[str] = field(default_factory=list)  # What CAN be inferred
    invalid_inferences: List[str] = field(default_factory=list)  # What CANNOT be inferred
    appropriate_terminology: Dict[str, str] = field(default_factory=dict)  # e.g., {"composition": "circulating immune profile"}
    interpretation_caveat: str = ""  # Caveat to include in report


def get_sample_type_category(sample_type: str) -> str:
    """
    Categorize sample type into peripheral, tissue, or fluid.
    
    This is used to determine appropriate interpretation terminology.
    """
    if not sample_type:
        return "unknown"
    
    sample_lower = sample_type.lower().strip()
    
    # Peripheral blood samples
    peripheral_patterns = [
        "whole blood", "pbmc", "peripheral blood", "blood", "serum", 
        "plasma", "buffy coat", "leukocyte", "wbc"
    ]
    for pattern in peripheral_patterns:
        if pattern in sample_lower:
            return "peripheral"
    
    # Tissue samples
    tissue_patterns = [
        "tissue", "biopsy", "tumor", "synovial tissue", "lesion",
        "surgical", "resection", "ffpe", "frozen tissue", "fresh tissue"
    ]
    for pattern in tissue_patterns:
        if pattern in sample_lower:
            return "tissue"
    
    # Fluid samples (non-blood)
    fluid_patterns = [
        "synovial fluid", "csf", "cerebrospinal", "pleural", "ascites",
        "bronchoalveolar", "bal", "lavage", "effusion"
    ]
    for pattern in fluid_patterns:
        if pattern in sample_lower:
            return "fluid"
    
    return "unknown"


# =============================================================================
# DEG CLASSIFICATION UTILITIES (FIX 4 & FIX 5)
# =============================================================================

@dataclass
class DEGClassification:
    """Classification of a differentially expressed gene."""
    gene: str
    log2fc: float
    adj_pvalue: float
    classification: str  # "significant", "trend", "noise"
    is_significant: bool
    effect_size_label: str  # "high", "moderate", "low", "minimal"
    significance_note: str  # Description for reporting


def classify_deg_significance(gene: str, log2fc: float, adj_pvalue: float) -> DEGClassification:
    """
    Classify a DEG into significant, trend, or noise based on p-value AND effect size.
    
    Classifications:
    - "significant": adj_p < 0.05 AND |log2FC| >= DEG_LOG2FC_THRESHOLD (0.58)
    - "trend": adj_p < 0.05 AND |log2FC| >= DEG_TREND_THRESHOLD (0.3) but < DEG_LOG2FC_THRESHOLD
    - "noise": Does not meet either threshold
    
    Returns DEGClassification with full metadata for reporting.
    """
    abs_log2fc = abs(log2fc) if log2fc else 0
    
    # Determine effect size label
    if abs_log2fc >= DEG_HIGH_CONFIDENCE_LOG2FC:  # >= 1.0 (2-fold)
        effect_size_label = "high"
    elif abs_log2fc >= DEG_LOG2FC_THRESHOLD:  # >= 0.58 (1.5-fold)
        effect_size_label = "moderate"
    elif abs_log2fc >= DEG_TREND_THRESHOLD:  # >= 0.3
        effect_size_label = "low"
    else:
        effect_size_label = "minimal"
    
    # Determine classification
    is_significant = adj_pvalue < DEG_ADJ_PVALUE_THRESHOLD
    
    if is_significant and abs_log2fc >= DEG_LOG2FC_THRESHOLD:
        classification = "significant"
        significance_note = f"Significant DEG (p={adj_pvalue:.2e}, |log2FC|={abs_log2fc:.2f})"
    elif is_significant and abs_log2fc >= DEG_TREND_THRESHOLD:
        classification = "trend"
        significance_note = f"Trend (p={adj_pvalue:.2e}, |log2FC|={abs_log2fc:.2f} - below threshold)"
    elif not is_significant and abs_log2fc >= DEG_LOG2FC_THRESHOLD:
        classification = "trend"
        significance_note = f"Non-significant (p={adj_pvalue:.2e} > 0.05, but high |log2FC|={abs_log2fc:.2f})"
    else:
        classification = "noise"
        significance_note = f"Not significant (p={adj_pvalue:.2e}, |log2FC|={abs_log2fc:.2f})"
    
    return DEGClassification(
        gene=gene,
        log2fc=log2fc,
        adj_pvalue=adj_pvalue,
        classification=classification,
        is_significant=is_significant,
        effect_size_label=effect_size_label,
        significance_note=significance_note
    )


def filter_significant_degs(genes: List[Dict], include_trends: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter genes into significant DEGs and trends.
    
    Args:
        genes: List of gene dicts with 'gene', 'log2fc', 'adj_pvalue' keys
        include_trends: If True, include trends in the primary list with labels
        
    Returns:
        Tuple of (significant_degs, trend_degs)
    """
    significant = []
    trends = []
    
    for g in genes:
        gene_name = g.get('gene', g.get('Gene', ''))
        log2fc = float(g.get('log2fc', g.get('log2FC', g.get('Patient_LFC_mean', 0))) or 0)
        adj_p = float(g.get('adj_pvalue', g.get('padj', g.get('adj.P.Val', 0.05))) or 0.05)
        
        classification = classify_deg_significance(gene_name, log2fc, adj_p)
        
        # Add classification metadata to gene dict
        g_with_class = {**g, 
                        'deg_classification': classification.classification,
                        'significance_note': classification.significance_note,
                        'is_significant_deg': classification.classification == 'significant'}
        
        if classification.classification == 'significant':
            significant.append(g_with_class)
        elif classification.classification == 'trend':
            trends.append(g_with_class)
    
    return significant, trends


# =============================================================================
# ENUMS
# =============================================================================

class DeconvolutionMethod(Enum):
    """Supported deconvolution methods."""
    XCELL = "xCell"
    CIBERSORT = "CIBERSORT"
    BISQUE = "BisqueRNA"
    UNKNOWN = "Deconvolution"


class ContradictionSeverity(Enum):
    """Severity levels for detected contradictions."""
    CRITICAL = "critical"      # Major clinical error - must be flagged prominently
    WARNING = "warning"        # Potentially misleading - should be noted
    INFO = "info"              # Minor inconsistency - for transparency


# =============================================================================
# STATUS CLASSES
# =============================================================================

class BiomarkerStatus:
    """Standardized biomarker status terminology."""
    # Gene is in DEG data but below significance/expression thresholds
    BELOW_THRESHOLD = "Below threshold (detected but not significant)"
    # Gene is not present in the analyzed DEG dataset
    NOT_IN_DATA = "Not in analyzed gene set"
    # Gene was found and is significant
    DETECTED_SIGNIFICANT = "Detected (significant)"
    # Gene was found but not significant
    DETECTED_NOT_SIGNIFICANT = "Detected (not significant)"
    # Test not performed (e.g., requires different assay)
    REQUIRES_DIFFERENT_ASSAY = "Requires protein/DNA assay"
    
    @classmethod
    def get_status(cls, gene: str, gene_dict: Dict, adj_p_threshold: float = 0.05) -> str:
        """Get standardized status for a gene."""
        gene_upper = gene.upper() if gene else ''
        if gene_upper not in gene_dict:
            return cls.NOT_IN_DATA
        
        gene_info = gene_dict[gene_upper]
        adj_p = gene_info.adj_pvalue if hasattr(gene_info, 'adj_pvalue') else \
                gene_info.get('adj_pvalue', gene_info.get('padj', 1.0))
        
        try:
            adj_p = float(adj_p) if adj_p is not None else 1.0
        except (ValueError, TypeError):
            adj_p = 1.0
            
        if adj_p < adj_p_threshold:
            return cls.DETECTED_SIGNIFICANT
        else:
            return cls.DETECTED_NOT_SIGNIFICANT


class GeneCategory:
    """
    Gene classification categories for internal prioritization.
    
    NOTE: These categories are used INTERNALLY for gene ranking and filtering.
    The classification display was removed from the final report but the logic 
    remains active in the background.
    
    Category Definitions:
    - Category 1: Patient & Disease Specific (known disease drivers/markers)
    - Category 2: Patient Specific/Novel (extreme log2FC but not disease drivers)
    - Category 3: Others (any other worthwhile genes)
    - Category 4: Technical/Artifact (TRD-GTC2 family, pseudogenes)
    """
    PATIENT_AND_DISEASE_SPECIFIC = "Category 1: Patient & Disease Specific"
    PATIENT_SPECIFIC_NOVEL = "Category 2: Patient Specific/Novel"
    KNOWN_IN_OTHER_CONDITIONS = "Category 3: Others"
    TECHNICAL_ARTIFACT = "Category 4: Technical/Uncharacterized"  # NEW: For artifact genes
    
    # Short names for tables
    SHORT_NAMES = {
        "Category 1: Patient & Disease Specific": "Disease-Specific",
        "Category 2: Patient Specific/Novel": "Novel/High LFC",
        "Category 3: Others": "Others",
        "Category 4: Technical/Uncharacterized": "Technical/Artifact"
    }


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class DeconvolutionConfig:
    """Configuration and metadata for a deconvolution method."""
    method: DeconvolutionMethod
    display_name: str
    score_column_name: str  # What to call the score column
    score_description: str  # Description for reports
    score_unit: str  # Unit description (%, enrichment score, proportion)
    level_thresholds: Dict[str, float]  # Thresholds for High/Moderate/Low
    presence_thresholds: Dict[str, float]  # Thresholds for presence classification
    interpretation_templates: Dict[str, str]  # Templates for generating interpretations
    reference: str  # Citation/reference for the method
    
    @classmethod
    def get_config_for_method(cls, method: DeconvolutionMethod) -> 'DeconvolutionConfig':
        """Factory method to get configuration for a specific method."""
        configs = {
            DeconvolutionMethod.XCELL: cls._get_xcell_config,
            DeconvolutionMethod.CIBERSORT: cls._get_cibersort_config,
            DeconvolutionMethod.BISQUE: cls._get_bisque_config,
            DeconvolutionMethod.UNKNOWN: cls._get_unknown_config,
        }
        return configs.get(method, cls._get_unknown_config)()
    
    @classmethod
    def _get_xcell_config(cls) -> 'DeconvolutionConfig':
        """Configuration for xCell deconvolution."""
        return cls(
            method=DeconvolutionMethod.XCELL,
            display_name="xCell",
            score_column_name="Enrichment Score",
            score_description="xCell enrichment scores represent relative cell type abundance based on gene expression signatures",
            score_unit="enrichment score",
            level_thresholds={
                'high': 0.1,      # Above 0.1 = High enrichment
                'moderate': 0.01,  # 0.01-0.1 = Moderate
                'low': 0.0        # Below 0.01 = Low
            },
            presence_thresholds={
                'consistent': 0.75,  # Present in >75% samples
                'variable': 0.5,     # Present in 50-75%
                'rare': 0.0          # Present in <50%
            },
            interpretation_templates={
                'high': "{cell_type} shows high enrichment, indicating substantial presence in the microenvironment",
                'moderate': "{cell_type} shows moderate enrichment, suggesting detectable presence",
                'low': "{cell_type} shows low enrichment, indicating minimal presence",
                'not_detected': "{cell_type} not detected in microenvironment analysis"
            },
            reference="Aran D, et al. xCell: digitally portraying the tissue cellular heterogeneity landscape. Genome Biol. 2017"
        )
    
    @classmethod
    def _get_cibersort_config(cls) -> 'DeconvolutionConfig':
        """Configuration for CIBERSORT deconvolution."""
        return cls(
            method=DeconvolutionMethod.CIBERSORT,
            display_name="CIBERSORT",
            score_column_name="Cell Fraction",
            score_description="CIBERSORT cell fractions represent estimated proportions of immune cell types (sum to 1 per sample)",
            score_unit="fraction",
            level_thresholds={
                'high': 0.10,     # Above 10% = High
                'moderate': 0.02, # 2-10% = Moderate
                'low': 0.0        # Below 2% = Low
            },
            presence_thresholds={
                'consistent': 0.75,
                'variable': 0.5,
                'rare': 0.0
            },
            interpretation_templates={
                'high': "{cell_type} represents a major component of the immune infiltrate",
                'moderate': "{cell_type} is present at moderate levels in the immune compartment",
                'low': "{cell_type} is present at low levels",
                'not_detected': "{cell_type} not detected or below detection threshold"
            },
            reference="Newman AM, et al. Robust enumeration of cell subsets from tissue expression profiles. Nat Methods. 2015"
        )
    
    @classmethod
    def _get_bisque_config(cls) -> 'DeconvolutionConfig':
        """Configuration for BisqueRNA deconvolution."""
        return cls(
            method=DeconvolutionMethod.BISQUE,
            display_name="BisqueRNA",
            score_column_name="Cell Proportion",
            score_description="BisqueRNA proportions represent estimated cell type abundances using reference-based decomposition",
            score_unit="proportion",
            level_thresholds={
                'high': 0.15,     # Above 15% = High
                'moderate': 0.05, # 5-15% = Moderate
                'low': 0.0        # Below 5% = Low
            },
            presence_thresholds={
                'consistent': 0.75,
                'variable': 0.5,
                'rare': 0.0
            },
            interpretation_templates={
                'high': "{cell_type} shows high abundance ({score:.1%}) based on reference decomposition",
                'moderate': "{cell_type} shows moderate abundance ({score:.1%})",
                'low': "{cell_type} shows low abundance ({score:.1%})",
                'not_detected': "{cell_type} not detected in decomposition analysis"
            },
            reference="Jew B, et al. Accurate estimation of cell composition in bulk expression through robust integration of single-cell information. Nat Commun. 2020"
        )
    
    @classmethod
    def _get_unknown_config(cls) -> 'DeconvolutionConfig':
        """Configuration for unknown deconvolution methods."""
        return cls(
            method=DeconvolutionMethod.UNKNOWN,
            display_name="Deconvolution",
            score_column_name="Score",
            score_description="Cell type deconvolution scores representing relative cell type abundance",
            score_unit="score",
            level_thresholds={
                'high': 0.1,
                'moderate': 0.01,
                'low': 0.0
            },
            presence_thresholds={
                'consistent': 0.75,
                'variable': 0.5,
                'rare': 0.0
            },
            interpretation_templates={
                'high': "{cell_type} shows high levels",
                'moderate': "{cell_type} shows moderate levels",
                'low': "{cell_type} shows low levels",
                'not_detected': "{cell_type} not detected"
            },
            reference="Cell type deconvolution analysis"
        )


@dataclass
class DeconvolutionResult:
    """
    Standardized deconvolution result that preserves method-specific information.
    
    This is the common interface used throughout the pipeline, but it preserves
    the original method's terminology and scoring conventions.
    """
    cell_type: str
    score: float  # The primary score (enrichment for xCell, fraction for CIBERSORT, proportion for Bisque)
    level: str  # "High", "Moderate", "Low", "Not detected"
    presence_fraction: float  # Fraction of samples with this cell type
    presence_class: str  # "Consistent", "Variable", "Rare"
    interpretation: str  # Method-specific interpretation
    
    # Method-specific metadata
    method: DeconvolutionMethod = field(default=DeconvolutionMethod.UNKNOWN)
    method_display_name: str = ""
    score_column_name: str = "Score"  # Method-specific column name
    
    # Additional method-specific fields
    raw_values: List[float] = field(default_factory=list)  # Raw values across samples
    p_value: Optional[float] = None  # CIBERSORT p-value if available
    correlation: Optional[float] = None  # CIBERSORT correlation if available
    rmse: Optional[float] = None  # CIBERSORT RMSE if available
    
    # Linked data (populated during analysis)
    marker_genes_found: List[str] = field(default_factory=list)
    supporting_pathways: List[str] = field(default_factory=list)
    concordance_status: str = ""


@dataclass
class ClinicalContradiction:
    """Represents a detected clinical contradiction in the analysis."""
    contradiction_id: str
    severity: ContradictionSeverity
    category: str              # e.g., "drug_recommendation", "signature_discordance"
    title: str                 # Short description
    description: str           # Detailed explanation
    affected_elements: List[str]  # Genes, drugs, pathways involved
    resolution: str            # How we resolved/handled it
    clinical_note: str         # What this means clinically


@dataclass
class DiseaseContext:
    """Container for disease information from LLM, Neo4j, or other sources."""
    disease_name: str
    standard_name: str = ""
    description: str = ""
    key_pathways: List[str] = field(default_factory=list)
    pathogenic_mechanisms: List[str] = field(default_factory=list)
    key_genes: Dict[str, Dict] = field(default_factory=dict)
    fda_drugs: List[Dict] = field(default_factory=list)
    biomarkers: List[Dict] = field(default_factory=list)
    cell_types: List[str] = field(default_factory=list)
    icd10_codes: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    # Disease-agnostic fields for enhanced reporting
    known_disease_genes: List[str] = field(default_factory=list)  # Genes known to be associated with this disease
    disease_subtypes: List[Dict] = field(default_factory=list)  # Known subtypes with classification criteria
    biomarker_therapy_mappings: Dict[str, Dict] = field(default_factory=dict)  # Biomarker -> therapy associations


@dataclass
class GeneMapping:
    """
    Result of mapping a patient gene to a disease gene.
    
    Core Fields:
    - category: Classifies gene into one of three buckets
    - evidence_source: Tracks where the gene-disease association came from (CSV, literature, etc.)
    - is_significant: Whether gene passes adj.p < 0.05 threshold
    - composite_score: Disease relevance score (0-1) for ranking
    """
    gene: str
    log2fc: float
    adj_pvalue: float
    disease_role: str = ""
    expected_direction: str = "variable"
    observed_direction: str = ""
    concordant: bool = True
    therapeutic_target: bool = False
    score: float = 0.0
    main_class: str = ""  # Main_Class from CSV for Appendix categorization
    composite_score: float = 0.0  # Disease relevance score (0-1) combining expression, significance, disease association
    # Classification fields
    category: str = ""  # GeneCategory classification
    evidence_source: str = ""  # "Patient CSV", "Literature", "CGC", "COSMIC", etc.
    is_significant: bool = False  # adj.p < 0.05
    other_diseases: List[str] = field(default_factory=list)  # For Category 3 genes
    # KG / PPI evidence from upstream prioritization pipeline
    gene_score: float = 0.0       # GeneCards disease relevance score from Neo4j
    disorder_score: float = 0.0   # Disorder-specific association score from Neo4j
    ppi_degree: float = 0.0       # PPI interaction count (retained for compatibility, not displayed)
    ppi_score: float = 0.0        # STRING PPI confidence score (primary PPI metric)
    jl_score: float = 0.0         # JL prioritization score from upstream
    cgc_flag: bool = False         # Cancer Gene Census membership
    evidence_stratum: str = ""     # Evidence tier: known_driver / ppi_connected / expression_significant / novel_candidate
    trend_consensus: str = ""      # Patient vs cohort trend consensus
    cohort_lfc: Optional[float] = None  # Cohort-level log2FC if available
    causal_tier: str = ""          # Optional causal linkage tier from upstream causal-inference inputs


@dataclass 
class PathwayMapping:
    """Result of mapping a patient pathway to disease relevance."""
    pathway_name: str
    disease_relevance: str = ""
    fdr: float = 0.0
    gene_count: int = 0
    regulation: str = ""
    clinical_relevance: str = ""
    functional_relevance: str = ""
    genes: List[str] = field(default_factory=list)  # Associated genes from CSV
    main_class: str = ""  # Main_Class from CSV for categorization


@dataclass
class XCellEnrichment:
    """
    Cell type enrichment result for deconvolution analysis.
    
    Stores deconvolution results for a single cell type including
    enrichment level, presence, and interpretation.
    
    Multi-method support:
    Supports xCell, CIBERSORT, and BisqueRNA with method-specific metadata.
    """
    cell_type: str
    median_enrichment: float
    enrichment_level: str  # "High", "Moderate", "Low"
    presence_fraction: float
    presence_class: str  # "Frequent", "Occasional/rare"
    interpretation: str
    # Linked data (populated during analysis)
    marker_genes_found: List[str] = field(default_factory=list)
    supporting_pathways: List[str] = field(default_factory=list)
    concordance_status: str = ""  # "Concordant", "Discordant", "Uncertain"
    # Method-specific metadata
    method: Optional[Any] = None  # DeconvolutionMethod enum
    method_display_name: str = ""  # "xCell", "CIBERSORT", "BisqueRNA"
    score_column_name: str = "Enrichment Score"  # Method-specific column name


@dataclass
class DrugRecommendation:
    """
    Enhanced drug recommendation with mechanistic reasoning.
    
    Features:
    - Mechanistic reasoning for each drug recommendation
    - Biomarker concordance mapping
    - Expression signature support
    - Evidence level classification
    
    CRITICAL Requirements:
    - RNA expression CANNOT determine drug eligibility
    - Use hedged language: "suggestive," "supports hypothesis"
    - NEVER state "eligible" based on RNA alone
    - Always include confirmation requirements
    """
    drug_name: str
    target_gene: str
    priority: str  # "High", "Moderate", "Low" - based on EXPRESSION EVIDENCE STRENGTH
    priority_score: float  # 0-100 weighted score
    mechanistic_reasoning: str
    biomarker_concordance: str  # Now uses hedged language
    expression_support: str  # "Upregulated", "Downregulated", "Normal"
    signature_support: List[str] = field(default_factory=list)  # Supporting signatures
    contraindication_flags: List[str] = field(default_factory=list)
    contraindication_entries: List = field(default_factory=list)  # ContraindicationEntry objects with tier
    approval_status: str = ""  # FDA status for the drug itself
    evidence_level: str = ""  # "Level 1A", "Level 2B", etc.
    clinical_context: str = ""  # Disease-specific context
    log2fc: float = 0.0  # Target gene expression
    adj_pvalue: float = 1.0  # Statistical significance
    # Fields for appropriate clinical language
    confirmation_required: str = ""  # What test is needed for eligibility determination
    rna_limitation: str = ""  # Why RNA alone is insufficient for this biomarker
    # Drug-target validation fields
    validation_warning: str = ""  # Warning if drug-target mapping is questionable
    validated_target: str = ""  # Actual validated target if different from claimed
    
    def get_clinical_recommendation(self) -> str:
        """
        Generate appropriately hedged clinical recommendation.
        
        NEVER says "eligible" - uses "suggestive," "supports," etc.
        
        Returns:
            Properly hedged recommendation string with confirmation requirements
        """
        if self.expression_support == "Upregulated":
            expression_lang = f"upregulation of {self.target_gene} (log2FC: {self.log2fc:.2f})"
            support_lang = "is suggestive of potential"
        elif self.expression_support == "Downregulated":
            expression_lang = f"downregulation of {self.target_gene}"
            support_lang = "may indicate reduced"
        else:
            expression_lang = f"expression pattern of {self.target_gene}"
            support_lang = "provides limited evidence for"
        
        # Build recommendation with appropriate hedging
        recommendation = (
            f"Transcriptomic {expression_lang} {support_lang} "
            f"{self.drug_name} candidacy. "
        )
        
        # Add mechanistic context
        if self.mechanistic_reasoning:
            recommendation += f"Mechanism: {self.mechanistic_reasoning[:150]}. "
        
        return recommendation


# =============================================================================
# CELL TYPE NORMALIZER
# =============================================================================
# Normalizes variant cell type names to canonical names for consistent matching

def _build_cell_type_aliases() -> Dict[str, str]:
    """Build cell type aliases programmatically from canonical names and variant patterns."""
    aliases = {}
    # Format: canonical_name -> list of variant patterns (lowercase)
    _mappings = {
        "CD8+ T-cells": ["cd8+ t-cells", "cd8+ t cells", "cd8+_t-cells", "cd8_t_cells", "cd8+ tcells", "cd8 t cells", "cytotoxic t cells", "cytotoxic t-cells"],
        "CD4+ T-cells": ["cd4+ t-cells", "cd4+ t cells", "cd4+_t-cells", "cd4_t_cells", "helper t cells"],
        "Tregs": ["tregs", "regulatory t cells", "regulatory t-cells", "t regulatory cells", "t-regs", "cd4+ tregs"],
        "Macrophages": ["macrophages", "macrophage", "macs"],
        "Macrophages M1": ["m1 macrophages", "m1_macrophages", "macrophages_m1"],
        "Macrophages M2": ["m2 macrophages", "m2_macrophages", "macrophages_m2"],
        "NK cells": ["nk cells", "nk_cells", "natural killer cells", "natural killer"],
        "B-cells": ["b-cells", "b cells", "b_cells", "bcells"],
        "Dendritic cells": ["dendritic cells", "dendritic_cells", "dcs", "dc"],
        "Fibroblasts": ["fibroblasts", "fibroblast", "cafs", "cancer-associated fibroblasts"],
        "Endothelial cells": ["endothelial cells", "endothelial_cells", "ecs", "endothelium"],
        "Epithelial cells": ["epithelial cells", "epithelial_cells", "epithelium"],
        "Monocytes": ["monocytes", "monocyte", "monos"],
        "Neutrophils": ["neutrophils", "neutrophil", "pmns"],
        "Plasma cells": ["plasma cells", "plasma_cells", "plasmacytes"],
        "Mast cells": ["mast cells", "mast_cells", "mastocytes"],
        "Eosinophils": ["eosinophils", "eosinophil"],
        "Basophils": ["basophils", "basophil"],
    }
    
    for canonical, variants in _mappings.items():
        for variant in variants:
            aliases[variant] = canonical
        # Also map canonical to itself (lowercase)
        aliases[canonical.lower()] = canonical
    
    return aliases

# Build alias lookup on module load
CELL_TYPE_ALIASES = _build_cell_type_aliases()


def normalize_celltype_name(cell_type: str) -> str:
    """
    Normalize cell type name to canonical form.
    
    This ensures consistent cell type naming across the pipeline,
    regardless of input format variations.
    
    Args:
        cell_type: Raw cell type name from input data
        
    Returns:
        Normalized canonical cell type name
    """
    if not cell_type:
        return cell_type
    
    # Clean input
    cleaned = cell_type.strip().lower()
    
    # Check alias lookup
    if cleaned in CELL_TYPE_ALIASES:
        return CELL_TYPE_ALIASES[cleaned]
    
    # Return original if no alias found (preserve case from input)
    return cell_type


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_df_column(df, options: List[str]) -> str:
    """
    Find column name from a list of options in a DataFrame.
    
    This utility function is used across multiple classes to handle
    varying column naming conventions in input CSV files.
    
    Args:
        df: pandas DataFrame to search
        options: List of possible column names in order of preference
        
    Returns:
        First matching column name, or first column of df if none match
    """
    for col in options:
        if col in df.columns:
            return col
    return df.columns[0] if len(df.columns) > 0 else ''


def is_artifact_gene(gene_name: str) -> bool:
    """Check if gene is a known/suspected artifact or pseudogene.
    
    These genes (like TRD-GTC2 family) show high differential expression
    but are T-cell receptor pseudogenes - NOT established disease biomarkers.
    They may represent technical artifacts, batch effects, or pseudogene
    expression and should be excluded from clinical recommendations.
    """
    if not gene_name:
        return False
    gene_upper = gene_name.upper()
    
    # Check explicit list first
    if gene_name in KNOWN_ARTIFACT_GENES or gene_upper in KNOWN_ARTIFACT_GENES:
        return True
    
    # Check pattern match for artifact gene families
    for pattern in ARTIFACT_GENE_PATTERNS:
        if gene_upper.startswith(pattern.upper()):
            return True
    
    return False


# Cross-patient confounders: genes that appear in nearly every patient regardless
# of disease, likely due to ubiquitous expression or batch effects.
# They are NOT permanently excluded — only suppressed when lacking disease-specific
# validation (evidence_stratum != known_driver/ppi_connected).
SUSPECTED_CROSS_PATIENT_CONFOUNDERS = {
    "ATF2", "CDK2", "B4GALT3", "TRAPPC10", "RPAIN",
}


def is_suspected_confounder(gene_name: str, evidence_stratum: str = '') -> bool:
    """Return True if gene is a cross-patient confounder WITHOUT disease validation."""
    if not gene_name:
        return False
    if gene_name.upper() not in SUSPECTED_CROSS_PATIENT_CONFOUNDERS:
        return False
    # Disease-validated strata → keep the gene
    return evidence_stratum not in ('known_driver', 'ppi_connected')


def calculate_composite_score(
    log2fc: float, 
    adj_pvalue: float, 
    category: str, 
    is_therapeutic_target: bool
) -> float:
    """
    Calculate composite score for gene ranking.
    
    Components (weights must sum to 1.0):
    - Expression Magnitude (40%): |log2FC| normalized
    - Significance (30%): -log10(adj_pvalue) normalized
    - Disease Association (20%): Category-based
    - Therapeutic Potential (10%): Is druggable target
    
    Returns:
        Score between 0 and 1
    """
    try:
        log2fc = float(log2fc) if log2fc is not None else 0
        adj_pvalue = float(adj_pvalue) if adj_pvalue is not None else 1.0
    except (ValueError, TypeError):
        log2fc = 0
        adj_pvalue = 1.0
    
    # Component 1: Expression Magnitude (40%)
    expression_raw = min(abs(log2fc) / 10, 1.0)
    expression_component = expression_raw * 0.4
    
    # Component 2: Significance (30%)
    if adj_pvalue > 0 and adj_pvalue < 1:
        sig_raw = min(-math.log10(adj_pvalue) / 10, 1.0)
    elif adj_pvalue == 0:
        sig_raw = 1.0
    else:
        sig_raw = 0.0
    significance_component = sig_raw * 0.3
    
    # Component 3: Disease Association (20%)
    if category == GeneCategory.PATIENT_AND_DISEASE_SPECIFIC:
        disease_raw = 1.0
    elif category == GeneCategory.KNOWN_IN_OTHER_CONDITIONS:
        disease_raw = 0.5
    elif category == GeneCategory.PATIENT_SPECIFIC_NOVEL:
        disease_raw = 0.3
    else:
        disease_raw = 0.0
    disease_component = disease_raw * 0.2
    
    # Component 4: Therapeutic Target (10%)
    target_raw = 1.0 if is_therapeutic_target else 0.0
    target_component = target_raw * 0.1
    
    # Total composite score
    total = expression_component + significance_component + disease_component + target_component
    
    return round(total, 3)


def sanitize_clinical_text(text: str) -> str:
    """
    Remove inappropriate eligibility language from generated text.
    
    RNA expression cannot determine drug eligibility.
    This function is a safety net that catches inappropriate language.
    
    Args:
        text: Text that may contain inappropriate clinical language
        
    Returns:
        Sanitized text with appropriate hedging
    """
    if not text:
        return text
    
    # Replacements for forbidden phrases (case-insensitive)
    replacements = [
        # Eligibility language -> hedged language
        (r'\beligible for\b', 'expression pattern supports consideration of'),
        (r'\bnot eligible for\b', 'limited RNA evidence for'),
        (r'\beligibility\b', 'potential candidacy (confirmation required)'),
        (r'\bqualifies for\b', 'expression supports evaluation for'),
        (r'\bdisqualifies from\b', 'expression does not support'),
        (r'\bindicated for\b', 'may warrant evaluation for'),
        (r'\bcontraindicated\b', 'expression pattern does not support'),
        
        # Absolute statements -> hedged statements
        (r'\bwill respond to\b', 'may respond to'),
        (r'\bwill not respond to\b', 'may have reduced response to'),
        (r'\bshould receive\b', 'may be considered for'),
        (r'\bshould not receive\b', 'requires careful evaluation before'),
        
        # Definitive language -> uncertain language
        (r'\bconfirms\s+eligibility\b', 'suggests potential candidacy'),
        (r'\bproves\b', 'supports'),
        (r'\bdefinitely\b', 'likely'),
        (r'\bcertainly\b', 'possibly'),
        
        # Fix "patient is eligible" patterns
        (r'patient is eligible', 'patient may be considered'),
        (r'patient not eligible', 'limited RNA support for patient'),
    ]
    
    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def clean_llm_text(text: str) -> str:
    """
    Remove markdown formatting from LLM-generated text.
    
    Fixes Issue #5: ** marks appearing in report sections.
    """
    if not text:
        return text
    
    # Remove markdown tables (| col | col | format)
    # Match lines that are part of markdown tables
    text = re.sub(r'^\|[^\n]+\|\s*$', '', text, flags=re.MULTILINE)
    # Remove table separator lines (| --- | --- |)
    text = re.sub(r'^\|[\s\-:|]+\|\s*$', '', text, flags=re.MULTILINE)
    
    # Remove the "D. BIOMARKER-THERAPY CONCORDANCE TABLE" heading and any empty content
    # since we render this as a proper table programmatically
    text = re.sub(r'D\.\s*BIOMARKER-THERAPY CONCORDANCE TABLE[^\n]*\n*', '', text, flags=re.IGNORECASE)
    # Also remove any note about it being rendered separately
    text = re.sub(r'\[NOTE:.*BIOMARKER-THERAPY.*\]\n*', '', text, flags=re.IGNORECASE)
    
    # Remove "Associated Pathways:" lines (especially uninformative ones)
    # This removes lines like "Associated Pathways: Not directly identified..." or empty pathway associations
    text = re.sub(r'^\*?\*?Associated Pathways:\*?\*?[^\n]*\n*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove bold markdown (**text**)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Remove italic markdown (*text*)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)
    # Remove headers (### Header)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove bullet markers at start of lines (but keep content)
    text = re.sub(r'^[-*•]\s+', '', text, flags=re.MULTILINE)
    # Remove numbered list prefixes (1. 2. etc)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # Clean up backslashes often added by LLMs
    text = text.replace('\\\\', '')
    text = text.replace('\\', '')
    # Clean up excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Clean up dashes used as separators
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    
    return text.strip()


def smart_truncate(text: str, max_len: int = 80) -> str:
    """
    Truncate text at word boundary for cleaner display.
    
    Fixes Issues #4, #7, #8: Incomplete content in columns.
    """
    if not text:
        return ""
    
    text = str(text).strip()
    
    if len(text) <= max_len:
        return text
    
    # Find last space before max_len
    last_space = text[:max_len].rfind(' ')
    
    # If we found a space at a reasonable position (at least 60% of content)
    if last_space > max_len * 0.6:
        return text[:last_space] + "..."
    
    # Otherwise just truncate at max_len
    return text[:max_len] + "..."

def is_valid_drug_name(drug_name: str) -> bool:
    """
    Validates if a drug name is a valid, actionable therapeutic recommendation.
    
    Returns False for placeholder values that indicate no real drug was found.
    This ensures only clinically meaningful recommendations appear in reports.
    
    Uses dynamic pattern matching - no hardcoded drug lists.
    
    Args:
        drug_name: The drug name to validate
        
    Returns:
        bool: True if the drug name represents a valid recommendation
    """
    if not drug_name:
        return False
    
    # Normalize for comparison
    normalized = str(drug_name).strip().lower()
    
    # Empty or whitespace-only
    if not normalized:
        return False
    
    # Check if it contains only non-alphanumeric characters
    if not any(c.isalnum() for c in normalized):
        return False
    
    # List of invalid placeholder patterns (dynamically catches variations)
    # These are common LLM responses when no valid drug exists for a target
    invalid_patterns = [
        'n/a',
        'na',
        'none',
        'unknown',
        'not available',
        'no drug',
        'no known drug',
        'not applicable',
        'unavailable',
        'no therapeutic',
        'no approved',
        'no fda',
        'not found',
        'undefined',
        'null',
        'tbd',
        'to be determined',
        'pending',
        'no specific',
        'no identified',
        'no recognized',
        '-',
        '--',
        '---',
    ]
    
    # Check exact matches
    if normalized in invalid_patterns:
        return False
    
    # Check if the name starts with common invalid prefixes
    invalid_prefixes = (
        'no ',
        'none ',
        'n/a ',
        'unknown ',
        'not ',
        'no specific ',
        'no known ',
        'no identified ',
    )
    if normalized.startswith(invalid_prefixes):
        return False
    
    # Check if it ends with patterns indicating uncertainty
    invalid_suffixes = (
        ' unknown',
        ' n/a',
        ' not available',
        ' not found',
        ' pending',
    )
    if normalized.endswith(invalid_suffixes):
        return False
    
    # Reject placebo entries from clinical trial data
    if 'placebo' in normalized:
        return False
    
    # Check if the name is too short (likely placeholder)
    # Valid drug names are typically at least 3 characters
    if len(normalized) < 3:
        return False
    
    return True


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'LLM_TEMPERATURE',
    'DEG_ADJ_PVALUE_THRESHOLD',
    'DEG_LOG2FC_THRESHOLD',
    'DEG_HIGH_CONFIDENCE_LOG2FC',
    'DRUG_HIGH_PRIORITY_THRESHOLD',
    'DRUG_MODERATE_PRIORITY_THRESHOLD',
    'XCELL_LOW_SIGNAL_THRESHOLD',
    'CIBERSORT_LOW_SIGNAL_THRESHOLD',
    'BISQUE_LOW_SIGNAL_THRESHOLD',
    'XCELL_MIN_HIGH_ENRICHED',
    'CLINICAL_LANGUAGE_TEMPLATES',
    'BIOMARKER_CONFIRMATION_REQUIREMENTS',
    'ARTIFACT_GENE_PATTERNS',
    'KNOWN_ARTIFACT_GENES',
    'ARTIFACT_GENES',
    'BASE_SIGNATURE_GENES',
    'DRUGGABLE_GENES',
    'DISEASE_BIOMARKERS',
    'CELL_TYPE_MARKERS',
    'CELL_TYPE_PATHWAYS',
    'DRUG_MECHANISM_INFO',
    'CELL_TYPE_ALIASES',
    # Enums
    'DeconvolutionMethod',
    'ContradictionSeverity',
    # Status Classes
    'BiomarkerStatus',
    'GeneCategory',
    # Dataclasses
    'DeconvolutionConfig',
    'DeconvolutionResult',
    'ClinicalContradiction',
    'DiseaseContext',
    'GeneMapping',
    'PathwayMapping',
    'XCellEnrichment',
    'DrugRecommendation',
    # Functions
    'find_df_column',
    'is_artifact_gene',
    'normalize_celltype_name',
    'calculate_composite_score',
    'sanitize_clinical_text',
    'clean_llm_text',
    'smart_truncate',
    'is_valid_drug_name',
]
