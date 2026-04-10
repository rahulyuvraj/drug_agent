"""
Drug Ranker Module (Dynamic)
============================

Ranks drug recommendations using multi-factor scoring.
Fully dynamic - no hardcoded gene, drug, or disease information.
"""

import re
import logging
from typing import List, Dict, Set, Optional
from dataclasses import dataclass

from ..models.data_models import DrugRecommendation, DrugAgentInput
from ..retrieval.hybrid_search import FusedResult
from ..config.settings import RankingConfig, RankingWeights

logger = logging.getLogger(__name__)


# Common biochemical compounds, metabolites, amino acids, nucleotides,
# lab reagents, and recombinant proteins that are not therapeutic drugs.
# Names are lowercased for matching.
NON_DRUG_PATTERNS = frozenset({
    # Amino acids
    "alanine", "arginine", "asparagine", "aspartate", "aspartic acid",
    "cysteine", "glutamate", "glutamic acid", "glutamine", "glycine",
    "histidine", "isoleucine", "leucine", "lysine", "methionine",
    "phenylalanine", "proline", "serine", "threonine", "tryptophan",
    "tyrosine", "valine",
    # Nucleotides and energy molecules
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "utp", "ctp",
    "adenosine triphosphate", "adenosine disphosphate",
    "adenosine diphosphate", "adenosine monophosphate",
    # Sugars and basic metabolites
    "glucose", "fructose", "galactose", "sucrose", "lactose",
    "pyruvate", "lactate", "citrate", "acetate", "butyrate",
    # Lipids and basic biochemicals
    "phosphatidylinositol", "phosphatidylcholine",
    "phosphatidylserine", "phosphatidylethanolamine",
    "cholesterol", "sphingomyelin",
    # Lab reagents and stimulants
    "endotoxin", "lipopolysaccharide", "lps",
    "12-o-tetradecanoylphorbol 13-acetate", "tpa", "pma",
    "phorbol myristate acetate",
    "cuprophan", "fmlp", "concanavalin",
    "actinomycin", "aebsf", "c2ceramide", "c2-ceramide",
    # Ions and minerals
    "calcium", "zinc", "iron", "magnesium", "sodium", "potassium",
    # Recombinant proteins (not drugs)
    "recombinant human", "recombinant rat", "recombinant murine",
    "recombinant mouse",
    # Basic vitamins / cofactors (context-dependent, often not drug recs)
    "adpribose",
})

# Prefixes that indicate a non-drug entry
NON_DRUG_PREFIXES = (
    "recombinant human", "recombinant rat", "recombinant murine",
    "recombinant mouse",
)


@dataclass
class RankingContext:
    """Context for ranking drugs."""
    disease_name: str
    patient_genes: Set[str]
    patient_genes_lower: Set[str]
    patient_pathways: Set[str]
    gene_directions: Dict[str, str]
    upregulated_genes: Set[str]
    downregulated_genes: Set[str]


class DrugRanker:
    """
    Ranks drug candidates using multi-factor scoring.

    Scoring factors:
    1. Relevance score (from vector similarity)
    2. Gene match score (how many patient genes are targeted)
    3. Evidence level score (derived from approval status text)
    4. Approval status score (derived from status text patterns)
    5. Indication match bonus (disease relevance from indications)
    """

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.weights = self.config.weights
    
    def rank_results(
        self,
        results: List[FusedResult],
        input_data: DrugAgentInput,
        max_results: int = 15,
    ) -> List[DrugRecommendation]:
        """
        Rank fused search results and create drug recommendations.
        
        Args:
            results: List of FusedResult from hybrid search.
            input_data: Original input data with patient context.
            max_results: Maximum recommendations to return.
            
        Returns:
            Ranked list of DrugRecommendation objects.
        """
        # Build ranking context from input
        patient_genes = set(g.gene for g in input_data.gene_mappings)
        
        context = RankingContext(
            disease_name=input_data.disease_name,
            patient_genes=patient_genes,
            patient_genes_lower={g.lower() for g in patient_genes},
            patient_pathways=set(p.pathway_name for p in input_data.pathway_mappings),
            gene_directions=input_data.get_gene_directions(),
            upregulated_genes=set(input_data.get_upregulated_genes()),
            downregulated_genes=set(input_data.get_downregulated_genes()),
        )
        
        # Score each result
        scored_recommendations = []
        for result in results:
            recommendation = self._score_and_create_recommendation(result, context)
            if recommendation:
                scored_recommendations.append(recommendation)
        
        # Sort by composite score
        scored_recommendations.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Deduplicate by drug name
        seen_drugs: Set[str] = set()
        deduplicated = []
        for rec in scored_recommendations:
            drug_key = rec.drug_name.lower()
            if drug_key not in seen_drugs:
                seen_drugs.add(drug_key)
                deduplicated.append(rec)
        
        logger.info(f"Ranked {len(results)} results -> {len(deduplicated)} unique recommendations")
        
        return deduplicated[:max_results]
    
    @staticmethod
    def _is_non_drug(name: str) -> bool:
        """Check if a name is a common metabolite, reagent, or non-drug compound."""
        name_lower = name.lower().strip()

        # Exact match
        if name_lower in NON_DRUG_PATTERNS:
            return True

        # Prefix match (recombinant proteins etc.)
        for prefix in NON_DRUG_PREFIXES:
            if name_lower.startswith(prefix):
                return True

        return False

    def _score_and_create_recommendation(
        self,
        result: FusedResult,
        context: RankingContext,
    ) -> Optional[DrugRecommendation]:
        """Score a single result and create a recommendation."""
        drug_name = result.drug_name
        if not drug_name:
            return None

        # Filter out non-drug compounds
        if self._is_non_drug(drug_name):
            return None

        # Calculate individual scores
        relevance_score = min(result.score * 2, 1.0)  # Normalize RRF score
        gene_match_score = self._calculate_gene_match_score(result, context)
        evidence_score = self._parse_evidence_score(result.evidence_level)
        approval_score = self._parse_approval_score(result.approval_status, context.disease_name)
        
        # Calculate indication match score
        indication_match = self._get_indication_match(result, context.disease_name)
        indication_bonus = self._calculate_indication_bonus(indication_match)

        # Calculate composite score using weights + indication bonus
        composite_score = (
            self.weights.relevance * relevance_score +
            self.weights.gene_match * gene_match_score +
            self.weights.evidence * evidence_score +
            self.weights.approval_status * approval_score +
            indication_bonus
        )

        # Find patient gene matches
        target_gene = result.gene_symbol
        patient_gene_match = self._find_gene_matches(result, context)
        
        # Determine expression concordance
        expression_concordance = self._get_expression_concordance(target_gene, context)
        
        # Collect all target genes for this drug
        all_targets = list(set(
            ([target_gene] if target_gene else []) +
            result.payload.get("all_target_genes", [])
        ))

        # Create recommendation
        recommendation = DrugRecommendation(
            drug_name=drug_name,
            drug_aliases=result.payload.get("drug_aliases", []),
            drug_type=result.payload.get("drug_type", ""),
            target_genes=all_targets,
            target_pathways=[result.pathway_name] if result.pathway_name else [],
            mechanism_of_action=result.mechanism_of_action,
            approval_status=result.approval_status,
            indication_match=indication_match,
            evidence_level=result.evidence_level,
            patient_gene_match=patient_gene_match,
            patient_pathway_match=[],
            expression_concordance=expression_concordance,
            relevance_score=relevance_score,
            gene_match_score=gene_match_score,
            evidence_score=evidence_score,
            approval_score=approval_score,
            composite_score=composite_score,
            evidence_summary=self._generate_evidence_summary(result, context),
            evidence_sources=[result.payload.get("source", "")],
            confirmation_tests=self._infer_confirmation_tests(result, context),
        )
        
        return recommendation
    
    def _calculate_gene_match_score(
        self,
        result: FusedResult,
        context: RankingContext,
    ) -> float:
        """Calculate gene match score based on patient genes."""
        # Check all target genes (merged from multiple gene docs)
        all_targets = result.payload.get("all_target_genes", [])
        target_gene = result.gene_symbol

        # Combine primary gene_symbol with all_target_genes
        genes_to_check = set()
        if target_gene:
            genes_to_check.add(target_gene)
        genes_to_check.update(all_targets)

        if not genes_to_check:
            return 0.0

        # Find best match across all target genes
        best_score = 0.0
        for gene in genes_to_check:
            gene_lower = gene.lower()
            if gene_lower in context.patient_genes_lower:
                if gene in context.upregulated_genes:
                    best_score = max(best_score, 1.0)
                elif gene in context.downregulated_genes:
                    best_score = max(best_score, 0.85)
                else:
                    best_score = max(best_score, 0.9)

        if best_score > 0:
            return best_score

        # Check aliases from payload
        gene_aliases = result.payload.get("gene_aliases", [])
        for alias in gene_aliases:
            if alias.lower() in context.patient_genes_lower:
                return 0.8

        # Check pathway genes
        pathway_genes = result.payload.get("pathway_genes", [])
        if pathway_genes:
            matching = set(g.lower() for g in pathway_genes) & context.patient_genes_lower
            if matching:
                return 0.5 * min(len(matching) / 3, 1.0)

        return 0.0
    
    def _parse_evidence_score(self, evidence_level: str) -> float:
        """
        Parse evidence score from evidence level text.
        Dynamically interprets the text without hardcoded mappings.
        """
        if not evidence_level:
            return 0.1
        
        level_lower = evidence_level.lower()
        
        # Pattern matching for evidence levels
        if any(x in level_lower for x in ["1a", "level 1a", "highest", "strong"]):
            return 1.0
        elif any(x in level_lower for x in ["1b", "level 1b"]):
            return 0.85
        elif any(x in level_lower for x in ["2a", "level 2a", "moderate"]):
            return 0.7
        elif any(x in level_lower for x in ["2b", "level 2b"]):
            return 0.55
        elif any(x in level_lower for x in ["3", "level 3", "weak"]):
            return 0.3
        elif any(x in level_lower for x in ["4", "level 4", "limited", "preclinical"]):
            return 0.15
        elif "fda" in level_lower or "approved" in level_lower:
            return 1.0
        elif "phase" in level_lower:
            # Extract phase number
            phase_match = re.search(r'phase\s*([iI1-4]+)', level_lower)
            if phase_match:
                phase = phase_match.group(1).lower()
                if phase in ["3", "iii"]:
                    return 0.7
                elif phase in ["2", "ii"]:
                    return 0.5
                elif phase in ["1", "i"]:
                    return 0.3
        
        return 0.2  # Default for unknown
    
    def _parse_approval_score(self, approval_status: str, disease: str) -> float:
        """
        Parse approval score from status text.
        Dynamically interprets the text without hardcoded mappings.
        """
        if not approval_status:
            return 0.1
        
        status_lower = approval_status.lower()
        disease_lower = disease.lower()
        
        # Check for approval keywords
        if "approved" in status_lower or "fda" in status_lower:
            # Higher score if disease is mentioned
            if disease_lower in status_lower:
                return 1.0
            return 0.75
        elif "ema" in status_lower:
            return 0.7
        elif "phase" in status_lower:
            # Extract phase
            phase_match = re.search(r'phase\s*([iI1-4]+)', status_lower)
            if phase_match:
                phase = phase_match.group(1).lower()
                if phase in ["3", "iii"]:
                    return 0.6
                elif phase in ["2", "ii"]:
                    return 0.4
                elif phase in ["1", "i"]:
                    return 0.25
        elif "preclinical" in status_lower:
            return 0.15
        elif "experimental" in status_lower or "investigational" in status_lower:
            return 0.2
        
        return 0.1

    @staticmethod
    def _calculate_indication_bonus(indication_match: str) -> float:
        """
        Calculate a bonus score based on how well the drug's known
        indications match the patient's disease.

        This additive bonus rewards drugs with established disease relevance
        so they rank above compounds with no disease association.
        """
        if indication_match == "Approved for indication":
            return 0.20
        elif indication_match == "Associated with indication":
            return 0.15
        elif indication_match == "Related indication":
            return 0.10
        elif indication_match == "Approved for different indication":
            return 0.05
        elif indication_match == "Under clinical investigation":
            return 0.03
        return 0.0

    def _find_gene_matches(
        self,
        result: FusedResult,
        context: RankingContext,
    ) -> List[str]:
        """Find all patient genes that match this drug's targets."""
        matches = []
        patient_upper = {g.upper() for g in context.patient_genes}

        # Check primary gene_symbol and all merged target genes
        all_targets = set()
        if result.gene_symbol:
            all_targets.add(result.gene_symbol)
        all_targets.update(result.payload.get("all_target_genes", []))

        for gene in all_targets:
            if gene.upper() in patient_upper and gene not in matches:
                matches.append(gene)

        gene_aliases = result.payload.get("gene_aliases", [])
        for alias in gene_aliases:
            if alias.upper() in patient_upper and alias not in matches:
                matches.append(alias)

        return matches
    
    def _get_indication_match(self, result: FusedResult, disease: str) -> str:
        """Determine how well the drug indication matches the disease."""
        indications = result.payload.get("indications", [])
        approval_status = result.approval_status.lower() if result.approval_status else ""
        disease_lower = disease.lower()
        # Also check disease keywords (e.g., "rheumatoid" matches "rheumatoid arthritis")
        disease_keywords = [w.lower() for w in disease.split() if len(w) > 3]

        # Check indications list (from gene doc diseases array)
        for indication in indications:
            if not indication:
                continue
            ind_lower = indication.lower()
            if disease_lower in ind_lower:
                return "Associated with indication"
            if any(kw in ind_lower for kw in disease_keywords):
                return "Related indication"

        # Check approval status text
        if disease_lower in approval_status:
            return "Approved for indication"
        elif "approved" in approval_status:
            return "Approved for different indication"
        elif "phase" in approval_status:
            return "Under clinical investigation"

        return "Preclinical or experimental"
    
    def _get_expression_concordance(
        self,
        target_gene: Optional[str],
        context: RankingContext,
    ) -> str:
        """Determine if drug target matches expression pattern."""
        if not target_gene:
            return "Target unknown"
        
        if target_gene in context.upregulated_genes:
            return f"Target {target_gene} is upregulated - inhibition may be beneficial"
        elif target_gene in context.downregulated_genes:
            return f"Target {target_gene} is downregulated - activation may be beneficial"
        elif target_gene in context.patient_genes:
            direction = context.gene_directions.get(target_gene, "differentially")
            return f"Target {target_gene} is {direction} expressed"
        
        return "Target not in patient DEG profile"
    
    def _generate_evidence_summary(
        self,
        result: FusedResult,
        context: RankingContext,
    ) -> str:
        """Generate a brief evidence summary."""
        parts = []
        
        drug_name = result.drug_name
        target_gene = result.gene_symbol
        
        if target_gene:
            if target_gene in context.patient_genes:
                direction = context.gene_directions.get(target_gene, "differentially expressed")
                parts.append(f"{drug_name} targets {target_gene} which is {direction} in this patient.")
            else:
                parts.append(f"{drug_name} targets {target_gene}.")
        
        if result.mechanism_of_action:
            parts.append(f"Mechanism: {result.mechanism_of_action}.")
        
        if result.approval_status:
            parts.append(f"Status: {result.approval_status}.")
        
        return " ".join(parts) if parts else "Evidence details from knowledge base."
    
    def _infer_confirmation_tests(
        self,
        result: FusedResult,
        context: RankingContext,
    ) -> List[str]:
        """
        Infer recommended confirmation tests from the data.
        Dynamically generated based on drug/gene information.
        """
        tests = []
        
        target_gene = result.gene_symbol
        mechanism = result.mechanism_of_action.lower() if result.mechanism_of_action else ""
        drug_type_raw = result.payload.get("drug_type", "")
        drug_type = drug_type_raw.lower() if isinstance(drug_type_raw, str) else ""
        
        if target_gene:
            # Generic recommendation based on target
            tests.append(f"Confirm {target_gene} expression/status")
            
            # Add specific tests based on mechanism keywords
            if any(x in mechanism for x in ["kinase", "receptor tyrosine"]):
                tests.append(f"{target_gene} mutation/amplification testing")
            elif any(x in mechanism for x in ["hormone", "estrogen", "androgen"]):
                tests.append(f"{target_gene} receptor status (IHC)")
            elif any(x in mechanism for x in ["antibody", "monoclonal"]):
                tests.append(f"{target_gene} expression level (IHC/FISH)")
            elif any(x in mechanism for x in ["parp", "dna repair"]):
                tests.append("Homologous recombination deficiency (HRD) testing")
            elif any(x in mechanism for x in ["checkpoint", "pd-1", "pd-l1", "ctla"]):
                tests.append("PD-L1 expression and/or MSI status")
        
        # Add gene-specific test if in patient profile
        if target_gene in context.patient_genes:
            tests.append(f"Validate {target_gene} differential expression")
        
        if not tests:
            tests.append("Confirmatory molecular testing as indicated")
        
        return tests[:3]  # Limit to 3 tests
