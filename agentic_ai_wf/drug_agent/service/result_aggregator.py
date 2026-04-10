"""Cross-collection drug entity resolution and candidate merging."""

import re
import logging
from typing import Dict, List, Optional, Tuple

from .schemas import (
    DrugCandidate, DrugIdentity, TargetEvidence, TrialEvidence,
    SafetyProfile, ScoreBreakdown,
)

logger = logging.getLogger(__name__)

_PLACEBO_RE = re.compile(r'\bplacebo\b', re.IGNORECASE)

# Regex for stripping parentheticals and bracket suffixes
_PAREN_RE = re.compile(r"\s*\(.*?\)")
_BRACKET_RE = re.compile(r"\s*\[.*?\]")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9\s\-]")
_SALT_RE = re.compile(
    r"\s+(?:HYDROCHLORIDE|MONOHYDRATE|DIHYDRATE|TRIHYDRATE|MESYLATE|MESILATE|"
    r"TOSYLATE|DITOSYLATE|MALEATE|DIMALEATE|FUMARATE|CITRATE|ACETATE|"
    r"SODIUM|POTASSIUM|CALCIUM|SUCCINATE|TARTRATE|BESYLATE|BESILATE|"
    r"PHOSPHATE|SULFATE|NITRATE|BROMIDE|CHLORIDE|"
    r"MOFETIL|AXETIL|PIVOXIL|MARBOXIL|MEDOXOMIL|CILEXETIL|PROXETIL|ALAFENAMIDE)\b"
)

_ADC_PREFIXES = re.compile(r"^(?:FAM|ADO|BEL|MOX|POL|SAC|LON|MIR|TIS)-?", re.IGNORECASE)
_ADC_CONJUGATE_SUFFIXES = re.compile(
    r"[\s\-]*(?:EMTANSINE|DERUXTECAN|VEDOTIN|GOVITECAN|OZOGAMICIN|MERTANSINE|"
    r"RAVTANSINE|MAFODOTIN|TESIRINE|DUOCARMAZINE|CALICHEAMICIN)\b",
    re.IGNORECASE,
)
_BIOSIMILAR_SUFFIXES = re.compile(
    r"[\s\-]+(?:DKST|ANNS|BVZR|ARRX|ADAZ|AXXQ|BKOS|CBIV|DYYB|FKJW|"
    r"GXNZ|HYTO|JMCP|KXWC|LBHQ|MALY|NCRS|PHVC|QDWK|RMBS|SELI|TNJY|"
    r"TRBW|THHN|RWPE|EPBZ|HHDQ|NLBA|XXHL|BMEZ|JUJN|KFCD|FHLL|BTML)\b",
    re.IGNORECASE,
)

_DIAGNOSTIC_RE = re.compile(
    r"(?:DIAGNOSTIC|IMAGING|RADIOACTIVE|CONTRAST|TRACER|RADIOPHARMACEUTICAL|"
    r"TECHNETIUM|GADOLINIUM|INDIUM|GALLIUM|IODINE-\d+|BARIUM|FLORBETAPIR|"
    r"FLUDEOXYGLUCOSE|FDG)",
    re.IGNORECASE,
)


class ResultAggregator:

    def normalize_drug_name(self, name: str) -> str:
        if not name:
            return ""
        n = name.upper().strip()
        n = _PAREN_RE.sub("", n)
        n = _BRACKET_RE.sub("", n)
        n = _NON_ALNUM_RE.sub("", n)
        # Iteratively strip salt form suffixes
        prev = None
        while prev != n:
            prev = n
            n = _SALT_RE.sub("", n).strip()
        return n.strip()

    def extract_inn(self, name: str) -> Tuple[str, bool]:
        """Return (INN, is_adc). Strips ADC prefixes/conjugates and biosimilar suffixes."""
        n = name.upper().strip()
        n = _PAREN_RE.sub("", n)
        n = _BRACKET_RE.sub("", n)
        n = _NON_ALNUM_RE.sub("", n)
        adc = bool(_ADC_PREFIXES.search(n)) or bool(_ADC_CONJUGATE_SUFFIXES.search(n))
        if adc:
            n = _ADC_PREFIXES.sub("", n)
            n = _ADC_CONJUGATE_SUFFIXES.sub("", n)
        n = _BIOSIMILAR_SUFFIXES.sub("", n)
        prev = None
        while prev != n:
            prev = n
            n = _SALT_RE.sub("", n).strip()
        return n.strip(), adc

    @staticmethod
    def is_diagnostic_agent(pharm_class_epc: str = "", product_type: str = "", drug_name: str = "") -> bool:
        text = f"{pharm_class_epc} {product_type} {drug_name}"
        return bool(_DIAGNOSTIC_RE.search(text))

    def merge_candidates(self, discovery_results: Dict[str, List[Dict]],
                         max_candidates: int = 150) -> List[Dict]:
        """Deduplicate discovered drugs by normalized name, merge sources."""
        grouped: Dict[str, Dict] = {}

        for source_key, drugs in discovery_results.items():
            for d in drugs:
                raw_name = d.get("drug_name", "")
                norm = self.normalize_drug_name(raw_name)
                if not norm or _PLACEBO_RE.search(raw_name):
                    continue
                if norm not in grouped:
                    grouped[norm] = {
                        "drug_name": d["drug_name"],
                        "normalized": norm,
                        "sources": set(),
                        "best_score": 0.0,
                        "targets": set(),
                        "mechanisms": set(),
                        "original_names": set(),
                        "discovery_paths": set(),
                    }
                entry = grouped[norm]
                entry["sources"].add(d.get("source", source_key))
                entry["best_score"] = max(entry["best_score"], d.get("score", 0))
                # Classify discovery path from source_key prefix
                if source_key.startswith("target_"):
                    entry["discovery_paths"].add("gene")
                elif source_key == "disease":
                    entry["discovery_paths"].add("disease")
                elif source_key.startswith("pathway_"):
                    entry["discovery_paths"].add("pathway")
                entry["original_names"].add(d.get("drug_name", ""))
                if d.get("gene_symbol"):
                    entry["targets"].add(d["gene_symbol"])
                if d.get("mechanism"):
                    entry["mechanisms"].add(d["mechanism"][:100])
                if d.get("chembl_id"):
                    entry.setdefault("chembl_ids", set()).add(d["chembl_id"])
                if d.get("pathway_class") and not entry.get("pathway_class"):
                    entry["pathway_class"] = d["pathway_class"]

        # ChEMBL ID deduplication: iteratively merge entries sharing any chembl_id.
        # Iteration handles transitive chains (A↔B via ID1, B↔C via ID2).
        merged_any = True
        while merged_any:
            merged_any = False
            chembl_to_norms: Dict[str, List[str]] = {}
            for norm, entry in grouped.items():
                for cid in entry.get("chembl_ids", set()):
                    chembl_to_norms.setdefault(cid, []).append(norm)
            for cid, norms in chembl_to_norms.items():
                norms = [n for n in norms if n in grouped]
                if len(norms) <= 1:
                    continue
                norms_sorted = sorted(norms, key=lambda n: (
                    -len(grouped[n]["sources"]),
                    -grouped[n]["best_score"],
                    len(grouped[n]["drug_name"]),
                ))
                canonical = norms_sorted[0]
                for dup in norms_sorted[1:]:
                    if dup not in grouped:
                        continue
                    dup_entry = grouped.pop(dup)
                    canonical_entry = grouped[canonical]
                    canonical_entry["sources"].update(dup_entry["sources"])
                    canonical_entry["best_score"] = max(canonical_entry["best_score"], dup_entry["best_score"])
                    canonical_entry["targets"].update(dup_entry["targets"])
                    canonical_entry["mechanisms"].update(dup_entry["mechanisms"])
                    canonical_entry.setdefault("chembl_ids", set()).update(dup_entry.get("chembl_ids", set()))
                    canonical_entry["original_names"].update(dup_entry.get("original_names", set()))
                    canonical_entry.setdefault("discovery_paths", set()).update(dup_entry.get("discovery_paths", set()))
                    logger.info(f"Dedup: merged '{dup_entry['drug_name']}' into '{canonical_entry['drug_name']}' (ChEMBL {cid})")
                    merged_any = True

        # INN-based biosimilar merge: group by INN, keep ADCs separate from parent mAb
        for norm, entry in grouped.items():
            inn, adc = self.extract_inn(entry["drug_name"])
            entry["inn"] = inn
            entry["is_adc"] = adc

        inn_groups: Dict[str, List[str]] = {}
        for norm, entry in grouped.items():
            key = entry["inn"] + ("|ADC" if entry["is_adc"] else "")
            inn_groups.setdefault(key, []).append(norm)

        for inn_key, norms in inn_groups.items():
            norms = [n for n in norms if n in grouped]
            if len(norms) <= 1:
                continue
            norms_sorted = sorted(norms, key=lambda n: (
                -len(grouped[n]["sources"]),
                -grouped[n]["best_score"],
                len(grouped[n]["drug_name"]),
            ))
            canonical = norms_sorted[0]
            for dup in norms_sorted[1:]:
                if dup not in grouped:
                    continue
                dup_entry = grouped.pop(dup)
                canonical_entry = grouped[canonical]
                canonical_entry["sources"].update(dup_entry["sources"])
                canonical_entry["best_score"] = max(canonical_entry["best_score"], dup_entry["best_score"])
                canonical_entry["targets"].update(dup_entry["targets"])
                canonical_entry["mechanisms"].update(dup_entry["mechanisms"])
                canonical_entry.setdefault("chembl_ids", set()).update(dup_entry.get("chembl_ids", set()))
                canonical_entry["original_names"].update(dup_entry.get("original_names", set()))
                canonical_entry.setdefault("discovery_paths", set()).update(dup_entry.get("discovery_paths", set()))
                canonical_entry.setdefault("merged_from", []).append(dup_entry["drug_name"])
                logger.info(f"INN dedup: merged '{dup_entry['drug_name']}' into '{canonical_entry['drug_name']}' (INN {inn_key})")

        ranked = sorted(grouped.values(), key=lambda x: (len(x["sources"]), x["best_score"]), reverse=True)

        # Reserve slots for disease-sourced drugs to prevent gene-hit flooding.
        # Disease-sourced drugs come from the "disease" key and often have
        # only one source, so they rank below gene-discovered multi-source drugs.
        _DISEASE_RESERVE = 20
        disease_keys = {k for k in discovery_results if k == "disease" or k.startswith("pathway_")}
        disease_norm_set: set = set()
        for dk in disease_keys:
            for d in discovery_results.get(dk, []):
                norm = self.normalize_drug_name(d.get("drug_name", ""))
                if norm:
                    disease_norm_set.add(norm)

        # Partition: gene-only vs disease-relevant
        disease_ranked = [r for r in ranked if r["normalized"] in disease_norm_set]
        gene_only_ranked = [r for r in ranked if r["normalized"] not in disease_norm_set]

        # Interleave: take max_candidates with at least _DISEASE_RESERVE disease slots
        result = []
        used = set()
        # First pass: take all drugs that appear in both gene AND disease results
        for r in ranked:
            if r["normalized"] in disease_norm_set and len(r["targets"]) > 0:
                result.append(r)
                used.add(r["normalized"])
        # Second pass: fill disease reserve from remaining disease drugs
        for r in disease_ranked:
            if len(result) >= _DISEASE_RESERVE:
                break
            if r["normalized"] not in used:
                result.append(r)
                used.add(r["normalized"])
        # Third pass: fill remaining with gene-only drugs by rank
        for r in ranked:
            if len(result) >= max_candidates:
                break
            if r["normalized"] not in used:
                result.append(r)
                used.add(r["normalized"])

        for r in result:
            r["sources"] = list(r["sources"])
            r["targets"] = list(r["targets"])
            r["mechanisms"] = list(r["mechanisms"])
            r["original_names"] = list(r.get("original_names", set()))
            r["discovery_paths"] = list(r.get("discovery_paths", set()))
            r["merged_from"] = list(r.get("merged_from", []))
            r.setdefault("inn", "")
            r.setdefault("is_adc", False)
            stripped = self.normalize_drug_name(r["drug_name"]).title()
            if len(stripped) < len(r["drug_name"]):
                r["drug_name"] = stripped

        return result[:max_candidates]

    def build_candidate(self, drug_name: str, identity: Dict, targets: List[Dict],
                        indication: Dict, trials: Dict, safety: Dict,
                        evidence_sources: Optional[List[str]] = None) -> DrugCandidate:
        """Assemble a DrugCandidate from all evidence dicts."""
        did = DrugIdentity(
            drug_name=identity.get("drug_name", drug_name),
            chembl_id=identity.get("chembl_id"),
            drug_type=identity.get("drug_type"),
            max_phase=identity.get("max_phase"),
            first_approval=identity.get("first_approval"),
            is_fda_approved=identity.get("is_fda_approved", False),
            brand_names=identity.get("brand_names", []),
            patent_count=identity.get("patent_count", 0),
            exclusivity_count=identity.get("exclusivity_count", 0),
            generics_available=identity.get("generics_available", False),
            pharm_class_moa=identity.get("pharm_class_moa"),
            pharm_class_epc=identity.get("pharm_class_epc"),
            indication_text=identity.get("indication_text"),
            approved_indications=identity.get("approved_indications", []),
            withdrawn=identity.get("withdrawn", False),
            genetic_eligibility_required=identity.get("genetic_eligibility_required", False),
            genetic_eligibility_detail=identity.get("genetic_eligibility_detail", ""),
        )

        target_evidences = []
        for t in targets:
            target_evidences.append(TargetEvidence(
                gene_symbol=t.get("gene_symbol", ""),
                action_type=t.get("action_type", "UNKNOWN"),
                mechanism_of_action=t.get("mechanism"),
                fda_moa_narrative=t.get("fda_narrative"),
                ot_association_score=t.get("ot_association_score"),
                known_effectors=t.get("known_effectors"),
            ))

        trial_ev = None
        if trials and trials.get("total_trials", 0) > 0:
            trial_ev = TrialEvidence(
                total_trials=trials["total_trials"],
                highest_phase=trials.get("highest_phase"),
                completed_trials=trials.get("completed_trials", 0),
                trials_with_results=trials.get("trials_with_results", 0),
                best_p_value=trials.get("best_p_value"),
                total_enrollment=trials.get("total_enrollment", 0),
                top_trials=trials.get("top_trials", []),
                stopped_for_safety=trials.get("stopped_for_safety", False),
            )

        safety_prof = None
        if safety:
            safety_prof = SafetyProfile(
                boxed_warnings=safety.get("boxed_warnings", []),
                top_adverse_events=safety.get("top_adverse_events", []),
                serious_ratio=safety.get("serious_ratio"),
                fatal_ratio=safety.get("fatal_ratio"),
                contraindications=safety.get("contraindications", []),
                pgx_warnings=safety.get("pgx_warnings", []),
                recall_history=safety.get("recall_history", []),
            )

        return DrugCandidate(
            identity=did,
            targets=target_evidences,
            trial_evidence=trial_ev,
            safety=safety_prof,
            evidence_sources=evidence_sources or [],
        )
