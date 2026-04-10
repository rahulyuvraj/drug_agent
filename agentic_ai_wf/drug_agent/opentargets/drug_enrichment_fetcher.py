#!/usr/bin/env python3
"""
Open Targets Drug Enrichment Fetcher
====================================
Fetches detailed drug data including mechanisms, targets, and indications
for drug recommendation use cases.
"""

import logging
import time
import requests
from typing import Dict, List, Generator, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"


@dataclass
class FetchConfig:
    page_size: int = 100
    max_retries: int = 5
    retry_delay: float = 2.0
    request_delay: float = 0.3


class DrugEnrichmentFetcher:
    """Fetches enriched drug data from Open Targets."""
    
    def __init__(self, config: FetchConfig = None):
        self.config = config or FetchConfig()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._seen_ids: Set[str] = set()
    
    def _query(self, query: str, variables: Dict = None) -> Dict:
        """Execute GraphQL query with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(self.config.request_delay)
                response = self.session.post(
                    OPENTARGETS_API,
                    json={"query": query, "variables": variables or {}},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                if "errors" in data:
                    logger.warning(f"GraphQL errors: {data['errors']}")
                return data.get("data", {})
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [429, 502, 503, 504]:
                    delay = self.config.retry_delay * (attempt + 1) * 2
                    logger.warning(f"API error {e.response.status_code}, waiting {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"HTTP error: {e}")
                    break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(self.config.retry_delay * (attempt + 1))
        return {}

    def fetch_drug_ids(self, limit: int = 5000) -> List[str]:
        """Fetch drug ChEMBL IDs using search."""
        logger.info(f"Fetching up to {limit:,} drug IDs...")
        
        search_terms = [
            "inhibitor", "agonist", "antagonist", "blocker", "antibody",
            "cancer", "diabetes", "inflammation", "infection", "cardiovascular",
            "approved", "phase 3", "phase 4", "kinase", "receptor",
            "immunotherapy", "chemotherapy", "biologic", "small molecule",
            "monoclonal", "vaccine", "antiviral", "antibiotic"
        ]
        
        drug_ids = []
        seen = set()
        
        for term in search_terms:
            if len(drug_ids) >= limit:
                break
            
            for page_idx in range(0, 5):
                if len(drug_ids) >= limit:
                    break
                
                query = """
                query SearchDrugs($term: String!, $size: Int!, $index: Int!) {
                    search(queryString: $term, entityNames: ["drug"], page: {size: $size, index: $index}) {
                        hits {
                            id
                            entity
                        }
                    }
                }
                """
                
                try:
                    data = self._query(query, {"term": term, "size": self.config.page_size, "index": page_idx})
                    hits = data.get("search", {}).get("hits", [])
                    if not hits:
                        break
                    
                    for hit in hits:
                        if hit.get("entity") == "drug" and hit.get("id") not in seen:
                            seen.add(hit["id"])
                            drug_ids.append(hit["id"])
                            if len(drug_ids) >= limit:
                                break
                except Exception as e:
                    logger.warning(f"Search for '{term}' failed: {e}")
                    break
        
        logger.info(f"Found {len(drug_ids):,} unique drug IDs")
        return drug_ids

    def fetch_drug_details(self, drug_id: str) -> Dict:
        """Fetch detailed drug information."""
        query = """
        query DrugDetails($chemblId: String!) {
            drug(chemblId: $chemblId) {
                id
                name
                description
                drugType
                maximumClinicalTrialPhase
                hasBeenWithdrawn
                mechanismsOfAction {
                    rows {
                        mechanismOfAction
                        targetName
                        targets {
                            id
                            approvedSymbol
                        }
                    }
                }
                indications {
                    rows {
                        disease {
                            id
                            name
                        }
                        maxPhaseForIndication
                    }
                }
                linkedTargets {
                    rows {
                        id
                        approvedSymbol
                    }
                }
                linkedDiseases {
                    rows {
                        id
                        name
                    }
                }
            }
        }
        """
        
        try:
            data = self._query(query, {"chemblId": drug_id})
            drug = data.get("drug")
            if not drug:
                return None
            
            # Extract mechanisms
            mechanisms = []
            mechanism_targets = []
            for m in drug.get("mechanismsOfAction", {}).get("rows", []):
                mechanisms.append(m.get("mechanismOfAction", ""))
                for t in m.get("targets", []):
                    mechanism_targets.append(t.get("approvedSymbol", ""))
            
            # Extract indications
            indications = []
            for i in drug.get("indications", {}).get("rows", []):
                disease = i.get("disease", {})
                indications.append({
                    "disease_id": disease.get("id", ""),
                    "disease_name": disease.get("name", ""),
                    "phase": i.get("maxPhaseForIndication", 0)
                })
            
            # Extract linked targets and diseases
            linked_targets = [t.get("approvedSymbol", "") for t in drug.get("linkedTargets", {}).get("rows", [])]
            linked_diseases = [d.get("name", "") for d in drug.get("linkedDiseases", {}).get("rows", [])]
            
            # Build rich text content for embeddings
            text_parts = [
                f"Drug: {drug.get('name', '')}",
                f"Type: {drug.get('drugType', '')}",
                f"Max Clinical Phase: {drug.get('maximumClinicalTrialPhase', 0)}",
            ]
            
            if drug.get('description'):
                text_parts.append(f"Description: {drug.get('description')[:300]}")
            
            if mechanisms:
                text_parts.append(f"Mechanisms: {', '.join(mechanisms[:5])}")
            
            if mechanism_targets:
                unique_targets = list(set(mechanism_targets))[:10]
                text_parts.append(f"Target proteins: {', '.join(unique_targets)}")
            
            if indications:
                top_indications = [i['disease_name'] for i in indications[:10] if i.get('phase', 0) >= 3]
                if top_indications:
                    text_parts.append(f"Approved/Phase3+ Indications: {', '.join(top_indications)}")
            
            return {
                "id": drug.get("id", ""),
                "name": drug.get("name", ""),
                "description": drug.get("description", "")[:500] if drug.get("description") else "",
                "drug_type": drug.get("drugType", ""),
                "max_phase": drug.get("maximumClinicalTrialPhase", 0),
                "withdrawn": drug.get("hasBeenWithdrawn", False),
                "mechanisms": mechanisms[:10],
                "mechanism_targets": list(set(mechanism_targets))[:20],
                "indications": indications[:20],
                "linked_targets": linked_targets[:30],
                "linked_diseases": linked_diseases[:30],
                "text_content": " | ".join(text_parts),
                "entity_type": "drug_enriched",
                "source": "OpenTargets"
            }
        
        except Exception as e:
            logger.warning(f"Failed to fetch details for {drug_id}: {e}")
            return None

    def fetch_enriched_drugs(self, limit: int = 5000) -> Generator[Dict, None, None]:
        """Fetch all enriched drug data."""
        drug_ids = self.fetch_drug_ids(limit)
        
        logger.info(f"Fetching detailed data for {len(drug_ids):,} drugs...")
        
        fetched = 0
        failed = 0
        
        for i, drug_id in enumerate(drug_ids):
            if i % 100 == 0:
                logger.info(f"  Processing drug {i+1}/{len(drug_ids)} ({fetched} fetched, {failed} failed)")
            
            drug_data = self.fetch_drug_details(drug_id)
            if drug_data:
                fetched += 1
                yield drug_data
            else:
                failed += 1
        
        logger.info(f"Completed: {fetched} drugs fetched, {failed} failed")

    def fetch_drug_disease_associations(self, limit: int = 10000) -> Generator[Dict, None, None]:
        """Fetch drug-disease associations (known drug uses)."""
        logger.info(f"Fetching drug-disease associations...")
        
        # Get diseases that have known drugs
        disease_terms = [
            "cancer", "leukemia", "lymphoma", "diabetes", "arthritis",
            "asthma", "hypertension", "depression", "schizophrenia",
            "Alzheimer", "Parkinson", "HIV", "hepatitis", "COVID",
            "psoriasis", "colitis", "Crohn", "multiple sclerosis"
        ]
        
        count = 0
        seen = set()
        
        for term in disease_terms:
            if count >= limit:
                break
            
            # Search for diseases
            query = """
            query SearchDiseases($term: String!, $size: Int!) {
                search(queryString: $term, entityNames: ["disease"], page: {size: $size, index: 0}) {
                    hits { id name entity }
                }
            }
            """
            
            try:
                data = self._query(query, {"term": term, "size": 50})
                diseases = [h for h in data.get("search", {}).get("hits", []) if h.get("entity") == "disease"]
                
                for disease in diseases[:10]:
                    if count >= limit:
                        break
                    
                    # Get drugs for this disease
                    drug_query = """
                    query DiseaseKnownDrugs($diseaseId: String!, $size: Int!) {
                        disease(efoId: $diseaseId) {
                            id
                            name
                            knownDrugs(size: $size) {
                                rows {
                                    drug {
                                        id
                                        name
                                        drugType
                                        maximumClinicalTrialPhase
                                    }
                                    phase
                                    status
                                    mechanismOfAction
                                    targetName
                                }
                            }
                        }
                    }
                    """
                    
                    drug_data = self._query(drug_query, {"diseaseId": disease["id"], "size": 100})
                    disease_info = drug_data.get("disease", {})
                    
                    if not disease_info:
                        continue
                    
                    for row in disease_info.get("knownDrugs", {}).get("rows", []):
                        drug = row.get("drug", {})
                        if not drug:
                            continue
                        
                        assoc_id = f"{drug.get('id')}_{disease['id']}"
                        if assoc_id in seen:
                            continue
                        seen.add(assoc_id)
                        
                        text_content = (
                            f"Drug-Disease Association: {drug.get('name', '')} is used for {disease_info.get('name', '')} | "
                            f"Phase: {row.get('phase', 'N/A')} | Status: {row.get('status', 'N/A')} | "
                            f"Mechanism: {row.get('mechanismOfAction', 'N/A')} | Target: {row.get('targetName', 'N/A')}"
                        )
                        
                        yield {
                            "id": assoc_id,
                            "drug_id": drug.get("id", ""),
                            "drug_name": drug.get("name", ""),
                            "drug_type": drug.get("drugType", ""),
                            "drug_max_phase": drug.get("maximumClinicalTrialPhase", 0),
                            "disease_id": disease["id"],
                            "disease_name": disease_info.get("name", ""),
                            "clinical_phase": row.get("phase"),
                            "clinical_status": row.get("status", ""),
                            "mechanism_of_action": row.get("mechanismOfAction", ""),
                            "target_name": row.get("targetName", ""),
                            "text_content": text_content,
                            "entity_type": "drug_indication",
                            "source": "OpenTargets"
                        }
                        count += 1
                        
                        if count >= limit:
                            break
            
            except Exception as e:
                logger.warning(f"Failed fetching drugs for term '{term}': {e}")
        
        logger.info(f"Fetched {count:,} drug-disease associations")

    def fetch_all(self, drug_limit: int = 5000, association_limit: int = 10000) -> Generator[Dict, None, None]:
        """Fetch all enriched drug data and associations."""
        
        # Enriched drugs
        logger.info("=" * 50)
        logger.info("FETCHING ENRICHED DRUGS")
        logger.info("=" * 50)
        drug_count = 0
        for drug in self.fetch_enriched_drugs(drug_limit):
            drug_count += 1
            yield drug
        logger.info(f"Total enriched drugs: {drug_count:,}")
        
        # Drug-disease associations
        logger.info("=" * 50)
        logger.info("FETCHING DRUG-DISEASE ASSOCIATIONS")
        logger.info("=" * 50)
        assoc_count = 0
        for assoc in self.fetch_drug_disease_associations(association_limit):
            assoc_count += 1
            yield assoc
        logger.info(f"Total drug-disease associations: {assoc_count:,}")
        
        logger.info("=" * 50)
        logger.info(f"GRAND TOTAL: {drug_count + assoc_count:,}")
        logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = DrugEnrichmentFetcher()
    
    # Test with a single drug
    drug = fetcher.fetch_drug_details("CHEMBL941")  # Imatinib
    if drug:
        print("\n=== Sample Enriched Drug ===")
        print(f"Name: {drug['name']}")
        print(f"Type: {drug['drug_type']}")
        print(f"Max Phase: {drug['max_phase']}")
        print(f"Mechanisms: {drug['mechanisms'][:3]}")
        print(f"Targets: {drug['mechanism_targets'][:5]}")
        print(f"Indications: {[i['disease_name'] for i in drug['indications'][:5]]}")
        print(f"\nText content: {drug['text_content'][:300]}...")
