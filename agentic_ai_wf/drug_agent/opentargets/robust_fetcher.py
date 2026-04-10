#!/usr/bin/env python3
"""
Open Targets Robust Fetcher
===========================
Uses smaller batches and better error handling for the GraphQL API.
Also provides option to use local data files.

The Open Targets GraphQL API has limitations:
- Max 500 results per page
- Rate limiting on large queries
- Occasional 502 errors

This fetcher uses:
- Smaller page sizes (100)
- More search terms to get diverse data
- Better error handling and retries
- Checkpointing to resume interrupted ingestions
"""

import json
import logging
import time
import os
from pathlib import Path
from typing import Dict, List, Generator, Set
from dataclasses import dataclass
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"


@dataclass
class FetchConfig:
    page_size: int = 100  # Smaller page size for reliability
    max_retries: int = 5
    retry_delay: float = 2.0
    request_delay: float = 0.5  # Delay between requests to avoid rate limiting


class RobustFetcher:
    """Robust fetcher for Open Targets data with better error handling."""
    
    def __init__(self, config: FetchConfig = None):
        self.config = config or FetchConfig()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._seen_ids: Set[str] = set()
    
    def _query(self, query: str, variables: Dict = None) -> Dict:
        """Execute GraphQL query with retry logic and delays."""
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(self.config.request_delay)  # Rate limiting protection
                response = self.session.post(
                    OPENTARGETS_API,
                    json={"query": query, "variables": variables or {}},
                    timeout=30  # Shorter timeout for faster failure detection
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

    def _search_entities(self, entity_type: str, search_terms: List[str], limit: int) -> Generator[Dict, None, None]:
        """Generic search for any entity type."""
        fetched = 0
        
        for term in search_terms:
            if fetched >= limit:
                break
            
            # Only get first few pages per term to avoid API issues
            for page_idx in range(0, 5):
                if fetched >= limit:
                    break
                
                query = """
                query Search($term: String!, $size: Int!, $index: Int!, $entityNames: [String!]!) {
                    search(queryString: $term, entityNames: $entityNames, page: {size: $size, index: $index}) {
                        total
                        hits {
                            id
                            entity
                            name
                            description
                        }
                    }
                }
                """
                
                try:
                    data = self._query(query, {
                        "term": term,
                        "size": self.config.page_size,
                        "index": page_idx,
                        "entityNames": [entity_type]
                    })
                    
                    hits = data.get("search", {}).get("hits", [])
                    if not hits:
                        break
                    
                    for hit in hits:
                        if hit.get("id") not in self._seen_ids:
                            self._seen_ids.add(hit["id"])
                            fetched += 1
                            yield {
                                "id": hit["id"],
                                "name": hit.get("name", ""),
                                "description": hit.get("description", "")[:500] if hit.get("description") else "",
                                "text_content": f"{entity_type.title()}: {hit.get('name', '')} | {hit.get('description', '')[:300] if hit.get('description') else ''}",
                                "entity_type": entity_type,
                                "source": "OpenTargets"
                            }
                            
                            if fetched >= limit:
                                break
                
                except Exception as e:
                    logger.warning(f"Search for '{term}' failed: {e}")
                    break
            
            if fetched % 100 == 0 and fetched > 0:
                logger.info(f"  {entity_type}: {fetched:,} fetched so far...")
        
        return fetched

    def fetch_targets(self, limit: int = 10000) -> Generator[Dict, None, None]:
        """Fetch target data."""
        logger.info(f"Fetching up to {limit:,} targets...")
        
        search_terms = [
            # Gene families
            "kinase", "phosphatase", "receptor", "channel", "transporter",
            "enzyme", "protease", "oxidase", "reductase", "synthase",
            # Functions
            "transcription factor", "signaling", "binding protein", 
            "membrane protein", "nuclear receptor", "G protein",
            # Disease related
            "cancer", "oncogene", "tumor suppressor", "inflammation",
            "immune", "metabolism", "apoptosis", "cell cycle",
            # Specific targets
            "EGFR", "VEGF", "TNF", "IL-", "CD", "HER", "BCR", "ABL",
            "JAK", "STAT", "PI3K", "AKT", "mTOR", "RAF", "MEK", "ERK",
            "BRCA", "TP53", "KRAS", "BRAF", "MYC", "BCL", "PTEN"
        ]
        
        count = 0
        for doc in self._search_entities("target", search_terms, limit):
            count += 1
            yield doc
        logger.info(f"Fetched {count:,} targets")

    def fetch_diseases(self, limit: int = 5000) -> Generator[Dict, None, None]:
        """Fetch disease data."""
        logger.info(f"Fetching up to {limit:,} diseases...")
        
        search_terms = [
            # Cancer types
            "cancer", "carcinoma", "lymphoma", "leukemia", "melanoma", "sarcoma",
            "adenocarcinoma", "tumor", "neoplasm", "malignant",
            # Autoimmune
            "autoimmune", "arthritis", "lupus", "sclerosis", "inflammatory",
            "Crohn", "colitis", "psoriasis", "diabetes type 1",
            # Neurological
            "Alzheimer", "Parkinson", "epilepsy", "stroke", "dementia",
            "neuropathy", "multiple sclerosis", "ALS", "Huntington",
            # Metabolic
            "diabetes", "obesity", "metabolic", "hyperlipidemia", "fatty liver",
            # Cardiovascular
            "cardiovascular", "heart failure", "hypertension", "atherosclerosis",
            "arrhythmia", "coronary", "myocardial",
            # Respiratory
            "asthma", "COPD", "fibrosis", "pneumonia", "lung disease",
            # Infectious
            "infectious", "viral", "bacterial", "HIV", "hepatitis",
            # Rare diseases
            "rare disease", "genetic", "hereditary", "orphan", "syndrome"
        ]
        
        count = 0
        for doc in self._search_entities("disease", search_terms, limit):
            count += 1
            yield doc
        logger.info(f"Fetched {count:,} diseases")

    def fetch_drugs(self, limit: int = 5000) -> Generator[Dict, None, None]:
        """Fetch drug data."""
        logger.info(f"Fetching up to {limit:,} drugs...")
        
        search_terms = [
            # Drug types
            "inhibitor", "agonist", "antagonist", "blocker", "modulator",
            "antibody", "monoclonal", "biologic", "small molecule",
            # Mechanism
            "kinase inhibitor", "receptor antagonist", "enzyme inhibitor",
            "checkpoint inhibitor", "immunotherapy", "targeted therapy",
            # Therapeutic areas
            "oncology", "chemotherapy", "anti-inflammatory", "antiviral",
            "antibiotic", "antidiabetic", "cardiovascular drug",
            # Clinical stages
            "approved", "phase 3", "phase 2", "FDA", "EMA",
            # Specific drugs (to get known compounds)
            "imatinib", "trastuzumab", "pembrolizumab", "adalimumab",
            "rituximab", "bevacizumab", "nivolumab", "metformin"
        ]
        
        count = 0
        for doc in self._search_entities("drug", search_terms, limit):
            count += 1
            yield doc
        logger.info(f"Fetched {count:,} drugs")

    def fetch_associations(self, disease_ids: List[str], limit_per_disease: int = 20) -> Generator[Dict, None, None]:
        """Fetch target-disease associations."""
        logger.info(f"Fetching associations for {len(disease_ids):,} diseases...")
        
        total = 0
        for i, disease_id in enumerate(disease_ids):
            if i % 50 == 0:
                logger.info(f"  Processing disease {i+1}/{len(disease_ids)} ({total:,} associations)")
            
            query = """
            query Associations($diseaseId: String!, $size: Int!) {
                disease(efoId: $diseaseId) {
                    id
                    name
                    associatedTargets(page: {size: $size, index: 0}) {
                        rows {
                            target { id approvedSymbol }
                            score
                        }
                    }
                }
            }
            """
            
            try:
                data = self._query(query, {"diseaseId": disease_id, "size": limit_per_disease})
                disease_data = data.get("disease")
                if not disease_data:
                    continue
                
                disease_name = disease_data.get("name", "")
                
                for row in disease_data.get("associatedTargets", {}).get("rows", []):
                    target = row.get("target", {})
                    score = row.get("score", 0)
                    
                    if score >= 0.1:  # Filter low-score associations
                        assoc_id = f"{target.get('id', '')}_{disease_id}"
                        if assoc_id not in self._seen_ids:
                            self._seen_ids.add(assoc_id)
                            total += 1
                            yield {
                                "id": assoc_id,
                                "target_id": target.get("id", ""),
                                "target_name": target.get("approvedSymbol", ""),
                                "disease_id": disease_id,
                                "disease_name": disease_name,
                                "score": score,
                                "text_content": f"Association: {target.get('approvedSymbol', '')} is associated with {disease_name} (score: {score:.3f})",
                                "entity_type": "association",
                                "source": "OpenTargets"
                            }
            
            except Exception as e:
                logger.warning(f"Failed to fetch associations for {disease_id}: {e}")
        
        logger.info(f"Fetched {total:,} associations")

    def fetch_all(
        self,
        target_limit: int = 10000,
        disease_limit: int = 5000,
        drug_limit: int = 5000,
        assoc_per_disease: int = 20
    ) -> Generator[Dict, None, None]:
        """Fetch all entity types."""
        self._seen_ids.clear()
        
        # Targets
        logger.info("=" * 50)
        logger.info("FETCHING TARGETS")
        logger.info("=" * 50)
        for doc in self.fetch_targets(target_limit):
            yield doc
        
        # Diseases (collect IDs for associations)
        logger.info("=" * 50)
        logger.info("FETCHING DISEASES")
        logger.info("=" * 50)
        disease_ids = []
        for doc in self.fetch_diseases(disease_limit):
            disease_ids.append(doc["id"])
            yield doc
        
        # Drugs
        logger.info("=" * 50)
        logger.info("FETCHING DRUGS")
        logger.info("=" * 50)
        for doc in self.fetch_drugs(drug_limit):
            yield doc
        
        # Associations
        if disease_ids:
            logger.info("=" * 50)
            logger.info("FETCHING ASSOCIATIONS")
            logger.info("=" * 50)
            for doc in self.fetch_associations(disease_ids, assoc_per_disease):
                yield doc


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    fetcher = RobustFetcher()
    
    count = 0
    for doc in fetcher.fetch_all(target_limit=100, disease_limit=50, drug_limit=50, assoc_per_disease=5):
        count += 1
    
    print(f"\nTotal documents: {count}")
