"""Shared infrastructure for ClinicalTrials.gov API v2 ingestion.

Reuses Qdrant, embedder, and upsert infrastructure from ot_base.
Adds CT.gov-specific API pagination, rate-limiting, field extraction,
and checkpoint helpers.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set

import requests

from agentic_ai_wf.drug_agent.opentargets.ot_base import (
    get_qdrant, get_embedder, ensure_collection, upsert_batch,
)

logger = logging.getLogger(__name__)

CT_API_BASE = "https://clinicaltrials.gov/api/v2"
CT_RATE_DELAY = 1.5          # 40 req/min — safe margin below 50/min limit
PAGE_SIZE = 1000
MAX_RETRIES = 5

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Phase enum → numeric for downstream scoring
PHASE_NUMERIC = {
    "EARLY_PHASE1": 0.5,
    "PHASE1": 1.0,
    "PHASE2": 2.0,
    "PHASE3": 3.0,
    "PHASE4": 4.0,
}


# ── Safe nested access ──────────────────────────────────────────────

def safe_get(obj: Any, *keys, default=None) -> Any:
    """Traverse nested dicts/lists safely, returning *default* on any miss."""
    cur = obj
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, (list, tuple)) and isinstance(k, int) and k < len(cur):
            cur = cur[k]
        else:
            return default
    return cur if cur is not None else default


# ── API layer ────────────────────────────────────────────────────────

def ct_api_query(session: requests.Session, endpoint: str,
                 params: Dict = None) -> Optional[Dict]:
    """Single GET against CT.gov API v2 with retry on transient errors."""
    url = f"{CT_API_BASE}/{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(CT_RATE_DELAY)
            r = session.get(url, params=params or {}, headers=_HEADERS, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                wait = CT_RATE_DELAY * (attempt + 1) * 3
                logger.warning(f"CT.gov {r.status_code}, retry in {wait:.0f}s")
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                logger.error(f"CT.gov {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            logger.warning(f"CT.gov HTTP error attempt {attempt+1}: {e}")
            time.sleep(CT_RATE_DELAY * (attempt + 1) * 2)
        except Exception as e:
            logger.warning(f"CT.gov request failed attempt {attempt+1}: {e}")
            time.sleep(CT_RATE_DELAY * (attempt + 1))
    return None


def iterate_studies(session: requests.Session,
                    filter_advanced: str,
                    extra_params: Dict = None,
                    ) -> Generator[Dict, None, None]:
    """Yield full study JSON objects, paginating via nextPageToken.

    Fetches complete records (no fields param) so we can extract
    client-side without worrying about field-path naming.

    Phase filtering is done via AREA[Phase](...) inside *filter_advanced*
    (the API has no separate ``filter.phase`` parameter).
    """
    params: Dict[str, Any] = {
        "pageSize": PAGE_SIZE,
        "format": "json",
    }
    if filter_advanced:
        params["filter.advanced"] = filter_advanced
    if extra_params:
        params.update(extra_params)

    page = 0
    total_yielded = 0
    while True:
        page += 1
        resp = ct_api_query(session, "studies", params)
        if not resp:
            logger.error(f"API returned None on page {page}, stopping iteration")
            break

        studies = resp.get("studies", [])
        if not studies:
            break

        for study in studies:
            yield study
            total_yielded += 1

        logger.info(f"Page {page}: fetched {len(studies)} studies (total: {total_yielded:,})")

        next_token = resp.get("nextPageToken")
        if not next_token:
            break
        params["pageToken"] = next_token

    logger.info(f"Iteration complete — {total_yielded:,} studies yielded across {page} pages")


# ── Field extraction helpers ─────────────────────────────────────────

def extract_drug_interventions(study: Dict) -> List[Dict]:
    """Return DRUG and BIOLOGICAL interventions from armsInterventionsModule."""
    interventions = safe_get(
        study, "protocolSection", "armsInterventionsModule", "interventions", default=[]
    )
    result = []
    for iv in interventions:
        iv_type = iv.get("type", "")
        if iv_type in ("DRUG", "BIOLOGICAL"):
            result.append({
                "name": iv.get("name", ""),
                "type": iv_type,
                "description": (iv.get("description") or "")[:300],
                "other_names": iv.get("otherNames", []),
            })
    return result


def extract_conditions(study: Dict) -> List[str]:
    return safe_get(study, "protocolSection", "conditionsModule", "conditions", default=[])


def extract_conditions_mesh(study: Dict) -> List[str]:
    meshes = safe_get(study, "derivedSection", "conditionBrowseModule", "meshes", default=[])
    return [m.get("term", "") for m in meshes if m.get("term")]


def extract_interventions_mesh(study: Dict) -> List[str]:
    meshes = safe_get(study, "derivedSection", "interventionBrowseModule", "meshes", default=[])
    return [m.get("term", "") for m in meshes if m.get("term")]


def phase_list_to_numeric(phases: List[str]) -> float:
    """Convert API phase list to a single numeric value for scoring."""
    if not phases:
        return 0.0
    nums = [PHASE_NUMERIC.get(p, 0.0) for p in phases]
    return max(nums) if len(nums) == 1 else sum(nums) / len(nums)


def phase_list_to_str(phases: List[str]) -> str:
    """Format phase list for display: ['PHASE2','PHASE3'] → 'Phase 2/Phase 3'."""
    if not phases:
        return "N/A"
    labels = [p.replace("PHASE", "Phase ").replace("EARLY_", "Early ") for p in phases]
    return "/".join(labels)


# ── Text chunking (same logic as Labels pipeline) ───────────────────

CHUNK_SIZE = 3500
CHUNK_OVERLAP = 500


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ── Checkpoint helpers ───────────────────────────────────────────────

def _ckpt_path(label: str) -> Path:
    return Path(__file__).parent / f".checkpoint_{label}.json"


def load_ct_checkpoint(label: str) -> Set[str]:
    p = _ckpt_path(label)
    if p.exists():
        data = json.loads(p.read_text())
        logger.info(f"Resuming from checkpoint: {len(data):,} NCTs already processed")
        return set(data)
    return set()


def save_ct_checkpoint(label: str, processed: Set[str]):
    _ckpt_path(label).write_text(json.dumps(sorted(processed)))
