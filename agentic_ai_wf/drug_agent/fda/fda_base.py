"""Shared infrastructure for FDA openFDA ingestion pipelines.

Reuses Qdrant, embedder, and checkpoint infrastructure from ot_base,
adds FDA-specific download, parsing, and API query utilities.
"""

import io
import csv
import json
import time
import logging
import zipfile
from pathlib import Path
from typing import Dict, Generator, List, Optional

import requests

from agentic_ai_wf.drug_agent.opentargets.ot_base import (
    get_qdrant, get_embedder, ensure_collection, upsert_batch,
    load_drug_ids_from_qdrant, load_checkpoint, save_checkpoint,
)

logger = logging.getLogger(__name__)

OPENFDA_BASE = "https://api.fda.gov"
OPENFDA_MANIFEST_URL = f"{OPENFDA_BASE}/download.json"
ORANGE_BOOK_ZIP_URL = "https://www.fda.gov/media/76860/download"
DATA_DIR = Path(__file__).parent / "_data"

# FDA blocks bare python-requests; mimic a real browser
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def download_orange_book(dest_dir: Path = None) -> Dict[str, Path]:
    """Download and extract Orange Book ZIP → Products.txt, Patent.txt, Exclusivity.txt."""
    dest = dest_dir or _ensure_data_dir() / "orange_book"
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Orange Book ZIP from FDA...")
    resp = requests.get(ORANGE_BOOK_ZIP_URL, timeout=120, stream=True, headers=_BROWSER_HEADERS)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest)
        names = zf.namelist()

    files = {}
    for n in names:
        p = dest / n
        lower = n.lower()
        if "product" in lower:
            files["products"] = p
        elif "patent" in lower:
            files["patents"] = p
        elif "exclus" in lower:
            files["exclusivity"] = p
    logger.info(f"Orange Book extracted: {list(files.keys())}")
    return files


def parse_tilde_file(filepath: Path) -> List[Dict]:
    """Parse a tilde-delimited Orange Book file into list of dicts."""
    rows = []
    text = filepath.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="~")
    for row in reader:
        rows.append({k.strip(): v.strip() if v else "" for k, v in row.items() if k})
    logger.info(f"Parsed {len(rows):,} rows from {filepath.name}")
    return rows


def openfda_api_query(session: requests.Session, endpoint: str,
                      search: str = "", count_field: str = "",
                      limit: int = 100, max_retries: int = 5,
                      delay: float = 0.25) -> Optional[Dict]:
    """Query the openFDA REST API with retry logic.

    endpoint: e.g. "drug/event", "drug/drugsfda"
    Returns the full JSON response dict or None on failure.
    """
    url = f"{OPENFDA_BASE}/{endpoint}.json"
    params = {}
    if search:
        params["search"] = search
    if count_field:
        params["count"] = count_field
    else:
        params["limit"] = limit

    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                wait = delay * (attempt + 1) * 3
                logger.warning(f"openFDA {r.status_code}, retry in {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError:
            logger.warning(f"openFDA HTTP error on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1) * 2)
        except Exception as e:
            logger.warning(f"openFDA request failed: {e}")
            time.sleep(delay * (attempt + 1))
    return None


def get_bulk_download_urls(endpoint: str) -> List[str]:
    """Fetch the openFDA download manifest and return all partition URLs for an endpoint.

    endpoint: e.g. "drug/label", "drug/drugsfda", "drug/enforcement"
    """
    logger.info(f"Fetching openFDA manifest for {endpoint}...")
    resp = requests.get(OPENFDA_MANIFEST_URL, timeout=30, headers=_BROWSER_HEADERS)
    resp.raise_for_status()
    manifest = resp.json()

    parts = endpoint.split("/")
    node = manifest.get("results", {})
    for part in parts:
        node = node.get(part, {})

    urls = []
    for partition in node.get("partitions", []):
        if partition.get("file"):
            urls.append(partition["file"])
    logger.info(f"Found {len(urls)} bulk files for {endpoint}")
    return urls


def stream_bulk_json(endpoint: str, dest_dir: Path = None) -> Generator[Dict, None, None]:
    """Download bulk JSON ZIPs for an endpoint, yield each record."""
    dest = dest_dir or _ensure_data_dir() / endpoint.replace("/", "_")
    dest.mkdir(parents=True, exist_ok=True)

    urls = get_bulk_download_urls(endpoint)
    for i, url in enumerate(urls):
        logger.info(f"Downloading bulk file {i + 1}/{len(urls)}: {url.split('/')[-1]}")
        try:
            resp = requests.get(url, timeout=300, stream=True, headers=_BROWSER_HEADERS)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in zf.namelist():
                    if name.endswith(".json"):
                        data = json.loads(zf.read(name))
                        for record in data.get("results", []):
                            yield record
        except Exception as e:
            logger.error(f"Failed on bulk file {url}: {e}")
            continue


def checkpoint_path(label: str) -> Path:
    return Path(__file__).parent / f".checkpoint_{label}.json"


def load_fda_checkpoint(label: str) -> set:
    p = checkpoint_path(label)
    if p.exists():
        data = json.loads(p.read_text())
        logger.info(f"Resuming from checkpoint: {len(data):,} items already processed")
        return set(data)
    return set()


def save_fda_checkpoint(label: str, processed: set):
    checkpoint_path(label).write_text(json.dumps(sorted(processed)))
