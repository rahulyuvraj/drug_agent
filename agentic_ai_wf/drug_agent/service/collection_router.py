"""Routes queries to all 16 Qdrant collections with parallel execution."""

import logging
import re
import time
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

import pybreaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range, SearchParams, PayloadSchemaType, QueryRequest
from agentic_ai_wf.drug_agent.service.schemas import _GENE_TO_PROTEINS

logger = logging.getLogger(__name__)

_RESET_TIMEOUTS = [120, 240, 480, 600]


class _EscalatingListener(pybreaker.CircuitBreakerListener):
    def __init__(self, breaker: pybreaker.CircuitBreaker):
        self._breaker = breaker
        self._open_count = 0

    def state_change(self, cb, old_state, new_state):
        if new_state.name == "open":
            self._open_count = min(self._open_count + 1, len(_RESET_TIMEOUTS) - 1)
            self._breaker.reset_timeout = _RESET_TIMEOUTS[self._open_count]
            logger.warning(f"CB {cb.name} OPEN (#{self._open_count}), reset_timeout={self._breaker.reset_timeout}s")
        elif new_state.name == "closed":
            self._open_count = max(0, self._open_count - 1)


def _make_breaker(name: str) -> pybreaker.CircuitBreaker:
    cb = pybreaker.CircuitBreaker(
        fail_max=3, reset_timeout=_RESET_TIMEOUTS[0], name=name,
    )
    cb.add_listener(_EscalatingListener(cb))
    return cb


def _disease_matches(alias: str, text: str) -> bool:
    """Word-boundary match to prevent 'myopathy' matching 'cardiomyopathy'."""
    if not alias or not text:
        return False
    return bool(re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE))


ALL_COLLECTIONS = [
    "Drug_agent", "ChEMBL_drugs", "Raw_csv_KG",
    "OpenTargets_data", "OpenTargets_drugs_enriched",
    "OpenTargets_adverse_events", "OpenTargets_pharmacogenomics",
    "FDA_Orange_Book", "FDA_DrugsFDA", "FDA_FAERS",
    "FDA_Drug_Labels", "FDA_Enforcement",
    "ClinicalTrials_summaries", "ClinicalTrials_results",
    "DrugPath_KEGG", "PDR_Drugs_Data",
]


class CollectionRouter:

    def __init__(self):
        from ..opentargets.ot_base import get_qdrant, get_embedder
        self.client = get_qdrant()
        self.embedder = get_embedder()
        self._embed_cache: Dict[str, list] = {}
        self._ensembl_cache: Dict[str, str] = {}
        self._available: set = set()
        self._collection_counts: Dict[str, int] = {}
        self._queried_collections: set = set()
        self._executor = ThreadPoolExecutor(max_workers=12)
        self._intra_executor = ThreadPoolExecutor(max_workers=4)
        self._qdrant_sem = threading.Semaphore(20)
        self._breakers: Dict[str, pybreaker.CircuitBreaker] = {}
        self._search_count = 0
        self._collection_timings: Dict[str, list] = {}
        self._timing_lock = threading.Lock()
        self._brand_to_generic: Dict[str, str] = {}
        self._generic_to_brands: Dict[str, List[str]] = {}
        self._base_to_generic: Dict[str, str] = {}

        self._init_available_collections()
        for coll in self._available:
            self._breakers[coll] = _make_breaker(coll)
        self._ensure_payload_indexes()
        self._load_brand_generic_cache()
        self._load_ensembl_cache()
        self._ot_score_fields: list = []
        self._probe_ot_schema()
        logger.info(f"CollectionRouter ready — {len(self._available)}/{len(ALL_COLLECTIONS)} collections available")

    # ── Infrastructure ───────────────────────────────────────────────────────

    _PAYLOAD_INDEX_SPEC = {
        "FDA_Drug_Labels": {"section_name": PayloadSchemaType.KEYWORD, "brand_name": PayloadSchemaType.KEYWORD, "generic_name": PayloadSchemaType.KEYWORD},
        "FDA_DrugsFDA": {"brand_name": PayloadSchemaType.KEYWORD, "generic_name": PayloadSchemaType.KEYWORD},
        "OpenTargets_data": {"entity_type": PayloadSchemaType.KEYWORD, "target_name": PayloadSchemaType.KEYWORD, "disease_name": PayloadSchemaType.KEYWORD},
        "OpenTargets_drugs_enriched": {"drug_name": PayloadSchemaType.KEYWORD},
        "DrugPath_KEGG": {"gene_symbol": PayloadSchemaType.KEYWORD, "fdr": PayloadSchemaType.FLOAT},
    }

    def _init_available_collections(self):
        try:
            for c in self.client.get_collections().collections:
                if c.name in ALL_COLLECTIONS:
                    self._available.add(c.name)
                    try:
                        self._collection_counts[c.name] = self.client.count(c.name).count
                    except Exception:
                        self._collection_counts[c.name] = -1
            missing = set(ALL_COLLECTIONS) - self._available
            if missing:
                logger.warning(f"Unavailable collections: {missing}")
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")

    def _ensure_payload_indexes(self):
        for coll, fields in self._PAYLOAD_INDEX_SPEC.items():
            if coll not in self._available:
                continue
            for field, schema_type in fields.items():
                try:
                    self.client.create_payload_index(coll, field, schema_type, wait=False)
                except Exception:
                    pass

    def _load_brand_generic_cache(self):
        """Startup cache: FDA_DrugsFDA brand→generic and generic→brands maps."""
        if "FDA_DrugsFDA" not in self._available:
            return
        try:
            offset = None
            while True:
                points, offset = self.client.scroll(
                    "FDA_DrugsFDA", limit=500,
                    with_payload=["brand_name", "generic_name"], offset=offset,
                )
                for p in points:
                    brands_raw = p.payload.get("brand_name", "")
                    generics_raw = p.payload.get("generic_name", "")
                    if not brands_raw or not generics_raw:
                        continue
                    generic_primary = generics_raw.split(",")[0].strip().upper()
                    for brand in brands_raw.split(","):
                        brand = brand.strip().upper()
                        if brand and generic_primary:
                            self._brand_to_generic[brand] = generic_primary
                            self._generic_to_brands.setdefault(generic_primary, []).append(brand)
                if offset is None:
                    break
            # Self-map every generic so resolve_generic_name always returns for known generics
            for generic in self._generic_to_brands:
                if generic not in self._brand_to_generic:
                    self._brand_to_generic[generic] = generic
            # Also self-map generics that appear as *values* (brand→generic targets)
            # but are NOT keys in generic_to_brands (e.g. FAM-TRASTUZUMAB DERUXTECAN-NXKI)
            for generic_val in set(self._brand_to_generic.values()):
                if generic_val not in self._brand_to_generic:
                    self._brand_to_generic[generic_val] = generic_val
            # Add biosimilar-suffix-stripped aliases so "MARGETUXIMAB" finds "MARGETUXIMAB-CMKB"
            _bio_re = self._BIOSIMILAR_SUFFIX_RE
            for generic in list(self._brand_to_generic.keys()):
                stripped = _bio_re.sub('', generic)
                if stripped != generic and stripped not in self._brand_to_generic:
                    self._brand_to_generic[stripped] = self._brand_to_generic[generic]

            # Build base-name lookup: strip common salt suffixes and INN prefixes
            # so "tamoxifen" → TAMOXIFEN CITRATE, "eribulin" → ERIBULIN MESYLATE, etc.
            _SALT_SUFFIXES = [
                'DIHYDROCHLORIDE', 'DIMESYLATE', 'HYDROCHLORIDE', 'MESYLATE',
                'MALEATE', 'TARTRATE', 'FUMARATE', 'SUCCINATE', 'BESYLATE',
                'ACETATE', 'SULFATE', 'PHOSPHATE', 'SODIUM', 'POTASSIUM',
                'CALCIUM', 'CITRATE', 'BROMIDE', 'NITRATE', 'LACTATE',
                'TOSYLATE', 'BENZOATE', 'PAMOATE', 'DECANOATE', 'VALERATE',
            ]
            self._base_to_generic: Dict[str, str] = {}
            for generic in set(self._brand_to_generic.values()):
                for suffix in _SALT_SUFFIXES:
                    if generic.endswith(' ' + suffix):
                        base = generic[: -(len(suffix) + 1)].strip()
                        if base and base not in self._brand_to_generic:
                            self._base_to_generic.setdefault(base, generic)
                        break
            logger.info(f"Brand↔Generic cache loaded: {len(self._brand_to_generic):,} brand names, "
                        f"{len(self._generic_to_brands):,} generics, {len(self._base_to_generic):,} base-name aliases")
        except Exception as e:
            logger.warning(f"Brand↔Generic cache load failed: {e}")

    _BIOSIMILAR_SUFFIX_RE = re.compile(r'-[a-z]{4}$', re.IGNORECASE)

    def resolve_generic_name(self, drug_name: str) -> Optional[str]:
        """Resolve a drug name to its generic/INN via cached FDA brand→generic map.

        Resolution order:
        1. Exact brand/generic lookup
        2. Strip WHO biosimilar suffix (-dkst, -anns) and retry
        3. Salt-stripped base name lookup (tamoxifen → TAMOXIFEN CITRATE)
        """
        key = drug_name.upper().strip()
        # 1. Exact lookup
        generic = self._brand_to_generic.get(key)
        if not generic:
            # 2. Strip biosimilar suffix
            stripped = self._BIOSIMILAR_SUFFIX_RE.sub('', key)
            if stripped != key:
                generic = self._brand_to_generic.get(stripped)
        if not generic:
            # 3. Base-name (salt-stripped) fallback
            generic = self._base_to_generic.get(key)
            if not generic:
                stripped = self._BIOSIMILAR_SUFFIX_RE.sub('', key)
                if stripped != key:
                    generic = self._base_to_generic.get(stripped)
        if not generic:
            # 4. Word-level splitting for multi-word / combination names
            words = re.sub(r'\b(?:AND|WITH|OR|PLUS)\b', ' ', key).split()
            for w in sorted(words, key=len, reverse=True):
                if len(w) <= 2:
                    continue
                generic = self._brand_to_generic.get(w) or self._base_to_generic.get(w)
                if generic:
                    break
        if generic:
            return self._BIOSIMILAR_SUFFIX_RE.sub('', generic)
        return None

    def _load_ensembl_cache(self):
        if "OpenTargets_data" not in self._available:
            logger.warning("OpenTargets_data unavailable — Ensembl cache empty")
            return
        try:
            offset = None
            filt = Filter(must=[FieldCondition(key="entity_type", match=MatchValue(value="target"))])
            while True:
                points, offset = self.client.scroll(
                    "OpenTargets_data", scroll_filter=filt, limit=1000,
                    with_payload=["id", "name"], offset=offset,
                )
                for p in points:
                    gene = p.payload.get("name", "")
                    ens = p.payload.get("id", "")
                    if gene and ens:
                        self._ensembl_cache[gene.upper()] = ens
                if offset is None:
                    break
            logger.info(f"Ensembl cache loaded: {len(self._ensembl_cache):,} genes")
        except Exception as e:
            logger.warning(f"Ensembl cache load failed: {e}")

    def _probe_ot_schema(self):
        """Sample OpenTargets_data association records to discover available score fields."""
        if "OpenTargets_data" not in self._available:
            return
        try:
            filt = Filter(must=[FieldCondition(key="entity_type", match=MatchValue(value="association"))])
            points, _ = self.client.scroll("OpenTargets_data", scroll_filter=filt, limit=3, with_payload=True)
            all_keys: set = set()
            for p in points:
                all_keys.update(p.payload.keys())
            score_candidates = [k for k in sorted(all_keys) if "score" in k.lower() or "overall" in k.lower()]
            self._ot_score_fields = score_candidates
            logger.info(f"OT probe: {len(points)} samples, score-like fields={score_candidates}, all keys={sorted(all_keys)}")
        except Exception as e:
            logger.warning(f"OT schema probe failed: {e}")

    def _embed(self, text: str) -> list:
        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached
        vec = self.embedder.encode(text)
        result = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        self._embed_cache[text] = result
        return result

    def batch_prewarm(self, texts: List[str], batch_size: int = 64):
        """Pre-warm the embedding cache with a batch encode (single call to GPU)."""
        needed = [t for t in texts if t not in self._embed_cache]
        if not needed:
            return
        for i in range(0, len(needed), batch_size):
            chunk = needed[i:i + batch_size]
            vecs = self.embedder.encode(chunk, batch_size=batch_size)
            for t, v in zip(chunk, vecs):
                self._embed_cache[t] = v.tolist() if hasattr(v, 'tolist') else list(v)
        logger.info(f"Pre-warmed {len(needed)} embeddings")

    @staticmethod
    def _ascii_fold(text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').replace("'", "").replace("\u2019", "")

    def _get_collection_timeout(self, collection: str) -> int:
        count = self._collection_counts.get(collection, -1)
        if count > 200_000:
            return 30
        if count > 50_000:
            return 20
        return 15

    def _search(self, collection: str, query_text: str,
                top_k: int = 10, qfilter: Optional[Filter] = None,
                precomputed_vec: Optional[list] = None) -> List[Dict]:
        if collection not in self._available:
            return []
        self._queried_collections.add(collection)
        cb = self._breakers.get(collection)
        if cb and cb.current_state == "open":
            logger.info(f"CB {collection} open — skipping search")
            return []
        # Pre-compute embedding BEFORE acquiring semaphore to avoid GIL
        # contention under concurrent threads (encode is CPU-bound).
        if precomputed_vec is None:
            precomputed_vec = self._embed(query_text)
        t0 = time.perf_counter()
        self._qdrant_sem.acquire()
        t_sem = time.perf_counter()
        try:
            result = self._search_with_retry(collection, query_text, top_k, qfilter, precomputed_vec=precomputed_vec)
            return result
        except pybreaker.CircuitBreakerError:
            logger.warning(f"CB {collection} tripped — returning empty")
            return []
        except Exception as e:
            logger.warning(f"Search failed on {collection} after retries: {e}")
            return []
        finally:
            self._qdrant_sem.release()
            elapsed = time.perf_counter() - t0
            sem_wait = t_sem - t0
            qdrant_time = time.perf_counter() - t_sem
            with self._timing_lock:
                self._search_count += 1
                self._collection_timings.setdefault(collection, []).append(elapsed)
                # Track sem wait vs qdrant time separately
                self._collection_timings.setdefault(f"_sem_{collection}", []).append(sem_wait)
                self._collection_timings.setdefault(f"_net_{collection}", []).append(qdrant_time)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4),
           retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
           reraise=True)
    def _search_with_retry(self, collection: str, query_text: str,
                           top_k: int, qfilter: Optional[Filter],
                           precomputed_vec: Optional[list] = None) -> List[Dict]:
        cb = self._breakers.get(collection)
        def _inner():
            vec = precomputed_vec if precomputed_vec is not None else self._embed(query_text)
            results = self.client.query_points(
                collection_name=collection, query=vec,
                query_filter=qfilter, limit=top_k, with_payload=True,
                search_params=SearchParams(hnsw_ef=128),
                timeout=self._get_collection_timeout(collection),
            )
            return [{"score": r.score, "payload": r.payload} for r in results.points]
        if cb:
            return cb.call(_inner)
        return _inner()

    def reset_query_tracking(self):
        self._queried_collections.clear()
        with self._timing_lock:
            self._search_count = 0
            self._collection_timings.clear()

    def get_queried_collections(self) -> set:
        return set(self._queried_collections)

    def get_timing_summary(self) -> Dict:
        with self._timing_lock:
            summary = {}
            total_time = 0.0
            total_sem_wait = 0.0
            total_net_time = 0.0
            for coll, times in sorted(self._collection_timings.items()):
                if coll.startswith("_sem_") or coll.startswith("_net_"):
                    continue
                s = sum(times)
                total_time += s
                avg = s / len(times)
                sem_times = self._collection_timings.get(f"_sem_{coll}", [])
                net_times = self._collection_timings.get(f"_net_{coll}", [])
                sem_avg = sum(sem_times) / len(sem_times) if sem_times else 0
                net_avg = sum(net_times) / len(net_times) if net_times else 0
                total_sem_wait += sum(sem_times)
                total_net_time += sum(net_times)
                summary[coll] = {
                    "calls": len(times), "total_s": round(s, 2), "avg_s": round(avg, 3),
                    "sem_avg_s": round(sem_avg, 4), "net_avg_s": round(net_avg, 3),
                }
            return {
                "search_count": self._search_count,
                "total_search_s": round(total_time, 2),
                "total_sem_wait_s": round(total_sem_wait, 2),
                "total_net_s": round(total_net_time, 2),
                "per_collection": summary,
            }

    def _parallel_search(self, queries: List[Tuple[str, str, int, Optional[Filter]]],
                         precomputed_vec: Optional[list] = None) -> Dict[str, List[Dict]]:
        """Execute multiple (collection, query_text, top_k, filter) searches in parallel."""
        results: Dict[str, List[Dict]] = {}
        futures = {}
        for coll, text, k, filt in queries:
            key = f"{coll}|{text[:40]}"
            futures[self._executor.submit(self._search, coll, text, k, filt, precomputed_vec=precomputed_vec)] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logger.warning(f"Parallel search failed for {key}: {e}")
                results[key] = []
        counts = {k.split('|')[0]: len(v) for k, v in results.items()}
        logger.debug(f"Parallel search: {counts}")
        return results

    # ── Gene-Gene Functional Relationships (KG) ─────────────────────────────

    _KG_GENE_TYPE = "gene/protein"

    def get_functionally_related_genes(self, gene_symbol: str) -> List[str]:
        """Resolve functionally related genes via Raw_csv_KG PPI triples + semantic search."""
        cache_key = f"_effectors_{gene_symbol.upper()}"
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        if "Raw_csv_KG" not in self._available:
            self._embed_cache[cache_key] = []
            return []

        related: set = set()
        gene_upper = gene_symbol.upper()

        # Payload-filtered scroll: exact gene name on either side of KG triple
        for side, other in [("x", "y"), ("y", "x")]:
            try:
                filt = Filter(must=[
                    FieldCondition(key=f"{side}_name", match=MatchValue(value=gene_symbol)),
                    FieldCondition(key=f"{side}_type", match=MatchValue(value=self._KG_GENE_TYPE)),
                    FieldCondition(key=f"{other}_type", match=MatchValue(value=self._KG_GENE_TYPE)),
                ])
                self._queried_collections.add("Raw_csv_KG")
                self._qdrant_sem.acquire()
                try:
                    pts, _ = self.client.scroll(
                        "Raw_csv_KG", scroll_filter=filt, limit=200, with_payload=True)
                finally:
                    self._qdrant_sem.release()
                for p in pts:
                    name = (p.payload.get(f"{other}_name") or "").upper()
                    if name and name != gene_upper:
                        related.add(name)
            except Exception:
                pass

        # Semantic search for broader coverage (still filtered to gene↔gene triples)
        try:
            vec = self._embed(f"{gene_symbol} protein interaction signaling")
            self._queried_collections.add("Raw_csv_KG")
            self._qdrant_sem.acquire()
            try:
                results = self.client.query_points(
                    "Raw_csv_KG", query=vec, limit=50, with_payload=True,
                    query_filter=Filter(must=[
                        FieldCondition(key="x_type", match=MatchValue(value=self._KG_GENE_TYPE)),
                        FieldCondition(key="y_type", match=MatchValue(value=self._KG_GENE_TYPE)),
                    ]),
                )
            finally:
                self._qdrant_sem.release()
            for r in results.points:
                pl = r.payload
                for s in ("x", "y"):
                    name = (pl.get(f"{s}_name") or "").upper()
                    if name and name != gene_upper:
                        related.add(name)
        except Exception:
            pass

        result = sorted(related)
        self._embed_cache[cache_key] = result
        logger.debug(f"KG effectors for {gene_symbol}: {len(result)} related genes")
        return result

    # ── Stage 1: Candidate Discovery ─────────────────────────────────────────

    def find_drugs_for_target(self, gene_symbol: str, top_k: int = 10) -> List[Dict]:
        """Find drugs targeting a gene — queries 5 collections in parallel."""
        gene_up = gene_symbol.upper()
        # Build protein alias set for synonym-expanded search
        protein_aliases = _GENE_TO_PROTEINS.get(gene_up, [])
        alias_names = [a.upper() for a in protein_aliases]  # e.g. ["CD20"] for MS4A1
        all_names = [gene_up] + alias_names  # all names to verify against
        # Semantic queries include aliases so Qdrant matches FDA labels like "anti-CD20"
        alias_clause = " ".join(protein_aliases) if protein_aliases else ""
        semantic_q = f"{gene_symbol} {alias_clause} inhibitor drug target".strip()

        # ChEMBL uses a wider Qdrant limit because biosimilars and salt
        # forms often dominate the top-k, pushing genuine drugs below the
        # cut-off.  Other collections are less duplicative.
        chembl_limit = top_k * 3

        queries = [
            ("Drug_agent", f"{gene_symbol} {alias_clause} drug target therapy".strip(),
             top_k, Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="gene_drug"))])),
            ("ChEMBL_drugs", semantic_q, chembl_limit, None),
            ("OpenTargets_drugs_enriched", f"{gene_symbol} {alias_clause} drug mechanism".strip(), top_k, None),
            ("FDA_Drug_Labels", f"{gene_symbol} {alias_clause}".strip(), top_k,
             Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="mechanism_of_action"))])),
            ("Raw_csv_KG", f"{gene_symbol} {alias_clause} drug target interaction".strip(), top_k, None),
            ("DrugPath_KEGG", f"{gene_symbol} drug pathway target",
             top_k, Filter(must=[
                 FieldCondition(key="gene_symbol", match=MatchValue(value=gene_symbol)),
                 FieldCondition(key="fdr", range=Range(lte=0.25)),
             ])),
            ("PDR_Drugs_Data", f"{gene_symbol} drug target mechanism",
             top_k, Filter(must=[
                 FieldCondition(key="gene_symbol", match=MatchValue(value=gene_symbol)),
                 FieldCondition(key="doc_type", match=MatchValue(value="pdr_drug_target")),
             ])),
        ]

        raw = self._parallel_search(queries)
        candidates = []
        seen: Dict[str, int] = {}  # norm → index in candidates
        # Track chembl_ids for names first seen from non-ChEMBL sources,
        # so the ID isn't lost when ChEMBL re-discovers the same name.
        seen_chembl: Dict[str, str] = {}

        # Process structured-target collections first so their higher-quality
        # entries win the dedup race (parallel search returns in arbitrary order).
        _PRIORITY = {"ChEMBL_drugs": 0, "OpenTargets_drugs_enriched": 1, "Drug_agent": 2}
        sorted_keys = sorted(raw.keys(), key=lambda k: _PRIORITY.get(k.split("|")[0], 9))

        for key in sorted_keys:
            hits = raw[key]
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                drugs = self._extract_drug_names(p, coll)
                # ChEMBL: only primary drug_name — synonyms (biosimilar codes)
                # create dedup noise (e.g. CT-P10 competing with RITUXIMAB)
                if coll == "ChEMBL_drugs":
                    drugs = drugs[:1]
                for dname in drugs:
                    norm = dname.upper().strip()
                    if not norm:
                        continue
                    if norm in seen:
                        if coll == "ChEMBL_drugs" and p.get("chembl_id"):
                            seen_chembl.setdefault(norm, p["chembl_id"])
                        continue

                    # Verify gene relevance for collections that have structured target data
                    if coll == "ChEMBL_drugs":
                        targets = [t.upper() for t in p.get("target_gene_symbols", [])]
                        if targets and not any(n in targets for n in all_names):
                            continue
                    elif coll == "OpenTargets_drugs_enriched":
                        lt = [t.upper() for t in p.get("linked_targets", []) + p.get("mechanism_targets", [])]
                        ensembl = self._ensembl_cache.get(gene_up, "")
                        if lt and not any(n in lt for n in all_names) and ensembl not in [x.upper() for x in p.get("linked_targets", [])]:
                            continue
                    elif coll == "Raw_csv_KG":
                        if not any(self._kg_involves_entity(p, n, {"gene", "protein"}) for n in all_names):
                            continue
                    elif coll == "DrugPath_KEGG":
                        if p.get("gene_symbol", "").upper() not in all_names:
                            continue
                    elif coll == "PDR_Drugs_Data":
                        if p.get("gene_symbol", "").upper() not in all_names:
                            continue

                    seen[norm] = len(candidates)
                    candidates.append({
                        "drug_name": dname,
                        "source": coll,
                        "action_type": self._extract_action_type(p, coll),
                        "mechanism": p.get("mechanism_of_action", p.get("text_content", ""))[:200],
                        "phase": p.get("max_phase", p.get("max_phase", None)),
                        "score": h["score"],
                        "gene_symbol": gene_symbol,
                        "chembl_id": p.get("chembl_id") if coll == "ChEMBL_drugs" else None,
                    })

        # Backfill chembl_ids captured from seen-but-skipped ChEMBL hits
        for c in candidates:
            if not c.get("chembl_id"):
                cid = seen_chembl.get(c["drug_name"].upper().strip())
                if cid:
                    c["chembl_id"] = cid

        per_coll = {}
        for c in candidates:
            per_coll[c["source"]] = per_coll.get(c["source"], 0) + 1
        logger.info(f"find_drugs_for_target({gene_symbol}): {per_coll}")
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k * 5]

    def find_drugs_for_disease(self, disease_name: str, top_k: int = 10,
                                disease_aliases: Optional[List[str]] = None,
                                skip_reverse_scans: bool = False) -> List[Dict]:
        """Find drugs associated with a disease — queries 5 collections per alias."""
        search_terms = [disease_name] + (disease_aliases or [])[:5]
        all_queries = []
        primary_keys: set = set()
        for i, term in enumerate(dict.fromkeys(t.lower() for t in search_terms)):
            # OT gets a larger top_k to catch drugs with secondary disease
            # indications that have lower vector similarity (e.g., Belimumab
            # stored under SLE but indicated for Sjögren's).
            ot_k = max(top_k * 3, 25)
            term_queries = [
                ("Drug_agent", f"{term} drug treatment therapy",
                 top_k, Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="disease_drug"))])),
                ("ClinicalTrials_summaries", f"{term} drug treatment", top_k, None),
                ("OpenTargets_drugs_enriched", term, ot_k, None),
                ("FDA_Drug_Labels", term, top_k,
                 Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="indications_and_usage"))])),
                ("Raw_csv_KG", f"{term} treatment drug therapy", top_k, None),
                ("PDR_Drugs_Data", f"{term} treatment indication drug", top_k, None),
            ]
            all_queries.extend(term_queries)
            if i == 0:
                primary_keys = {f"{c}|{t[:40]}" for c, t, _, _ in term_queries}

        raw = self._parallel_search(all_queries)
        drug_map: Dict[str, Dict] = {}
        # Build alias set for OT indication matching
        _alias_set = {self._ascii_fold(t).lower() for t in search_terms}

        for key, hits in raw.items():
            coll = key.split("|")[0]
            is_primary = key in primary_keys
            for h in hits:
                p = h["payload"]
                drugs = self._extract_drug_names(p, coll)

                # OT indication check: if this OT drug has the disease in its
                # indications list, force it as primary (disease-indicated drug)
                ot_indicated = False
                if coll == "OpenTargets_drugs_enriched":
                    for ind in p.get("indications", []):
                        ind_name = self._ascii_fold(ind.get("disease_name", "")).lower()
                        if any(_disease_matches(a, ind_name) or _disease_matches(ind_name, a) for a in _alias_set):
                            ot_indicated = True
                            break

                for dname in drugs:
                    norm = dname.upper().strip()
                    if not norm:
                        continue
                    if coll == "Raw_csv_KG" and not self._kg_involves_entity(p, "", {"drug", "compound"}):
                        continue
                    if norm in drug_map:
                        drug_map[norm]["score"] = max(drug_map[norm]["score"], h["score"])
                        if is_primary or ot_indicated:
                            drug_map[norm]["_primary"] = True
                        continue
                    drug_map[norm] = {
                        "drug_name": dname,
                        "source": coll,
                        "indication": disease_name,
                        "phase": p.get("max_phase"),
                        "score": h["score"],
                        "chembl_id": p.get("chembl_id") if coll == "ChEMBL_drugs" else None,
                        "_primary": is_primary or ot_indicated,
                    }

        # ── OT indication reverse scan ──
        # Gene-inferred diseases already have strong gene-based discovery;
        # reverse scans add noise and ~15 extra queries per alias.
        ot_scan_queries = []
        if not skip_reverse_scans:
            for term in list(dict.fromkeys(t.lower() for t in search_terms))[:3]:
                ot_scan_queries.extend([
                    ("OpenTargets_drugs_enriched", f"{term} treatment drug therapy indication", 50, None),
                    ("OpenTargets_drugs_enriched", f"{term} approved drug FDA", 50, None),
                    ("OpenTargets_drugs_enriched", f"autoimmune {term} monoclonal antibody biologic", 30, None),
                ])
        ot_raw = self._parallel_search(ot_scan_queries)
        ot_indicated_count = 0
        for key, hits in ot_raw.items():
            for h in hits:
                p = h["payload"]
                for ind in p.get("indications", []):
                    ind_name = self._ascii_fold(ind.get("disease_name", "")).lower()
                    if not any(_disease_matches(a, ind_name) or _disease_matches(ind_name, a) for a in _alias_set):
                        continue
                    drugs = self._extract_drug_names(p, "OpenTargets_drugs_enriched")
                    for dname in drugs:
                        norm = dname.upper().strip()
                        if not norm:
                            continue
                        if norm not in drug_map:
                            drug_map[norm] = {
                                "drug_name": dname,
                                "source": "OpenTargets_drugs_enriched",
                                "indication": disease_name,
                                "phase": ind.get("phase") or p.get("max_phase"),
                                "score": h["score"],
                                "chembl_id": p.get("chembl_id"),
                                "_primary": True,
                            }
                            ot_indicated_count += 1
                        else:
                            drug_map[norm]["_primary"] = True
                    break
        if ot_indicated_count:
            logger.info(f"OT indication scan: found {ot_indicated_count} new indicated drugs")

        # ── FDA label indication reverse scan ──
        fda_filt = Filter(must=[FieldCondition(
            key="section_name", match=MatchValue(value="indications_and_usage"))])
        fda_scan_queries = []
        if not skip_reverse_scans:
            for term in list(dict.fromkeys(t.lower() for t in search_terms))[:3]:
                fda_scan_queries.extend([
                    ("FDA_Drug_Labels", f"{term} treatment symptoms medication drug", 50, fda_filt),
                    ("FDA_Drug_Labels", f"{term} FDA approved indication therapy", 50, fda_filt),
                ])
        fda_raw = self._parallel_search(fda_scan_queries)
        fda_indicated_count = 0
        for key, hits in fda_raw.items():
            for h in hits:
                p = h["payload"]
                txt = self._ascii_fold(p.get("text_content", "")).lower()
                if not any(_disease_matches(a, txt) for a in _alias_set):
                    continue
                # Extract drug name from generic_name or brand_name
                gn = (p.get("generic_name") or "").strip()
                bn = (p.get("brand_name") or "").strip()
                dname = gn or bn
                if not dname:
                    continue
                # Normalise: take the first word of multi-salt names
                norm = dname.split()[0].upper() if dname else ""
                if not norm or len(norm) < 3:
                    continue
                # FDA label mentioning the disease → score at least 0.5
                # to ensure it survives the cap alongside gene-based drugs
                boosted_score = max(h["score"], 0.50)
                if norm not in drug_map:
                    drug_map[norm] = {
                        "drug_name": dname.split()[0].title(),
                        "source": "FDA_Drug_Labels",
                        "indication": disease_name,
                        "phase": 4,  # FDA-labelled ⇒ approved
                        "score": boosted_score,
                        "chembl_id": None,
                        "_primary": True,
                    }
                    fda_indicated_count += 1
                else:
                    drug_map[norm]["_primary"] = True
                    drug_map[norm]["score"] = max(drug_map[norm]["score"], boosted_score)
        if fda_indicated_count:
            logger.info(f"FDA indication scan: found {fda_indicated_count} new indicated drugs")

        # ── OT association disease-name scroll (catch-all for drugs with direct disease links) ──
        # Catches drugs like Mycophenolate whose gene targets (IMPDH) may not be in the
        # patient's DEG list but have strong OT disease associations.
        if "OpenTargets_data" in self._available:
            ot_assoc_count = 0
            try:
                ot_dis_filter = Filter(must=[
                    FieldCondition(key="entity_type", match=MatchValue(value="association")),
                ])
                self._queried_collections.add("OpenTargets_data")
                points, _ = self.client.scroll(
                    "OpenTargets_data", scroll_filter=ot_dis_filter,
                    limit=200, with_payload=True,
                )
                for pt in points:
                    p = pt.payload
                    dn = (p.get("disease_name") or "").lower()
                    if not any(_disease_matches(a, dn) for a in _alias_set):
                        continue
                    # This association is for our disease — extract the drug if possible
                    tn = (p.get("target_name") or "").strip()
                    if not tn:
                        continue
                    # target_name is a gene symbol; we need to find drugs targeting it
                    # The score tells us how strong the gene-disease link is
                    ot_score_val = None
                    for sf in (self._ot_score_fields or ["score"]):
                        ot_score_val = p.get(sf)
                        if ot_score_val is not None:
                            break
                    if ot_score_val is not None:
                        try:
                            ot_score_val = float(ot_score_val)
                        except (ValueError, TypeError):
                            ot_score_val = None
                    # Store gene-disease associations for downstream enrichment
                    # (drugs for these genes are discovered in Stage 2 enrichment)
            except Exception as e:
                logger.warning(f"OT association disease scroll failed: {e}")

        # Primary disease drugs retained first; alias-only drugs fill remaining slots
        ranked = sorted(drug_map.values(),
                        key=lambda x: (not x.get("_primary"), -x["score"]))
        cap = top_k * 5
        result = ranked[:cap]
        for r in result:
            r.pop("_primary", None)

        per_coll = {}
        for c in result:
            per_coll[c["source"]] = per_coll.get(c["source"], 0) + 1
        logger.info(f"find_drugs_for_disease({disease_name}): {per_coll}")
        return result

    def get_pathway_drugs(self, pathway_name: str, key_genes: Optional[List[str]] = None, top_k: int = 10) -> List[Dict]:
        """Find drugs targeting a pathway — queries Drug_agent + Raw_csv_KG + DrugPath_KEGG."""
        queries = [
            ("Drug_agent", f"{pathway_name} drug target",
             top_k, Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="pathway_drug"))])),
            ("Raw_csv_KG", f"{pathway_name} drug treatment pathway", top_k, None),
            ("DrugPath_KEGG", f"{pathway_name} drug pathway",
             top_k, Filter(must=[FieldCondition(key="fdr", range=Range(lte=0.25))])),
        ]

        raw = self._parallel_search(queries)
        candidates = []
        seen = set()
        key_genes_upper = {g.upper() for g in key_genes} if key_genes else None

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                # When key_genes specified, boost KEGG hits matching those genes
                hit_gene = (p.get("gene_symbol") or "").upper()
                gene_bonus = 0.05 if key_genes_upper and hit_gene in key_genes_upper else 0
                drugs = self._extract_drug_names(p, coll)
                for dname in drugs:
                    norm = dname.upper().strip()
                    if norm and norm not in seen:
                        seen.add(norm)
                        candidates.append({
                            "drug_name": dname, "source": coll,
                            "pathway": pathway_name, "score": h["score"] + gene_bonus,
                            "gene_symbol": hit_gene or "",
                        })

        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    def get_pathway_member_genes(self, pathway_name: str, top_k: int = 50) -> List[str]:
        """Extract member genes of a pathway from DrugPath_KEGG."""
        cache_key = f"_pathway_genes_{pathway_name}"
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        if "DrugPath_KEGG" not in self._available:
            self._embed_cache[cache_key] = []
            return []

        fdr_filter = Filter(must=[FieldCondition(key="fdr", range=Range(lte=0.25))])
        hits = self._search("DrugPath_KEGG", pathway_name, top_k, qfilter=fdr_filter)
        genes = set()
        for h in hits:
            gs = (h["payload"].get("gene_symbol") or "").upper().strip()
            if gs:
                genes.add(gs)

        result = sorted(genes)[:20]
        self._embed_cache[cache_key] = result
        return result

    def get_target_pathways(self, gene_symbol: str, top_k: int = 20) -> List[str]:
        """Get pathway names for a gene from DrugPath_KEGG."""
        cache_key = f"_target_pw_{gene_symbol}"
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        if "DrugPath_KEGG" not in self._available:
            self._embed_cache[cache_key] = []
            return []

        fdr_filter = Filter(must=[
            FieldCondition(key="gene_symbol", match=MatchValue(value=gene_symbol)),
            FieldCondition(key="fdr", range=Range(lte=0.25)),
        ])
        hits = self._search("DrugPath_KEGG", f"{gene_symbol} pathway", top_k, qfilter=fdr_filter)
        pathways = set()
        for h in hits:
            pn = (h["payload"].get("pathway_name") or "").strip()
            if pn:
                pathways.add(pn)

        result = sorted(pathways)[:10]
        self._embed_cache[cache_key] = result
        return result

    def find_drugs_by_pathway(self, pathway_names: List[str], top_k: int = 30) -> List[Dict]:
        """Cross-ontology pathway→drug discovery via DrugPath_KEGG (semantic + FDR gate)."""
        if "DrugPath_KEGG" not in self._available:
            return []
        fdr_filter = Filter(must=[FieldCondition(key="fdr", range=Range(lte=0.25))])
        queries = [
            ("DrugPath_KEGG", pname, top_k, fdr_filter)
            for pname in pathway_names[:10]
        ]
        raw = self._parallel_search(queries)
        candidates = []
        seen = set()
        for key, hits in raw.items():
            for h in hits:
                p = h["payload"]
                dname = p.get("drug_name", "")
                norm = dname.upper().strip()
                if norm and norm not in seen:
                    seen.add(norm)
                    candidates.append({
                        "drug_name": dname, "source": "DrugPath_KEGG",
                        "pathway": p.get("pathway_name", ""),
                        "gene_symbol": p.get("gene_symbol", ""),
                        "score": h["score"],
                    })
        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    # ── Stage 2: Evidence Enrichment ─────────────────────────────────────────

    def get_drug_identity(self, drug_name: str) -> Dict:
        """Merge identity from ChEMBL + FDA_DrugsFDA + OT_drugs_enriched + Orange Book."""
        queries = [
            ("ChEMBL_drugs", drug_name, 3, None),
            ("FDA_DrugsFDA", drug_name, 3, None),
            ("OpenTargets_drugs_enriched", drug_name, 3, None),
            ("FDA_Orange_Book", drug_name, 5, None),
            ("PDR_Drugs_Data", drug_name, 3,
             Filter(must=[FieldCondition(key="drug_name_lower", match=MatchValue(value=drug_name.lower()))])),
        ]
        raw = self._parallel_search(queries)
        identity: Dict = {
            "drug_name": drug_name, "chembl_id": None, "drug_type": None,
            "max_phase": None, "first_approval": None, "is_fda_approved": False,
            "brand_names": [], "patent_count": 0, "exclusivity_count": 0,
            "generics_available": False, "pharm_class_moa": None,
            "pharm_class_epc": None, "withdrawn": False,
        }

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                name_match = self._fuzzy_drug_match(drug_name, p, coll)
                if not name_match:
                    continue

                if coll == "ChEMBL_drugs":
                    identity["chembl_id"] = identity["chembl_id"] or p.get("chembl_id")
                    identity["drug_type"] = identity["drug_type"] or p.get("molecule_type")
                    identity["max_phase"] = max(filter(None, [identity["max_phase"], p.get("max_phase")]), default=None)
                    identity["first_approval"] = identity["first_approval"] or p.get("first_approval")
                    if p.get("approval_status", "").upper().startswith("FDA"):
                        identity["is_fda_approved"] = True
                    # Capture ChEMBL canonical drug_name (often the INN)
                    chembl_dn = p.get("drug_name", "")
                    if chembl_dn and not identity.get("chembl_drug_name"):
                        identity["chembl_drug_name"] = chembl_dn
                    for syn in p.get("synonyms", []):
                        if syn and syn not in identity["brand_names"]:
                            identity["brand_names"].append(syn)

                elif coll == "FDA_DrugsFDA":
                    identity["is_fda_approved"] = True
                    identity["pharm_class_moa"] = identity["pharm_class_moa"] or p.get("pharm_class_moa")
                    identity["pharm_class_epc"] = identity["pharm_class_epc"] or p.get("pharm_class_epc")
                    for bn in [p.get("brand_name", "")]:
                        if bn and bn not in identity["brand_names"]:
                            identity["brand_names"].append(bn)

                elif coll == "OpenTargets_drugs_enriched":
                    identity["drug_type"] = identity["drug_type"] or p.get("drug_type")
                    identity["max_phase"] = max(filter(None, [identity["max_phase"], p.get("max_phase")]), default=None)
                    identity["withdrawn"] = identity["withdrawn"] or p.get("withdrawn", False)

                elif coll == "FDA_Orange_Book":
                    identity["patent_count"] = max(identity["patent_count"], p.get("patent_count", 0))
                    identity["exclusivity_count"] = max(identity["exclusivity_count"], p.get("exclusivity_count", 0))
                    if p.get("approval_date"):
                        identity["is_fda_approved"] = True
                    trade = p.get("trade_name", "")
                    if trade and trade not in identity["brand_names"]:
                        identity["brand_names"].append(trade)
                    if p.get("nda_type", "").upper() == "A":
                        identity["generics_available"] = True

                elif coll == "PDR_Drugs_Data":
                    identity["pdr_moa"] = identity.get("pdr_moa") or p.get("mechanism_of_action", "")
                    identity["pdr_indication"] = identity.get("pdr_indication") or p.get("indication", "")
                    identity["pdr_route"] = identity.get("pdr_route") or p.get("route", "")
                    identity["pdr_is_anticancer"] = identity.get("pdr_is_anticancer") or p.get("is_anticancer", False)
                    identity["pdr_efficacy"] = identity.get("pdr_efficacy") or p.get("efficacy", "")

        return identity

    def get_drug_targets(self, drug_name: str) -> List[Dict]:
        """Get target genes — ChEMBL + FDA Labels MoA + Drug_agent + Raw_csv_KG."""
        queries = [
            ("ChEMBL_drugs", drug_name, 5, None),
            ("FDA_Drug_Labels", drug_name, 3,
             Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="mechanism_of_action"))])),
            ("Drug_agent", f"{drug_name} gene target", 5,
             Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="gene_drug"))])),
            ("Raw_csv_KG", f"{drug_name} target gene protein", 5, None),
            ("PDR_Drugs_Data", f"{drug_name} gene target", 5,
             Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="pdr_drug_target"))])),
        ]
        raw = self._parallel_search(queries)
        targets: Dict[str, Dict] = {}

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                if not self._fuzzy_drug_match(drug_name, p, coll):
                    continue

                if coll == "ChEMBL_drugs":
                    genes = p.get("target_gene_symbols", [])
                    actions = p.get("action_types", [])
                    moa = p.get("mechanism_of_action", "")
                    for i, gene in enumerate(genes):
                        gu = gene.upper()
                        if gu not in targets:
                            targets[gu] = {"gene_symbol": gene, "action_type": actions[i] if i < len(actions) else "UNKNOWN", "mechanism": moa}
                        else:
                            existing = targets[gu]["action_type"]
                            incoming = actions[i] if i < len(actions) else ""
                            if existing in ("", "UNKNOWN") and incoming not in ("", "UNKNOWN"):
                                targets[gu]["action_type"] = incoming

                elif coll == "FDA_Drug_Labels":
                    targets.setdefault("_FDA_MOA_", {})["fda_narrative"] = p.get("text_content", "")[:1000]

                elif coll == "Drug_agent":
                    gene = p.get("gene_symbol", "")
                    if gene:
                        gu = gene.upper()
                        if gu not in targets:
                            targets[gu] = {"gene_symbol": gene, "action_type": "UNKNOWN",
                                           "mechanism": p.get("mechanism_of_action", "")}

                elif coll == "Raw_csv_KG":
                    gene = self._kg_extract_gene(p)
                    if gene:
                        gu = gene.upper()
                        if gu not in targets:
                            targets[gu] = {"gene_symbol": gene, "action_type": "UNKNOWN",
                                           "mechanism": p.get("display_relation", p.get("relation", ""))}

                elif coll == "PDR_Drugs_Data":
                    gene = p.get("gene_symbol", "")
                    if gene:
                        gu = gene.upper()
                        if gu not in targets:
                            targets[gu] = {"gene_symbol": gene, "action_type": "UNKNOWN",
                                           "mechanism": p.get("mechanism_of_action", "")[:200]}

        # Attach FDA MoA narrative to all targets
        fda_narrative = targets.pop("_FDA_MOA_", {}).get("fda_narrative", "")
        result = []
        for data in targets.values():
            data["fda_narrative"] = fda_narrative
            result.append(data)
        return result

    def get_target_disease_score(self, gene_symbol: str, disease_name: str) -> Optional[float]:
        """OpenTargets association score + KG supporting evidence.

        Uses Qdrant payload filters on target_name/disease_name for OT data
        (vector search fails because ~20K association records share identical
        template text).  Raw_csv_KG still uses semantic search.
        """
        disease_name = re.sub(r"\s*\(.*?\)", "", disease_name).strip()
        cache_key = f"ot_score:{gene_symbol.upper()}:{disease_name.upper()}"
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        # Split compound disease names so each component is searched independently
        disease_components = [d.strip() for d in re.split(r"\s*/\s*|\s*,\s*|\s*\+\s*|\s+and\s+", disease_name) if d.strip()]
        if not disease_components:
            disease_components = [disease_name]

        best_score = None
        gene_upper = gene_symbol.upper()

        for component in disease_components:
            # ── OT: payload-filtered scroll (deterministic, no vector search) ──
            if "OpenTargets_data" in self._available:
                try:
                    ot_filter = Filter(must=[
                        FieldCondition(key="entity_type", match=MatchValue(value="association")),
                        FieldCondition(key="target_name", match=MatchValue(value=gene_upper)),
                    ])
                    self._queried_collections.add("OpenTargets_data")
                    self._qdrant_sem.acquire()
                    try:
                        points, _ = self.client.scroll(
                            "OpenTargets_data", scroll_filter=ot_filter,
                            limit=50, with_payload=True,
                        )
                    finally:
                        self._qdrant_sem.release()
                    comp_lower = component.lower()
                    for pt in points:
                        p = pt.payload
                        dn = (p.get("disease_name") or "").lower()
                        if not _disease_matches(comp_lower, dn):
                            continue
                        # Extract score from probed fields
                        s = None
                        for sf in (self._ot_score_fields or ["score", "overall_score", "overall_association_score"]):
                            s = p.get(sf)
                            if s is not None:
                                break
                        if s is not None:
                            try:
                                s = float(s)
                                best_score = max(best_score or 0, s)
                            except (ValueError, TypeError):
                                pass
                except Exception as e:
                    logger.warning(f"OT payload-filter scroll failed for {gene_upper}+{component}: {e}")

            # ── Raw_csv_KG: semantic search (heterogeneous text, works well) ──
            if "Raw_csv_KG" in self._available:
                kg_queries = [("Raw_csv_KG", f"{gene_symbol} {component} associated", 3, None)]
                raw = self._parallel_search(kg_queries)
                for key, hits in raw.items():
                    for h in hits:
                        p = h["payload"]
                        s = None
                        for sf in (self._ot_score_fields or ["score", "overall_score", "overall_association_score"]):
                            s = p.get(sf)
                            if s is not None:
                                break
                        if s is not None:
                            try:
                                s = float(s)
                                best_score = max(best_score or 0, s)
                            except (ValueError, TypeError):
                                pass

        self._embed_cache[cache_key] = best_score
        return best_score

    def get_indication_status(self, drug_name: str, disease_name: str,
                               disease_aliases: Optional[List[str]] = None) -> Dict:
        """Check if drug is approved for disease — FDA Labels + OT + Orange Book."""
        queries = [
            ("FDA_Drug_Labels", f"{drug_name} {disease_name}", 3,
             Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="indications_and_usage"))])),
            ("OpenTargets_drugs_enriched", f"{drug_name} {disease_name}", 5, None),
            ("OpenTargets_drugs_enriched", f"{drug_name} approved indications", 5, None),
            ("FDA_Orange_Book", drug_name, 3, None),
        ]
        raw = self._parallel_search(queries)
        result = {"is_approved": False, "highest_phase": None, "indication_text": "", "approval_date": None,
                  "approved_indications": []}
        alias_set = {self._ascii_fold(a).lower() for a in ([disease_name] + (disease_aliases or []))}
        _all_indications: Dict[str, Dict] = {}  # keyed by lowercased disease_name for dedup

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                if not self._fuzzy_drug_match(drug_name, p, coll):
                    continue

                if coll == "FDA_Drug_Labels":
                    text = p.get("text_content", "")
                    text_lower = self._ascii_fold(text).lower()
                    if any(_disease_matches(alias, text_lower) for alias in alias_set):
                        result["is_approved"] = True
                        result["indication_text"] = text[:500]

                elif coll == "OpenTargets_drugs_enriched":
                    for ind in p.get("indications", []):
                        ind_name = self._ascii_fold(ind.get("disease_name", "")).lower()
                        phase = ind.get("phase") or ind.get("clinical_phase")
                        phase_int = None
                        if phase is not None:
                            try:
                                phase_int = int(phase)
                            except (ValueError, TypeError):
                                pass
                        # Collect ALL indications for multi-disease display
                        if ind_name and ind_name not in _all_indications:
                            _all_indications[ind_name] = {
                                "disease_name": ind.get("disease_name", ""),
                                "phase": phase_int,
                            }
                        elif ind_name and phase_int is not None:
                            prev = _all_indications[ind_name].get("phase") or 0
                            if phase_int > prev:
                                _all_indications[ind_name]["phase"] = phase_int
                        # Disease-match check for the queried disease
                        if any(_disease_matches(alias, ind_name) or _disease_matches(ind_name, alias) for alias in alias_set):
                            result["is_approved"] = True
                            if not result["indication_text"]:
                                result["indication_text"] = ind.get("disease_name", "")
                            if phase_int is not None:
                                result["highest_phase"] = max(result["highest_phase"] or 0, phase_int)

                elif coll == "FDA_Orange_Book":
                    if p.get("approval_date"):
                        result["approval_date"] = p["approval_date"]

        # If FDA label confirmed approval but OT indications don't include the
        # queried disease, inject it so the UI shows the correct indication.
        if result["is_approved"]:
            disease_key = self._ascii_fold(disease_name).lower()
            if disease_key not in _all_indications:
                _all_indications[disease_key] = {
                    "disease_name": disease_name,
                    "phase": 4,  # FDA-approved
                }
            else:
                prev = _all_indications[disease_key].get("phase") or 0
                if prev < 4:
                    _all_indications[disease_key]["phase"] = 4
            result["highest_phase"] = max(result["highest_phase"] or 0, 4)

        # Sort indications by clinical phase descending (4→3→2→1→None), then cap at 20
        result["approved_indications"] = sorted(
            _all_indications.values(),
            key=lambda x: x.get("phase") or -1, reverse=True,
        )[:20]
        return result

    def get_trial_evidence(self, drug_name: str, disease_name: str,
                            disease_aliases: Optional[List[str]] = None) -> Dict:
        """Clinical trial data — summaries + results."""
        import numpy as np
        queries = [
            ("ClinicalTrials_results", f"{drug_name} {disease_name}", 15, None),
            ("ClinicalTrials_summaries", f"{drug_name} {disease_name}", 15, None),
        ]
        raw = self._parallel_search(queries)
        drug_upper = drug_name.upper()
        _GENERIC = {"cancer", "disease", "syndrome", "chronic", "acute", "primary", "advanced", "metastatic", "stage", "type", "with", "cell"}
        all_names = [disease_name] + (disease_aliases or [])
        disease_keywords = list({self._ascii_fold(w).lower() for name in all_names
                                 for w in name.split() if len(w) >= 4 and w.lower() not in _GENERIC})
        disease_vec = None
        evidence = {
            "total_trials": 0, "highest_phase": None, "completed_trials": 0,
            "trials_with_results": 0, "best_p_value": None, "total_enrollment": 0,
            "top_trials": [],
        }
        seen_ncts = set()

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                trial_text = (p.get("brief_title", "") + " " + p.get("text_content", "")[:500]).upper()

                # Verify the trial actually involves this drug
                trial_drugs = [d.upper() for d in p.get("drug_names", [])]
                if trial_drugs:
                    if not any(drug_upper in d or d in drug_upper for d in trial_drugs):
                        continue
                else:
                    if drug_upper not in trial_text:
                        continue

                # Verify trial is relevant to the target disease
                # Pass 1: keyword match
                disease_relevant = (
                    not disease_keywords
                    or any(kw.upper() in trial_text for kw in disease_keywords)
                )
                # Pass 2: semantic fallback when keywords miss synonyms
                #         (e.g. "myelogenous" vs "myeloid")
                if not disease_relevant and self.embedder:
                    try:
                        if disease_vec is None:
                            disease_vec = np.array(self._embed(disease_name), dtype=np.float32)
                        title = p.get("brief_title", "")[:200]
                        title_vec = np.array(self._embed(title), dtype=np.float32)
                        sim = float(np.dot(
                            disease_vec / (np.linalg.norm(disease_vec) + 1e-10),
                            title_vec / (np.linalg.norm(title_vec) + 1e-10),
                        ))
                        if sim >= 0.6:
                            disease_relevant = True
                    except Exception:
                        pass
                if not disease_relevant:
                    continue

                nct = p.get("nct_id", "")
                if nct in seen_ncts:
                    continue
                seen_ncts.add(nct)

                evidence["total_trials"] += 1
                phase_num = p.get("phase_numeric")
                if phase_num is not None:
                    try:
                        evidence["highest_phase"] = max(evidence["highest_phase"] or 0, float(phase_num))
                    except (ValueError, TypeError):
                        pass

                status = p.get("overall_status", "").upper()
                if "COMPLETED" in status:
                    evidence["completed_trials"] += 1

                enrollment = p.get("enrollment")
                if enrollment:
                    try:
                        evidence["total_enrollment"] += int(enrollment)
                    except (ValueError, TypeError):
                        pass

                if coll == "ClinicalTrials_results":
                    evidence["trials_with_results"] += 1
                    for pv in p.get("p_values", []):
                        try:
                            pv_f = float(pv)
                            if evidence["best_p_value"] is None or pv_f < evidence["best_p_value"]:
                                evidence["best_p_value"] = pv_f
                        except (ValueError, TypeError):
                            pass

                if len(evidence["top_trials"]) < 5:
                    why = p.get("why_stopped", "") or ""
                    evidence["top_trials"].append({
                        "nct_id": nct, "title": p.get("brief_title", ""),
                        "phase": p.get("phase", ""), "status": p.get("overall_status", ""),
                        "has_results": coll == "ClinicalTrials_results",
                        "why_stopped": why,
                    })

        _SAFETY_TERMS = {"safety", "toxicity", "adverse", "death", "fatal", "harmful"}
        evidence["stopped_for_safety"] = any(
            any(term in t.get("why_stopped", "").lower() for term in _SAFETY_TERMS)
            for t in evidence["top_trials"]
        )
        return evidence

    def get_safety_profile(self, drug_name: str) -> Dict:
        """Aggregate safety — FDA Labels (3 sections batched) + OT AEs + FAERS + PGx + Enforcement."""
        # Pre-embed drug_name once for the 3 FDA_Drug_Labels batch queries
        drug_vec = self._embed(drug_name)

        # Batch the 3 FDA_Drug_Labels section queries into 1 round-trip
        fda_label_sections = [
            ("boxed_warning", 3),
            ("adverse_reactions", 5),
            ("contraindications", 3),
        ]
        fda_batch_results: Dict[str, list] = {}
        if "FDA_Drug_Labels" in self._available:
            cb = self._breakers.get("FDA_Drug_Labels")
            if not (cb and cb.current_state == "open"):
                try:
                    batch_requests = [
                        QueryRequest(
                            query=drug_vec,
                            filter=Filter(must=[FieldCondition(key="section_name", match=MatchValue(value=section))]),
                            limit=limit,
                            with_payload=True,
                            params=SearchParams(hnsw_ef=128),
                        )
                        for section, limit in fda_label_sections
                    ]
                    t0 = time.perf_counter()
                    batch_response = self.client.query_batch_points(
                        "FDA_Drug_Labels", batch_requests,
                        timeout=self._get_collection_timeout("FDA_Drug_Labels"),
                    )
                    elapsed = time.perf_counter() - t0
                    with self._timing_lock:
                        self._search_count += len(fda_label_sections)
                        self._collection_timings.setdefault("FDA_Drug_Labels", []).extend(
                            [elapsed / len(fda_label_sections)] * len(fda_label_sections))
                    self._queried_collections.add("FDA_Drug_Labels")
                    for i, resp in enumerate(batch_response):
                        section = fda_label_sections[i][0]
                        fda_batch_results[section] = [{"score": r.score, "payload": r.payload} for r in resp.points]
                except Exception as e:
                    logger.warning(f"FDA_Drug_Labels batch failed, falling back to individual: {e}")

        # Remaining non-FDA-label queries run in parallel as before
        queries = [
            ("OpenTargets_adverse_events", drug_name, 10, None),
            ("FDA_FAERS", drug_name, 5, None),
            ("OpenTargets_pharmacogenomics", drug_name, 5, None),
            ("FDA_Enforcement", drug_name, 5, None),
            ("PDR_Drugs_Data", f"{drug_name} adverse reactions safety", 3,
             Filter(must=[FieldCondition(key="drug_name_lower", match=MatchValue(value=drug_name.lower()))])),
        ]
        # If batch failed, fall back to individual FDA searches
        if not fda_batch_results:
            queries = [
                ("FDA_Drug_Labels", drug_name, 3,
                 Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="boxed_warning"))])),
                ("FDA_Drug_Labels", drug_name, 5,
                 Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="adverse_reactions"))])),
                ("FDA_Drug_Labels", drug_name, 3,
                 Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="contraindications"))])),
            ] + queries
        raw = self._parallel_search(queries)

        # Merge FDA batch results into raw under the same key format
        for section, hits in fda_batch_results.items():
            raw[f"FDA_Drug_Labels|{drug_name[:40]}_{section}"] = hits

        profile: Dict = {
            "boxed_warnings": [], "top_adverse_events": [],
            "serious_ratio": None, "fatal_ratio": None,
            "contraindications": [], "pgx_warnings": [], "recall_history": [],
        }

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                if not self._fuzzy_drug_match(drug_name, p, coll):
                    continue

                if coll == "FDA_Drug_Labels":
                    section = p.get("section_name", "")
                    text = p.get("text_content", "")[:500]
                    if section == "boxed_warning" and text:
                        profile["boxed_warnings"].append(text)
                    elif section == "adverse_reactions" and text:
                        profile["top_adverse_events"].append({"text": text, "source": "FDA_Label"})
                    elif section == "contraindications" and text:
                        profile["contraindications"].append(text)

                elif coll == "OpenTargets_adverse_events":
                    profile["top_adverse_events"].append({
                        "event_name": p.get("event_name", ""),
                        "log_lr": p.get("log_lr"), "count": p.get("report_count"),
                        "source": "OpenTargets",
                    })

                elif coll == "FDA_FAERS":
                    if p.get("entity_type") == "faers_summary":
                        try:
                            profile["serious_ratio"] = float(p["serious_pct"]) / 100 if p.get("serious_pct") else profile["serious_ratio"]
                            profile["fatal_ratio"] = float(p["fatal_pct"]) / 100 if p.get("fatal_pct") else profile["fatal_ratio"]
                        except (ValueError, TypeError):
                            pass
                    else:
                        profile["top_adverse_events"].append({
                            "event_name": p.get("reaction_term", ""),
                            "count": p.get("reaction_count"), "source": "FAERS",
                        })

                elif coll == "OpenTargets_pharmacogenomics":
                    profile["pgx_warnings"].append({
                        "gene": p.get("gene_symbol", ""),
                        "variant": p.get("variant_rs_id", ""),
                        "phenotype": p.get("phenotype_text", ""),
                        "category": p.get("pgx_category", ""),
                        "evidence_level": p.get("evidence_level", ""),
                    })

                elif coll == "FDA_Enforcement":
                    profile["recall_history"].append({
                        "classification": p.get("classification", ""),
                        "reason": p.get("reason_for_recall", "")[:200],
                        "status": p.get("status", ""),
                        "date": p.get("recall_initiation_date", ""),
                    })

                elif coll == "PDR_Drugs_Data":
                    for reaction in p.get("severe_reactions", []):
                        profile["top_adverse_events"].append({
                            "event_name": reaction, "source": "PDR",
                        })
                    di = p.get("drug_interactions", "")
                    if di:
                        profile["pdr_drug_interactions"] = di

        # Sort AEs by log_lr where available
        profile["top_adverse_events"].sort(
            key=lambda x: x.get("log_lr") or 0, reverse=True)
        return profile

    def _get_disease_ae_synonyms(self, disease_name: str) -> set:
        """Dynamically discover MedDRA-style synonyms for a disease by querying the AE collection."""
        cache_key = f"ae_syn:{disease_name}"
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        import numpy as np
        disease_clean = re.sub(r"\s*\(.*?\)", "", disease_name).strip()
        synonyms = {disease_clean.lower()}

        hits = self._search("OpenTargets_adverse_events", disease_clean, 15)
        if hits and self.embedder:
            d_vec = np.array(self._embed(disease_clean), dtype=np.float32)
            d_norm = d_vec / (np.linalg.norm(d_vec) + 1e-10)
            for h in hits:
                term = h["payload"].get("event_name", "")
                if not term or len(term) < 4:
                    continue
                t_vec = np.array(self._embed(term), dtype=np.float32)
                sim = float(np.dot(d_norm, t_vec / (np.linalg.norm(t_vec) + 1e-10)))
                if sim >= 0.55:
                    synonyms.add(term.lower())

        self._embed_cache[cache_key] = synonyms
        return synonyms

    def get_disease_aliases(self, disease_name: str) -> List[str]:
        """Dynamically discover disease synonyms from OT disease entities + drug indication names."""
        import numpy as np
        cache_key = f"disease_aliases:{disease_name}"
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        d_vec = np.array(self._embed(disease_name), dtype=np.float32)
        d_norm = d_vec / (np.linalg.norm(d_vec) + 1e-10)
        candidate_names: set = set()

        # Source 1: OT disease entities
        filt = Filter(must=[FieldCondition(key="entity_type", match=MatchValue(value="disease"))])
        for h in self._search("OpenTargets_data", disease_name, 30, filt):
            name = h["payload"].get("name", "")
            if name and len(name) >= 4:
                candidate_names.add(name)

        # Source 2: indication disease_names from drugs semantically near this disease
        for h in self._search("OpenTargets_drugs_enriched", disease_name, 20):
            for ind in h["payload"].get("indications", []):
                name = ind.get("disease_name", "")
                if name and len(name) >= 4:
                    candidate_names.add(name)

        disease_lower = disease_name.lower()
        _STOPWORDS = {"syndrome", "disease", "disorder", "condition", "primary", "secondary", "chronic", "acute", "of", "the", "and", "in", "with"}
        disease_content = {self._ascii_fold(w).lower().rstrip("'s") for w in disease_name.split()
                           if len(w) >= 3 and w.lower().rstrip("'s") not in _STOPWORDS}
        scored = []
        for name in candidate_names:
            if name.lower() == disease_lower:
                continue
            t_vec = np.array(self._embed(name), dtype=np.float32)
            sim = float(np.dot(d_norm, t_vec / (np.linalg.norm(t_vec) + 1e-10)))
            if sim <= 0.60:
                continue
            # Reject aliases that share zero content words with the original disease
            alias_content = {self._ascii_fold(w).lower().rstrip("'s") for w in name.split()
                             if len(w) >= 3 and w.lower().rstrip("'s") not in _STOPWORDS}
            if disease_content and alias_content and not disease_content & alias_content:
                continue
            scored.append((sim, name))

        scored.sort(reverse=True)
        aliases = [disease_name] + [name for _, name in scored[:12]]

        self._embed_cache[cache_key] = aliases
        logger.info(f"Disease aliases for '{disease_name}': {aliases}")
        print(f"        Disease aliases for '{disease_name}': {aliases}")
        return aliases

    def check_disease_in_adverse_events(self, safety: Dict, disease_name: str) -> Dict:
        """Cross-reference a drug's adverse events against the patient's disease.
        Catches drugs that CAUSE the condition being treated."""
        import numpy as np
        disease_clean = re.sub(r"\s*\(.*?\)", "", disease_name).strip()
        disease_synonyms = self._get_disease_ae_synonyms(disease_name)
        disease_tokens_all = {w for syn in disease_synonyms for w in syn.split() if len(w) >= 4}
        matching = []

        for ae in safety.get("top_adverse_events", []):
            event = ae.get("event_name", "").lower()
            if not event:
                continue
            # Direct match against any synonym
            if any(syn in event or event in syn for syn in disease_synonyms):
                matching.append(ae.get("event_name", event))
                continue
            # Token overlap across all synonyms
            event_tokens = {w for w in event.split() if len(w) >= 4}
            if disease_tokens_all and event_tokens and len(disease_tokens_all & event_tokens) >= max(1, len(disease_tokens_all) // 2):
                matching.append(ae.get("event_name", event))
                continue
            # Semantic fallback
            if self.embedder:
                try:
                    d_vec = np.array(self._embed(disease_clean), dtype=np.float32)
                    e_vec = np.array(self._embed(event), dtype=np.float32)
                    sim = float(np.dot(
                        d_vec / (np.linalg.norm(d_vec) + 1e-10),
                        e_vec / (np.linalg.norm(e_vec) + 1e-10),
                    ))
                    if sim >= 0.50:
                        matching.append(f"{ae.get('event_name', event)} (semantic={sim:.2f})")
                except Exception:
                    pass

        # Scan FDA label text (boxed warnings + contraindications)
        for field_key in ("boxed_warnings", "contraindications"):
            for text in safety.get(field_key, []):
                text_lower = text.lower()
                if any(syn in text_lower for syn in disease_synonyms):
                    matching.append(f"FDA label {field_key}: mentions {disease_clean}")

        if matching:
            return {
                "is_contraindicated": True,
                "reason": f"Drug's adverse events include the patient's disease ({', '.join(matching[:3])})",
                "matching_events": matching,
            }
        return {"is_contraindicated": False, "reason": "", "matching_events": []}

    def check_contraindication(self, drug_name: str, gene_symbol: str,
                                gene_direction: str, log2fc: float = 0.0) -> Dict:
        """Check if drug's action on target conflicts with patient gene expression.
        Returns tier: 2 (Contraindicated) when |log2fc| >= clinical threshold,
                      3 (Use With Caution) when below threshold.
        """
        from agentic_ai_wf.reporting_pipeline_agent.core_types import DEG_LOG2FC_THRESHOLD
        results = self._search("ChEMBL_drugs", f"{drug_name} {gene_symbol}", 5)
        gene_up = gene_symbol.upper()

        for h in results:
            p = h["payload"]
            if not self._fuzzy_drug_match(drug_name, p, "ChEMBL_drugs"):
                continue
            targets = [t.upper() for t in p.get("target_gene_symbols", [])]
            actions = p.get("action_types", [])
            if gene_up not in targets:
                continue

            idx = targets.index(gene_up)
            action = actions[idx].upper() if idx < len(actions) else ""
            tier = 2 if abs(log2fc) >= DEG_LOG2FC_THRESHOLD else 3

            if gene_direction == "down" and action in ("INHIBITOR", "NEGATIVE MODULATOR", "ANTAGONIST", "BLOCKER"):
                return {"is_contraindicated": True,
                        "reason": f"{drug_name} is a {action} of {gene_symbol} which is already downregulated (inhibiting a suppressed target)",
                        "severity": "high", "tier": tier}
            if gene_direction == "up" and action in ("AGONIST", "POSITIVE MODULATOR", "ACTIVATOR"):
                return {"is_contraindicated": True,
                        "reason": f"{drug_name} is a {action} of {gene_symbol} which is upregulated (may reinforce oncogenic driver)",
                        "severity": "moderate", "tier": tier}

        return {"is_contraindicated": False, "reason": "", "severity": "", "tier": 0}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract_drug_names(self, payload: Dict, collection: str) -> List[str]:
        """Dynamically extract drug names from any collection's payload."""
        names = []
        # Direct drug name fields
        for field in ("drug_name", "name", "molecule_name", "generic_name",
                       "ingredient", "brand_name", "trade_name"):
            val = payload.get(field, "")
            if val and isinstance(val, str) and len(val) > 1:
                names.append(val)

        # List fields containing drug names
        for field in ("drug_names", "synonyms"):
            vals = payload.get(field, [])
            if isinstance(vals, list):
                names.extend([v for v in vals if isinstance(v, str) and len(v) > 1])

        # Nested drug lists in Drug_agent disease_drug docs
        for entry in payload.get("approved_drugs", []):
            if isinstance(entry, dict) and entry.get("drug_name"):
                names.append(entry["drug_name"])

        # Nested drug lists in Drug_agent pathway_drug docs
        for entry in payload.get("targeting_drugs", []):
            if isinstance(entry, dict) and entry.get("drug_name"):
                names.append(entry["drug_name"])

        # KG triples — extract the Drug entity
        if collection == "Raw_csv_KG":
            for side in ("x", "y"):
                if payload.get(f"{side}_type", "").lower() in ("drug", "compound"):
                    n = payload.get(f"{side}_name", "")
                    if n:
                        names.append(n)

        return list(dict.fromkeys(names))  # Deduplicate preserving order

    def _extract_action_type(self, payload: Dict, collection: str) -> str:
        if collection == "ChEMBL_drugs":
            actions = payload.get("action_types", [])
            return actions[0] if actions else "UNKNOWN"
        if collection == "DrugPath_KEGG":
            ed = payload.get("effect_direction", 0)
            return "INHIBITOR" if ed == -1 else "ACTIVATOR" if ed == 1 else "UNKNOWN"
        if collection == "PDR_Drugs_Data":
            return "UNKNOWN"
        return payload.get("action_type", "UNKNOWN")

    def _fuzzy_drug_match(self, query_drug: str, payload: Dict, collection: str) -> bool:
        """Check if payload plausibly refers to the queried drug."""
        q = query_drug.upper().strip()
        candidates = self._extract_drug_names(payload, collection)
        return any(q in c.upper() or c.upper() in q for c in candidates) if candidates else True

    def _kg_involves_entity(self, payload: Dict, entity_name: str, entity_types: set) -> bool:
        """Check if a KG triple involves a specific entity type."""
        for side in ("x", "y"):
            etype = payload.get(f"{side}_type", "").lower()
            if etype in entity_types:
                if not entity_name:
                    return True
                if entity_name.upper() in payload.get(f"{side}_name", "").upper():
                    return True
        return False

    def _kg_extract_gene(self, payload: Dict) -> Optional[str]:
        for side in ("x", "y"):
            if payload.get(f"{side}_type", "").lower() in ("gene", "protein"):
                return payload.get(f"{side}_name")
        return None

    def shutdown(self):
        self._executor.shutdown(wait=False)
