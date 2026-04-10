"""
PubMedBERT Embedder Module - Generates embeddings for biomedical text.
"""

import logging
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    embeddings: List[List[float]]
    texts: List[str]
    dimension: int
    model_name: str
    cached_count: int = 0


class PubMedBERTEmbedder:
    """
    Embedder using PubMedBERT for biomedical text.
    Supports caching and batch processing.
    """
    
    DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"
    EMBEDDING_DIMENSION = 768
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "auto",
        cache_enabled: bool = True,
        cache_directory: str = "./embedding_cache",
        batch_size: int = 32,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = self._resolve_device(device)
        self.cache_enabled = cache_enabled
        self.cache_directory = Path(cache_directory) if cache_enabled else None
        self.batch_size = batch_size
        
        self._model = None
        self._cache: Dict[str, List[float]] = {}
        
        if self.cache_enabled and self.cache_directory:
            self.cache_directory.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading model {self.model_name} on {self.device}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self):
        cache_file = self.cache_directory / "embeddings_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        if not self.cache_enabled or not self.cache_directory:
            return
        cache_file = self.cache_directory / "embeddings_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        if not text:
            return [0.0] * self.EMBEDDING_DIMENSION
        
        if self.cache_enabled:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        
        if self.cache_enabled:
            self._cache[cache_key] = embedding
        
        return embedding
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> EmbeddingResult:
        """Embed multiple texts with batching and caching."""
        if not texts:
            return EmbeddingResult([], [], self.EMBEDDING_DIMENSION, self.model_name)
        
        embeddings = []
        texts_to_embed = []
        text_indices = []
        cached_count = 0
        
        for i, text in enumerate(texts):
            if not text:
                embeddings.append([0.0] * self.EMBEDDING_DIMENSION)
                continue
            
            if self.cache_enabled:
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    embeddings.append(self._cache[cache_key])
                    cached_count += 1
                    continue
            
            embeddings.append(None)
            texts_to_embed.append(text)
            text_indices.append(i)
        
        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} texts (cached: {cached_count})")
            
            new_embeddings = self.model.encode(
                texts_to_embed,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            
            for idx, text, embedding in zip(text_indices, texts_to_embed, new_embeddings):
                embedding_list = embedding.tolist()
                embeddings[idx] = embedding_list
                if self.cache_enabled:
                    self._cache[self._get_cache_key(text)] = embedding_list
            
            if self.cache_enabled:
                self._save_cache()
        
        return EmbeddingResult(embeddings, texts, self.EMBEDDING_DIMENSION, self.model_name, cached_count)
    
    def get_dimension(self) -> int:
        return self.EMBEDDING_DIMENSION
    
    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_enabled": self.cache_enabled,
            "cached_embeddings": len(self._cache),
        }
    
    def clear_cache(self):
        self._cache = {}
        if self.cache_enabled and self.cache_directory:
            cache_file = self.cache_directory / "embeddings_cache.json"
            if cache_file.exists():
                cache_file.unlink()
        logger.info("Embedding cache cleared")
