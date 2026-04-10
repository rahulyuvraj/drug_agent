"""
Basic Auth Qdrant Client Wrapper
================================

Provides a QdrantClient that works with HTTP Basic Auth (e.g., nginx proxy).
"""

import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def create_qdrant_client_with_basic_auth(
    url: str,
    username: str,
    password: str,
    timeout: int = 60,
    port: int = 443,
):
    """
    Create a QdrantClient that works with HTTP Basic Auth.
    
    Args:
        url: Qdrant server URL (e.g., https://vector.f420.ai)
        username: HTTP Basic Auth username
        password: HTTP Basic Auth password
        timeout: Request timeout in seconds
        port: Server port (443 for nginx HTTPS proxy, 6333 for direct)
    
    Returns:
        QdrantClient instance configured with Basic Auth
    """
    from qdrant_client import QdrantClient
    import httpx
    
    # Create client with explicit port (443 for nginx proxy)
    client = QdrantClient(
        url=url, 
        port=port, 
        timeout=timeout, 
        prefer_grpc=False,
        check_compatibility=False
    )
    
    # Create httpx client with Basic Auth
    custom_http_client = httpx.Client(
        auth=httpx.BasicAuth(username, password),
        timeout=float(timeout),
    )
    
    # Patch ALL api endpoints with the new client
    http_apis = client._client.http
    api_names = [
        'collections_api', 'points_api', 'service_api', 
        'snapshots_api', 'indexes_api', 'search_api',
        'aliases_api', 'distributed_api', 'beta_api'
    ]
    
    for api_name in api_names:
        api = getattr(http_apis, api_name, None)
        if api and hasattr(api, 'api_client'):
            api.api_client._client = custom_http_client
    
    logger.info(f"Created QdrantClient with Basic Auth for {url}:{port}")
    return client


def get_qdrant_client_from_env(env_path: Optional[str] = None):
    """
    Create QdrantClient from environment variables.
    
    Supports:
    - QDRANT_URL: Server URL (required)
    - QDRANT_PORT: Server port (default: 443 for HTTPS, 6333 for HTTP)
    - QDRANT_USERNAME: HTTP Basic Auth username (optional)
    - QDRANT_PASSWORD: HTTP Basic Auth password (optional)  
    - QDRANT_API_KEY: Qdrant API key for cloud (optional)
    - QDRANT_COLLECTION: Collection name (default: drug_knowledge_base)
    
    Returns:
        tuple: (client, collection_name, error_message)
    """
    from dotenv import load_dotenv
    
    # Load from .env file
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()
    
    url = os.getenv("QDRANT_URL")
    username = os.getenv("QDRANT_USERNAME")
    password = os.getenv("QDRANT_PASSWORD")
    api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "drug_knowledge_base")
    
    # Determine port: 443 for HTTPS (nginx), 6333 for direct Qdrant
    default_port = 443 if url and url.startswith("https://") else 6333
    port = int(os.getenv("QDRANT_PORT", str(default_port)))
    
    if not url:
        return None, None, "Missing QDRANT_URL in environment"
    
    try:
        from qdrant_client import QdrantClient
        
        if username and password:
            # Use Basic Auth for self-hosted with nginx proxy
            client = create_qdrant_client_with_basic_auth(
                url=url,
                username=username,
                password=password,
                timeout=60,
                port=port
            )
            logger.info(f"Connected to Qdrant with Basic Auth: {url}:{port}")
        elif api_key:
            # Use API key for Qdrant Cloud
            client = QdrantClient(url=url, api_key=api_key, timeout=60, prefer_grpc=False)
            logger.info(f"Connected to Qdrant Cloud: {url}")
        else:
            # No auth
            client = QdrantClient(url=url, port=port, timeout=60, prefer_grpc=False)
            logger.info(f"Connected to Qdrant (no auth): {url}:{port}")
        
        return client, collection_name, None
        
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return None, None, str(e)
