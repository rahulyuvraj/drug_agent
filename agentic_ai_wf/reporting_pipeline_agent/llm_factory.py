#!/usr/bin/env python3
"""
=============================================================================
REPORTING PIPELINE AGENT - LLM Factory Module
=============================================================================
Unified LLM interface with AWS Bedrock (Claude) and OpenAI support.
Provides SmartLLMRouter for tiered task routing and ClaudeValidator for
gene/drug validation against disease.

Configuration via environment variables:
    AWS_ACCESS_KEY_ID       - AWS access key for Bedrock
    AWS_SECRET_ACCESS_KEY   - AWS secret key for Bedrock
    AWS_REGION              - AWS region (default: us-east-1)
    BEDROCK_MODEL_ID        - Claude model ID
    USE_BEDROCK             - Set to "true" to enable Bedrock
    OPENAI_API_KEY          - OpenAI API key (fallback)
=============================================================================
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================
try:
    import boto3
    from botocore.config import Config as BotoConfig
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


# =============================================================================
# AWS BEDROCK CLIENT WRAPPER (Claude)
# =============================================================================
class BedrockLLMClient:
    """
    AWS Bedrock client wrapper with OpenAI-compatible interface.
    Allows seamless switching between OpenAI and AWS Bedrock Claude models.
    """
    
    class ChatCompletions:
        """Mimics OpenAI's chat.completions interface."""
        
        def __init__(self, bedrock_client, model_id: str):
            self.bedrock_client = bedrock_client
            self.model_id = model_id
        
        def create(self, model: str = None, messages: list = None, 
                   temperature: float = 0.0, max_tokens: int = 4096, **kwargs):
            """Create a chat completion using AWS Bedrock Claude."""
            # Convert OpenAI message format to Claude format
            system_prompt = ""
            conversation = []
            
            for msg in messages or []:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    conversation.append({"role": "user", "content": content})
                elif role == "assistant":
                    conversation.append({"role": "assistant", "content": content})
            
            # Build Claude request body
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": conversation
            }
            
            if system_prompt:
                request_body["system"] = system_prompt
            
            # Call Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response["body"].read())
            
            # Convert to OpenAI-compatible response format
            return BedrockResponse(
                content=response_body.get("content", [{}])[0].get("text", ""),
                model=self.model_id,
                usage=response_body.get("usage", {})
            )
    
    class Chat:
        """Mimics OpenAI's chat interface."""
        def __init__(self, completions):
            self.completions = completions
    
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, 
                 region_name: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        """Initialize Bedrock client with AWS credentials."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for AWS Bedrock. Run: pip install boto3")
        
        boto_config = BotoConfig(
            read_timeout=300,
            connect_timeout=60,
            retries={"max_attempts": 3}
        )
        
        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            config=boto_config
        )
        self.model_id = model_id
        self.model_name = model_id
        
        # Create OpenAI-compatible interface
        completions = self.ChatCompletions(self.bedrock_client, model_id)
        self.chat = self.Chat(completions)


class BedrockResponse:
    """OpenAI-compatible response object for Bedrock."""
    def __init__(self, content: str, model: str, usage: dict):
        self.choices = [BedrockChoice(content)]
        self.model = model
        self.usage = usage


class BedrockChoice:
    """OpenAI-compatible choice object."""
    def __init__(self, content: str):
        self.message = BedrockMessage(content)


class BedrockMessage:
    """OpenAI-compatible message object."""
    def __init__(self, content: str):
        self.content = content


# =============================================================================
# SMART LLM ROUTER
# =============================================================================
class SmartLLMRouter:
    """
    Smart LLM Router that uses Claude for complex/critical tasks and OpenAI for simple tasks.
    Provides automatic fallback if primary LLM fails.
    
    Task Types:
    - "simple": Column mapping, JSON parsing → OpenAI (fast, cheap)
    - "complex": Disease knowledge, TME classification → Claude (accurate)
    - "critical": Report sections, clinical validation → Claude (most important)
    """
    
    def __init__(self, claude_client=None, openai_client=None):
        self.claude_client = claude_client
        self.openai_client = openai_client
        self._task_routing = {
            'simple': 'openai',
            'complex': 'claude', 
            'critical': 'claude'
        }
    
    @property
    def model_name(self):
        """Resolve active model name from whichever client is primary."""
        for client in (self.claude_client, self.openai_client):
            if client and hasattr(client, 'model_name'):
                return client.model_name
        return "gpt-4"
    
    @property
    def chat(self):
        """Provide OpenAI-compatible interface - routes to Claude by default."""
        return self
    
    @property
    def completions(self):
        """Provide OpenAI-compatible interface."""
        return self
    
    def create(self, model: str = None, messages: list = None, 
               temperature: float = 0.0, max_tokens: int = 4096, 
               task_type: str = "complex", **kwargs):
        """Smart routing based on task_type with fallback."""
        primary_client = self._get_primary_client(task_type)
        fallback_client = self._get_fallback_client(task_type)
        
        # Try primary client
        if primary_client:
            try:
                return primary_client.chat.completions.create(
                    model=model, messages=messages, 
                    temperature=temperature, max_tokens=max_tokens, **kwargs
                )
            except Exception as e:
                logger.warning(f"Primary LLM ({self._get_client_name(primary_client)}) failed: {e}")
                if fallback_client:
                    logger.info(f"Falling back to {self._get_client_name(fallback_client)}")
        
        # Try fallback client
        if fallback_client:
            try:
                return fallback_client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens, **kwargs
                )
            except Exception as e:
                logger.error(f"Fallback LLM ({self._get_client_name(fallback_client)}) also failed: {e}")
        
        raise RuntimeError("All LLM clients failed")
    
    def _get_primary_client(self, task_type: str):
        routing = self._task_routing.get(task_type, 'claude')
        if routing == 'claude' and self.claude_client:
            return self.claude_client
        elif routing == 'openai' and self.openai_client:
            return self.openai_client
        return self.claude_client or self.openai_client
    
    def _get_fallback_client(self, task_type: str):
        routing = self._task_routing.get(task_type, 'claude')
        if routing == 'claude':
            return self.openai_client
        return self.claude_client
    
    def _get_client_name(self, client) -> str:
        if isinstance(client, BedrockLLMClient):
            return "Claude"
        return "OpenAI"


# =============================================================================
# CLAUDE VALIDATOR - Gene and Drug Validation
# =============================================================================
class ClaudeValidator:
    """
    Validates genes and drugs against disease using Claude.
    Prevents hallucinated disease relevance claims and fabricated drug recommendations.
    """
    
    # Validation status constants
    VALID = "VALID"           # Confirmed disease association
    PARTIAL = "PARTIAL"       # Indirect/weak association
    WEAK = "WEAK"             # Very limited evidence
    INVALID = "INVALID"       # No association - hallucinated
    FABRICATED = "FABRICATED" # Drug doesn't exist
    DANGEROUS = "DANGEROUS"   # Wrong indication (e.g., chemo for autoimmune)
    CONTRAINDICATED = "CONTRAINDICATED"  # Exists but wrong for this disease
    
    def __init__(self, llm_client, disease: str):
        self.llm_client = llm_client
        self.disease = disease
        self._gene_cache = {}
        self._drug_cache = {}
        self._pathway_cache = {}
    
    def validate_genes_batch(self, genes: List[Dict], batch_size: int = 15) -> Dict[str, Dict]:
        """Validate a batch of genes against the disease."""
        if not self.llm_client or not genes:
            return {}
        
        results = {}
        gene_list = [g.get('gene_symbol', g.get('gene', '')) for g in genes if g]
        gene_list = [g for g in gene_list if g and g not in self._gene_cache]
        
        # Return cached results
        for g in genes:
            symbol = g.get('gene_symbol', g.get('gene', ''))
            if symbol in self._gene_cache:
                results[symbol] = self._gene_cache[symbol]
        
        # Validate new genes in batches
        for i in range(0, len(gene_list), batch_size):
            batch = gene_list[i:i + batch_size]
            if batch:
                batch_results = self._validate_gene_batch(batch)
                results.update(batch_results)
                self._gene_cache.update(batch_results)
        
        return results
    
    def _validate_gene_batch(self, gene_symbols: List[str]) -> Dict[str, Dict]:
        """Validate a single batch of genes via Claude."""
        genes_str = ", ".join(gene_symbols)
        
        prompt = f"""You are a biomedical literature expert. Validate these genes for association with {self.disease}.

GENES TO VALIDATE: {genes_str}

VALIDATION CRITERIA:
- VALID: Multiple peer-reviewed studies directly link gene to {self.disease}
- PARTIAL: Indirect association (related pathway, similar disease, animal studies only)
- WEAK: Speculative/single low-quality study
- INVALID: NO published evidence linking to {self.disease}

Return ONLY a JSON object:
{{
  "GENE1": {{"status": "VALID|PARTIAL|WEAK|INVALID", "evidence": "brief explanation", "pubmed_exists": true/false}},
  "GENE2": ...
}}

CRITICAL: Uncharacterized proteins (ORF genes) are almost always INVALID unless proven otherwise."""

        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(self.llm_client, 'model_name', 'gpt-4'),
                messages=[
                    {"role": "system", "content": "You are a biomedical literature expert. Be conservative - only mark VALID if strong evidence exists. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2000,
                task_type="critical"
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            if '```' in content:
                content = content.split('```')[1].lstrip('json\n')
                if '```' in content:
                    content = content.split('```')[0]
            
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:
            logger.error(f"Gene validation failed: {e}")
        
        return {g: {"status": self.INVALID, "evidence": "Validation failed", "pubmed_exists": False} for g in gene_symbols}
    
    def validate_drugs_batch(self, drugs: List[Dict], patient_genes: List[str] = None, batch_size: int = 10) -> Dict[str, Dict]:
        """Validate drug recommendations against disease."""
        if not self.llm_client or not drugs:
            return {}
        
        results = {}
        drug_list = []
        
        for d in drugs:
            drug_name = d.get('drug_name', d.get('drug', d.get('name', '')))
            if drug_name and drug_name not in self._drug_cache:
                drug_list.append(d)
            elif drug_name in self._drug_cache:
                results[drug_name] = self._drug_cache[drug_name]
        
        for i in range(0, len(drug_list), batch_size):
            batch = drug_list[i:i + batch_size]
            if batch:
                batch_results = self._validate_drug_batch(batch, patient_genes)
                results.update(batch_results)
                self._drug_cache.update(batch_results)
        
        return results
    
    def _validate_drug_batch(self, drugs: List[Dict], patient_genes: List[str] = None) -> Dict[str, Dict]:
        """Validate a single batch of drugs via Claude."""
        drugs_info = [f"- {d.get('drug_name', d.get('drug', d.get('name', '')))}: target={d.get('target_gene', d.get('target', ''))}, mechanism={d.get('mechanism', d.get('drug_class', ''))}" for d in drugs]
        drugs_str = "\n".join(drugs_info)
        genes_str = ", ".join(patient_genes[:50]) if patient_genes else "not provided"
        
        prompt = f"""You are a clinical pharmacology expert. Validate these drug recommendations for {self.disease}.

DRUGS TO VALIDATE:
{drugs_str}

PATIENT'S DYSREGULATED GENES: {genes_str}

VALIDATION STATUS:
- VALID: Real drug, correct mechanism, approved/studied for {self.disease}
- PARTIAL: Real drug but off-label for {self.disease}
- CONTRAINDICATED: Real drug but WRONG indication for {self.disease}
- DANGEROUS: Drug would harm patient (e.g., chemotherapy for autoimmune)
- FABRICATED: Drug name doesn't exist

Return ONLY a JSON object:
{{
  "DrugName1": {{"status": "VALID|PARTIAL|CONTRAINDICATED|DANGEROUS|FABRICATED", "reason": "explanation", "fda_approved_for": "indication or null"}},
  "DrugName2": ...
}}

CRITICAL: Chemotherapy drugs are DANGEROUS for autoimmune diseases. Research compounds are not approved drugs."""

        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(self.llm_client, 'model_name', 'gpt-4'),
                messages=[
                    {"role": "system", "content": "You are a clinical pharmacology expert. Patient safety is paramount. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2500,
                task_type="critical"
            )
            
            content = response.choices[0].message.content.strip()
            
            if '```' in content:
                content = content.split('```')[1].lstrip('json\n')
                if '```' in content:
                    content = content.split('```')[0]
            
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:
            logger.error(f"Drug validation failed: {e}")
        
        return {d.get('drug_name', d.get('drug', 'unknown')): {"status": self.FABRICATED, "reason": "Validation failed"} for d in drugs}
    
    def filter_validated_genes(self, gene_mappings: List, validation_results: Dict) -> Tuple[List, List]:
        """Separate genes into validated and unknown significance."""
        validated, unknown = [], []
        
        for gene in gene_mappings:
            symbol = gene.gene if hasattr(gene, 'gene') else gene.get('gene_symbol', gene.get('gene', ''))
            validation = validation_results.get(symbol, {})
            status = validation.get('status', self.INVALID)
            
            if status in [self.VALID, self.PARTIAL]:
                validated.append(gene)
            else:
                if hasattr(gene, '__dict__'):
                    gene.validation_status = status
                    gene.validation_evidence = validation.get('evidence', 'No known disease association')
                unknown.append(gene)
        
        return validated, unknown
    
    def filter_validated_drugs(self, drugs: List[Dict], validation_results: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Separate drugs into valid, contraindicated, and removed."""
        valid, contraindicated, removed = [], [], []
        
        for drug in drugs:
            name = drug.get('drug_name', drug.get('drug', drug.get('name', '')))
            validation = validation_results.get(name, {})
            status = validation.get('status', self.FABRICATED)
            
            drug['validation_status'] = status
            drug['validation_reason'] = validation.get('reason', '')
            
            if status == self.VALID:
                valid.append(drug)
            elif status == self.PARTIAL:
                drug['off_label_note'] = f"Off-label for {self.disease}: {validation.get('reason', '')}"
                valid.append(drug)
            elif status in [self.CONTRAINDICATED, self.WEAK]:
                contraindicated.append(drug)
            else:
                removed.append(drug)
        
        return valid, contraindicated, removed
    
    def validate_pathways_batch(self, pathways: List[Dict], batch_size: int = 20) -> Dict[str, Dict]:
        """Validate pathway relevance to disease."""
        if not self.llm_client or not pathways:
            return {}
        
        results = {}
        pathway_list = [p.get('pathway_name', p.get('name', '')) for p in pathways if p]
        uncached = [p for p in pathway_list if p and p not in self._pathway_cache]
        
        # Return cached results
        for p in pathways:
            name = p.get('pathway_name', p.get('name', ''))
            if name in self._pathway_cache:
                results[name] = self._pathway_cache[name]
        
        # Validate new pathways in batches
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            if batch:
                batch_results = self._validate_pathway_batch(batch)
                results.update(batch_results)
                self._pathway_cache.update(batch_results)
        
        return results
    
    def _validate_pathway_batch(self, pathway_names: List[str]) -> Dict[str, Dict]:
        """Validate a single batch of pathways via Claude."""
        pathways_str = "\n".join(f"- {p}" for p in pathway_names)
        
        prompt = f"""You are a biomedical pathway expert. Validate these pathways for relevance to {self.disease}.

PATHWAYS TO VALIDATE:
{pathways_str}

VALIDATION CRITERIA:
- VALID: Pathway directly implicated in {self.disease} pathophysiology
- PARTIAL: Pathway indirectly related (general inflammation, metabolism)
- WEAK: Minimal/speculative connection to {self.disease}
- INVALID: No known connection to {self.disease}

Return ONLY a JSON object:
{{
  "Pathway Name 1": {{"status": "VALID|PARTIAL|WEAK|INVALID", "relevance": "brief explanation"}},
  "Pathway Name 2": ...
}}

Be strict - generic pathways (cell cycle, apoptosis) should be PARTIAL unless specifically dysregulated in {self.disease}."""

        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(self.llm_client, 'model_name', 'gpt-4'),
                messages=[
                    {"role": "system", "content": "You are a biomedical pathway expert. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2000,
                task_type="critical"
            )
            
            content = response.choices[0].message.content.strip()
            
            if '```' in content:
                content = content.split('```')[1].lstrip('json\n')
                if '```' in content:
                    content = content.split('```')[0]
            
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:
            logger.error(f"Pathway validation failed: {e}")
        
        return {p: {"status": self.WEAK, "relevance": "Validation failed"} for p in pathway_names}
    
    def filter_validated_pathways(self, pathways: List, validation_results: Dict) -> Tuple[List, List]:
        """Separate pathways into validated and weak/irrelevant."""
        validated, weak = [], []
        
        for pathway in pathways:
            name = pathway.pathway_name if hasattr(pathway, 'pathway_name') else pathway.get('pathway_name', '')
            validation = validation_results.get(name, {})
            status = validation.get('status', self.WEAK)
            
            if status in [self.VALID, self.PARTIAL]:
                validated.append(pathway)
            else:
                weak.append(pathway)
        
        return validated, weak

    def validate_cell_types_batch(self, cell_types: List[Dict], batch_size: int = 20) -> Dict[str, Dict]:
        """Validate cell type relevance to disease."""
        if not self.llm_client or not cell_types:
            return {}

        results = {}
        names = [c.get('cell_type', '') for c in cell_types if c]
        uncached = [n for n in names if n and n not in self._pathway_cache]

        for n in names:
            if n in self._pathway_cache:
                results[n] = self._pathway_cache[n]

        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            if batch:
                batch_results = self._validate_cell_type_batch(batch)
                results.update(batch_results)
                self._pathway_cache.update(batch_results)

        return results

    def _validate_cell_type_batch(self, cell_type_names: List[str]) -> Dict[str, Dict]:
        """Validate a batch of cell types against the disease via Claude."""
        types_str = "\n".join(f"- {c}" for c in cell_type_names)

        prompt = f"""You are an immunology and disease biology expert. Validate these immune/stromal cell types for relevance to {self.disease}.

CELL TYPES TO VALIDATE:
{types_str}

VALIDATION CRITERIA:
- VALID: Cell type directly implicated in {self.disease} pathophysiology (published evidence)
- PARTIAL: Indirectly related (general immune role, related conditions)
- WEAK: Minimal connection to {self.disease}
- INVALID: No known role in {self.disease}

Return ONLY a JSON object:
{{
  "Cell Type 1": {{"status": "VALID|PARTIAL|WEAK|INVALID", "relevance": "brief explanation"}},
  "Cell Type 2": ...
}}

Be conservative - generic cell types should be PARTIAL unless specifically dysregulated in {self.disease}."""

        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(self.llm_client, 'model_name', 'gpt-4'),
                messages=[
                    {"role": "system", "content": "You are an immunology expert. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2000,
                task_type="critical"
            )

            content = response.choices[0].message.content.strip()
            if '```' in content:
                content = content.split('```')[1].lstrip('json\n')
                if '```' in content:
                    content = content.split('```')[0]

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.error(f"Cell type validation failed: {e}")

        return {c: {"status": self.WEAK, "relevance": "Validation failed"} for c in cell_type_names}


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================
_claude_validator = None


def create_llm_client():
    """
    Factory function to create a SmartLLMRouter with both Claude and OpenAI clients.
    Configuration is read from environment variables.
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    claude_client = None
    openai_client = None
    
    # Try to create Claude (Bedrock) client
    use_bedrock = os.getenv("USE_BEDROCK", "false").lower() == "true"
    
    if use_bedrock and BOTO3_AVAILABLE:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        
        if aws_access_key_id and aws_secret_access_key:
            try:
                claude_client = BedrockLLMClient(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region,
                    model_id=model_id
                )
                print(f"    ✓ Claude client initialized ({model_id.split('.')[-1] if '.' in model_id else model_id})")
            except Exception as e:
                print(f"    ✗ Claude client failed: {e}")
    
    # Try to create OpenAI client
    if OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                openai_client = OpenAI(api_key=api_key)
                print(f"    ✓ OpenAI client initialized (fallback)")
            except Exception as e:
                print(f"    ✗ OpenAI client failed: {e}")
    
    if not claude_client and not openai_client:
        print("    ✗ No LLM clients available!")
        return None
    
    router = SmartLLMRouter(claude_client=claude_client, openai_client=openai_client)
    
    if claude_client and openai_client:
        print("    → Smart routing: Claude (complex/critical) + OpenAI (simple/fallback)")
    elif claude_client:
        print("    → Using Claude for all tasks")
    else:
        print("    → Using OpenAI for all tasks")
    
    return router


def get_claude_validator(llm_client, disease: str) -> ClaudeValidator:
    """Get or create ClaudeValidator instance."""
    global _claude_validator
    if _claude_validator is None or _claude_validator.disease != disease:
        _claude_validator = ClaudeValidator(llm_client, disease)
    return _claude_validator


# Gene limits for LLM context (centralized config)
DEFAULT_TOP_UPREGULATED_GENES = 500
DEFAULT_TOP_DOWNREGULATED_GENES = 500


def get_gene_limits() -> Dict[str, int]:
    """Get gene count limits for LLM context."""
    return {
        'upregulated': DEFAULT_TOP_UPREGULATED_GENES,
        'downregulated': DEFAULT_TOP_DOWNREGULATED_GENES
    }


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    'BedrockLLMClient',
    'SmartLLMRouter',
    'ClaudeValidator',
    'create_llm_client',
    'get_claude_validator',
    'get_gene_limits',
    'BOTO3_AVAILABLE',
    'OPENAI_AVAILABLE',
]
