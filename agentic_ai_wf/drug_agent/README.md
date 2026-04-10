# Drug Discovery Agent

A disease-agnostic, RAG-powered drug recommendation system designed for integration with the Reporting Pipeline Agent.

## Key Features

- **Fully Dynamic**: No hardcoded disease, drug, or gene information - learns from the knowledge base
- **Qdrant Cloud**: Scalable vector storage with semantic search
- **HuggingFace PubMed-BERT**: Biomedical text embeddings from HuggingFace Hub
- **Inter-Agent Communication**: Standardized message interface for multi-agent systems
- **Three Mapping Types**:
  - Gene → Drug (direct target relationships)
  - Disease → Drug (approved therapies)
  - Pathway → Drug (pathway-targeted therapeutics)

## Quick Start

### 1. Set Environment Variables

```bash
# Required for Qdrant Cloud
export QDRANT_URL="https://your-cluster.cloud.qdrant.io"
export QDRANT_API_KEY="your-api-key"

# Optional: HuggingFace token for private models
export HF_TOKEN="your-hf-token"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize Agent

```python
from drug_discovery_agent import DrugDiscoveryAgent

# Initialize (auto-connects to Qdrant Cloud)
agent = DrugDiscoveryAgent()

# Or with explicit configuration
from drug_discovery_agent import create_agent

agent = create_agent(
    qdrant_url="https://your-cluster.cloud.qdrant.io",
    qdrant_api_key="your-api-key",
    collection_name="drug_knowledge_base",
    embedding_model="NeuML/pubmedbert-base-embeddings",
)
```

### 4. Ingest Knowledge Base

```python
# One-time ingestion of gene JSON files
result = agent.ingest_gene_data(
    json_directory="/mnt/c/Users/7wraa/Downloads/Ayass-v5.26/GeneALaCart-AllGenes",
    recreate_collection=True,
)

print(f"Ingested {result.total_documents_created} documents")
```

### 5. Generate Recommendations

```python
from drug_discovery_agent import DrugAgentInput
from drug_discovery_agent.models.data_models import GeneMapping, PathwayMapping

# Create input from your transcriptomic data
input_data = DrugAgentInput(
    disease_name="Breast Cancer",
    gene_mappings=[
        GeneMapping(gene="ERBB2", log2fc=4.25, observed_direction="up"),
        GeneMapping(gene="ESR1", log2fc=-3.6, observed_direction="down"),
        GeneMapping(gene="BRCA2", log2fc=1.67, observed_direction="up"),
    ],
    pathway_mappings=[
        PathwayMapping(pathway_name="PI3K-AKT Signaling", regulation="Up"),
    ],
)

# Generate recommendations
output = agent.generate_recommendations(input_data)

# Access results
for rec in output.drug_recommendations[:5]:
    print(f"{rec.drug_name}: {rec.composite_score:.3f} - {rec.approval_status}")
```

## Integration with Reporting Pipeline

```python
# In your reporting_pipeline_agent.py

from drug_discovery_agent import DrugDiscoveryAgent, DrugAgentInput

class ReportingPipelineAgent:
    def __init__(self):
        # Initialize Drug Discovery Agent
        self.drug_agent = DrugDiscoveryAgent()
    
    def run_pipeline(self, disease_name: str, degs_csv: str, pathways_csv: str):
        # ... existing DEG and Pathway analysis ...
        
        # Create drug agent input from pipeline data
        drug_input = DrugAgentInput.from_pipeline_data(
            disease_name=disease_name,
            gene_mappings=self.gene_mappings,
            pathway_mappings=self.pathway_mappings,
        )
        
        # Generate drug recommendations
        drug_output = self.drug_agent.generate_recommendations(drug_input)
        
        # Generate report section
        report_section = self.drug_agent.generate_report_section(drug_output)
        
        return drug_output
```

## Inter-Agent Communication

The agent supports standardized messaging for multi-agent systems:

```python
from drug_discovery_agent.drug_discovery_agent import AgentMessage

# Create a message from another agent
message = AgentMessage(
    message_id="msg-123",
    source_agent="deg_agent",
    target_agent="drug_discovery_agent",
    message_type="query",
    action="get_recommendations",
    payload={
        "disease_name": "Breast Cancer",
        "gene_mappings": [...],
        "pathway_mappings": [...],
    },
)

# Handle the message
response = agent.handle_message(message)

if response.success:
    recommendations = response.data
```

### Supported Actions

| Action | Description |
|--------|-------------|
| `get_recommendations` | Generate drug recommendations |
| `ingest_data` | Ingest gene data files |
| `health_check` | Check agent health |
| `get_stats` | Get knowledge base statistics |
| `query_gene` | Query drugs for a specific gene |
| `query_disease` | Query information for a disease |

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `QDRANT_URL` | Yes | Qdrant Cloud URL |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | No | Collection name (default: drug_knowledge_base) |
| `HF_TOKEN` | No | HuggingFace token for private models |

### Config File (Optional)

```yaml
# config/drug_agent_config.yaml

qdrant:
  url: null  # Set via QDRANT_URL env var
  api_key: null  # Set via QDRANT_API_KEY env var
  collection_name: "drug_knowledge_base"
  timeout_seconds: 60
  use_https: true
  prefer_grpc: true

embedding:
  model_name: "NeuML/pubmedbert-base-embeddings"
  device: "auto"
  batch_size: 32
  cache_enabled: true

retrieval:
  default_top_k: 50
  min_relevance_score: 0.3

ranking:
  weights:
    relevance: 0.30
    gene_match: 0.35
    evidence: 0.20
    approval_status: 0.15

output:
  max_recommendations: 15
```

## Project Structure

```
drug_discovery_agent/
├── __init__.py                 # Main exports
├── drug_discovery_agent.py     # Main agent class
├── requirements.txt
├── config/
│   ├── settings.py             # Configuration management
│   └── drug_agent_config.yaml  # Default config
├── models/
│   └── data_models.py          # Data classes
├── ingestion/
│   ├── json_parser.py          # Parse gene JSON files
│   ├── data_normalizer.py      # Dynamic normalization
│   └── document_generator.py   # Generate vector documents
├── embedding/
│   └── embedder.py             # HuggingFace PubMed-BERT
├── storage/
│   └── qdrant_client.py        # Qdrant Cloud client
├── retrieval/
│   ├── query_builder.py        # Dynamic query building
│   └── hybrid_search.py        # Multi-query RRF fusion
├── recommendation/
│   ├── drug_ranker.py          # Multi-factor ranking
│   ├── evidence_compiler.py    # Evidence aggregation
│   └── report_generator.py     # Report sections
├── utils/
│   ├── disease_mapper.py       # Dynamic disease mapping
│   └── gene_resolver.py        # Dynamic gene resolution
└── tests/
    └── test_drug_agent.py
```

## Dynamic Learning

The agent learns mappings dynamically during ingestion:

```python
# During ingestion, the agent learns:
# - Disease name variations from data
# - Gene aliases from JSON files
# - Drug name variations

# Access learned mappings
disease_aliases = agent.disease_mapper.export_mappings()
gene_aliases = agent.gene_resolver.export_mappings()
```

You can also provide external mapping files:

```python
# Load external HGNC gene mappings
agent.gene_resolver.load_mappings_from_file("hgnc_aliases.csv")

# Load custom disease mappings
agent.disease_mapper.load_mappings_from_file("disease_synonyms.json")
```

## API Reference

### DrugDiscoveryAgent

```python
agent = DrugDiscoveryAgent(
    config_path=None,      # Path to YAML config
    settings=None,         # Settings object
    auto_connect=True,     # Connect to Qdrant on init
)

# Ingestion
result = agent.ingest_gene_data(json_directory, recreate_collection=False)

# Recommendations
output = agent.generate_recommendations(input_data)

# Simple query
output = agent.query_drugs_for_disease(
    disease_name="Breast Cancer",
    top_genes=["ERBB2", "ESR1"],
)

# Report generation
section = agent.generate_report_section(output)

# Inter-agent messaging
response = agent.handle_message(message)

# Health check
status = agent.health_check()
```

### DrugAgentInput

```python
input_data = DrugAgentInput(
    disease_name="Breast Cancer",
    gene_mappings=[...],
    pathway_mappings=[...],
)

# Or from existing pipeline data
input_data = DrugAgentInput.from_pipeline_data(
    disease_name=disease_name,
    gene_mappings=gene_dicts,
    pathway_mappings=pathway_dicts,
)
```

## Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=drug_discovery_agent
```

## License

Internal use - Ayass BioScience
