"""
Tests for Drug Discovery Agent
==============================

Run with: pytest tests/ -v
"""

import pytest
from typing import List, Dict

# Import models
from drug_agent.models.data_models import (
    GeneMapping, PathwayMapping, DrugAgentInput,
    DrugRecommendation, DrugAgentOutput, DocumentType,
)

# Import utilities
from drug_agent.utils.disease_mapper import DiseaseMapper
from drug_agent.utils.gene_resolver import GeneResolver

# Import config
from drug_agent.config.settings import Settings, RankingWeights


# =============================================================================
# Test Data Models
# =============================================================================

class TestGeneMapping:
    """Tests for GeneMapping model."""
    
    def test_from_dict_basic(self):
        data = {
            "gene": "ERBB2",
            "log2fc": 4.25,
            "adj_p_value": 0.001,
            "observed_direction": "up",
        }
        mapping = GeneMapping.from_dict(data)
        
        assert mapping.gene == "ERBB2"
        assert mapping.log2fc == 4.25
        assert mapping.observed_direction == "up"
    
    def test_from_dict_with_pipeline_keys(self):
        """Test parsing with keys from existing pipeline."""
        data = {
            "Gene": "BRCA2",
            "Patient_LFC_mean": 1.67,
            "Patient_LFC_Trend": "UP",
        }
        mapping = GeneMapping.from_dict(data)
        
        assert mapping.gene == "BRCA2"
        assert mapping.log2fc == 1.67
        assert mapping.observed_direction == "up"
    
    def test_handles_none_values(self):
        """Test handling of None values."""
        data = {
            "gene": "TEST",
            "log2fc": None,
            "adj_p_value": None,
        }
        mapping = GeneMapping.from_dict(data)
        
        assert mapping.gene == "TEST"
        assert mapping.log2fc == 0
        assert mapping.adj_p_value == 1.0


class TestPathwayMapping:
    """Tests for PathwayMapping model."""
    
    def test_from_dict_basic(self):
        data = {
            "Pathway_Name": "Apoptosis",
            "Pathway_Source": "KEGG",
            "P_Value": 1.88e-09,
            "Regulation": "Down",
            "Pathway_Associated_Genes": "BAX,BCL2,PUMA",
        }
        mapping = PathwayMapping.from_dict(data)
        
        assert mapping.pathway_name == "Apoptosis"
        assert mapping.pathway_source == "KEGG"
        assert mapping.regulation == "Down"
        assert "BAX" in mapping.input_genes
    
    def test_handles_gene_list(self):
        """Test handling gene list as list."""
        data = {
            "pathway_name": "Test",
            "input_genes": ["GENE1", "GENE2"],
        }
        mapping = PathwayMapping.from_dict(data)
        assert mapping.input_genes == ["GENE1", "GENE2"]


class TestDrugAgentInput:
    """Tests for DrugAgentInput model."""
    
    def test_get_top_genes(self):
        input_data = DrugAgentInput(
            disease_name="Breast Cancer",
            gene_mappings=[
                GeneMapping(gene="ERBB2", composite_score=10),
                GeneMapping(gene="ESR1", composite_score=8),
                GeneMapping(gene="BRCA2", composite_score=6),
            ],
        )
        
        top = input_data.get_top_genes(2)
        assert len(top) == 2
        assert top[0] == "ERBB2"
        assert top[1] == "ESR1"
    
    def test_get_upregulated_genes(self):
        input_data = DrugAgentInput(
            disease_name="Test",
            gene_mappings=[
                GeneMapping(gene="GENE1", observed_direction="up"),
                GeneMapping(gene="GENE2", observed_direction="down"),
                GeneMapping(gene="GENE3", observed_direction="up"),
            ],
        )
        
        up = input_data.get_upregulated_genes()
        assert len(up) == 2
        assert "GENE1" in up
        assert "GENE3" in up
    
    def test_get_gene_directions(self):
        input_data = DrugAgentInput(
            disease_name="Test",
            gene_mappings=[
                GeneMapping(gene="GENE1", observed_direction="up"),
                GeneMapping(gene="GENE2", observed_direction="down"),
            ],
        )
        
        directions = input_data.get_gene_directions()
        assert directions["GENE1"] == "up"
        assert directions["GENE2"] == "down"


# =============================================================================
# Test Dynamic Utilities
# =============================================================================

class TestDiseaseMapper:
    """Tests for dynamic DiseaseMapper."""
    
    def test_normalize_unknown(self):
        """Test that unknown diseases are title-cased."""
        mapper = DiseaseMapper()
        result = mapper.normalize("some random disease name")
        assert result == "Some Random Disease Name"
    
    def test_add_and_resolve_mapping(self):
        """Test dynamic mapping addition."""
        mapper = DiseaseMapper()
        mapper.add_mapping("BC", "Breast Cancer")
        
        assert mapper.normalize("BC") == "Breast Cancer"
        assert mapper.normalize("bc") == "Breast Cancer"
    
    def test_learn_from_data(self):
        """Test learning from data."""
        mapper = DiseaseMapper()
        mapper.learn_from_data("Breast Cancer", ["breast carcinoma", "mammary cancer"])
        
        aliases = mapper.get_aliases("Breast Cancer")
        assert "breast carcinoma" in aliases
    
    def test_is_same_disease(self):
        """Test disease comparison."""
        mapper = DiseaseMapper()
        mapper.add_mapping("bc", "Breast Cancer")
        mapper.add_mapping("breast carcinoma", "Breast Cancer")
        
        assert mapper.is_same_disease("bc", "breast carcinoma")


class TestGeneResolver:
    """Tests for dynamic GeneResolver."""
    
    def test_resolve_unknown(self):
        """Test that unknown genes are uppercased."""
        resolver = GeneResolver()
        result = resolver.resolve("newgene123")
        assert result == "NEWGENE123"
    
    def test_add_and_resolve_mapping(self):
        """Test dynamic mapping addition."""
        resolver = GeneResolver()
        resolver.add_mapping("HER2", "ERBB2")
        
        assert resolver.resolve("HER2") == "ERBB2"
        assert resolver.resolve("her2") == "ERBB2"
    
    def test_learn_from_data(self):
        """Test learning from data."""
        resolver = GeneResolver()
        resolver.learn_from_data("ERBB2", ["HER2", "NEU"])
        
        aliases = resolver.get_aliases("ERBB2")
        assert "HER2" in aliases
        assert "NEU" in aliases
    
    def test_are_same_gene(self):
        """Test gene comparison."""
        resolver = GeneResolver()
        resolver.add_mapping("HER2", "ERBB2")
        resolver.add_mapping("NEU", "ERBB2")
        
        assert resolver.are_same_gene("HER2", "NEU")
    
    def test_expand_gene_list(self):
        """Test gene list expansion."""
        resolver = GeneResolver()
        resolver.add_mapping("HER2", "ERBB2")
        
        expanded = resolver.expand_gene_list(["ERBB2"])
        assert "ERBB2" in expanded
        assert "HER2" in expanded


# =============================================================================
# Test Configuration
# =============================================================================

class TestSettings:
    """Tests for Settings configuration."""
    
    def test_default_settings(self):
        settings = Settings()
        
        assert settings.qdrant.collection_name == "drug_knowledge_base"
        assert settings.embedding.model_name == "NeuML/pubmedbert-base-embeddings"
        assert settings.retrieval.default_top_k == 50
    
    def test_ranking_weights(self):
        weights = RankingWeights()
        
        total = weights.relevance + weights.gene_match + weights.evidence + weights.approval_status
        assert abs(total - 1.0) < 0.01
    
    def test_from_dict(self):
        config = {
            "qdrant": {"url": "custom-host", "collection_name": "custom"},
            "retrieval": {"default_top_k": 100},
        }
        settings = Settings.from_dict(config)
        
        assert settings.qdrant.url == "custom-host"
        assert settings.qdrant.collection_name == "custom"
        assert settings.retrieval.default_top_k == 100
    
    def test_validate_missing_url(self):
        settings = Settings()
        settings.qdrant.url = None
        
        errors = settings.validate()
        assert len(errors) > 0
        assert any("QDRANT_URL" in e for e in errors)


# =============================================================================
# Test Document Types
# =============================================================================

class TestDocumentType:
    """Tests for DocumentType enum."""
    
    def test_document_types(self):
        assert DocumentType.GENE_DRUG.value == "gene_drug"
        assert DocumentType.DISEASE_DRUG.value == "disease_drug"
        assert DocumentType.PATHWAY_DRUG.value == "pathway_drug"
        assert DocumentType.GENE_DISEASE_CONTEXT.value == "gene_disease_context"


# =============================================================================
# Test DrugRecommendation
# =============================================================================

class TestDrugRecommendation:
    """Tests for DrugRecommendation model."""
    
    def test_to_dict(self):
        rec = DrugRecommendation(
            drug_name="Trastuzumab",
            target_genes=["ERBB2"],
            approval_status="FDA Approved",
            composite_score=0.85,
        )
        
        result = rec.to_dict()
        assert result["drug_name"] == "Trastuzumab"
        assert "ERBB2" in result["target_genes"]
        assert result["composite_score"] == 0.85


# =============================================================================
# Test Inter-Agent Communication
# =============================================================================

class TestAgentMessage:
    """Tests for AgentMessage."""
    
    def test_message_creation(self):
        from drug_agent.drug_agent import AgentMessage
        
        msg = AgentMessage(
            message_id="test-123",
            source_agent="test_agent",
            target_agent="drug_agent",
            message_type="query",
            action="get_recommendations",
            payload={"disease_name": "Test"},
        )
        
        assert msg.message_id == "test-123"
        assert msg.action == "get_recommendations"
    
    def test_message_to_dict(self):
        from drug_agent.drug_agent import AgentMessage
        
        msg = AgentMessage(
            message_id="test-123",
            source_agent="test",
            target_agent="drug",
            message_type="query",
            action="test",
        )
        
        result = msg.to_dict()
        assert "message_id" in result
        assert "timestamp" in result


# =============================================================================
# Integration Tests (Require Qdrant Cloud)
# =============================================================================

@pytest.mark.integration
class TestDrugDiscoveryAgentIntegration:
    """Integration tests requiring Qdrant Cloud."""
    
    @pytest.fixture
    def sample_input(self):
        return DrugAgentInput(
            disease_name="Breast Cancer",
            gene_mappings=[
                GeneMapping(gene="ERBB2", log2fc=4.25, observed_direction="up", composite_score=10),
                GeneMapping(gene="ESR1", log2fc=-3.6, observed_direction="down", composite_score=8),
            ],
            pathway_mappings=[
                PathwayMapping(pathway_name="Apoptosis", regulation="Down", p_value=1e-9),
            ],
        )
    
    @pytest.mark.skip(reason="Requires Qdrant Cloud connection")
    def test_health_check(self):
        from drug_agent import DrugDiscoveryAgent
        
        agent = DrugDiscoveryAgent(auto_connect=False)
        status = agent.health_check()
        
        assert "qdrant_connected" in status
        assert "agent_id" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
