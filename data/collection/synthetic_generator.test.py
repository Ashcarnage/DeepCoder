"""
Unit tests for synthetic data generator

Tests API providers, conversation generation, quality assessment,
and error handling for the synthetic agentic data generation system.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.collection.synthetic_generator import (
    SyntheticConfig,
    GroqProvider,
    TogetherProvider,
    HuggingFaceProvider,
    SyntheticDataGenerator,
    collect_synthetic_data
)
from data.collection.base_collector import CollectionConfig, DataItem


class TestSyntheticConfig:
    """Test SyntheticConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SyntheticConfig()
        
        assert config.provider == "groq"
        assert config.model_name == "llama3-8b-8192"
        assert config.temperature == 0.8
        assert config.max_tokens == 2048
        assert config.require_tool_usage is True
        assert config.require_reasoning is True
        assert len(config.scenario_types) == 8
        assert "tool_usage" in config.scenario_types
        assert "multi_step_reasoning" in config.scenario_types
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = SyntheticConfig(
            provider="together",
            model_name="custom-model",
            temperature=0.5,
            scenario_types=["tool_usage", "error_handling"]
        )
        
        assert config.provider == "together"
        assert config.model_name == "custom-model"
        assert config.temperature == 0.5
        assert config.scenario_types == ["tool_usage", "error_handling"]


class TestAPIProviders:
    """Test API provider implementations"""
    
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session"""
        session = AsyncMock()
        return session
    
    @pytest.fixture
    def config_with_key(self):
        """Config with API key"""
        return SyntheticConfig(api_key="test_key_123")
    
    @pytest.mark.asyncio
    async def test_groq_provider_success(self, mock_session, config_with_key):
        """Test successful Groq API call"""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response from Groq API"
                    }
                }
            ]
        }
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        provider = GroqProvider(config_with_key)
        provider.session = mock_session
        
        messages = [{"role": "user", "content": "Test message"}]
        result = await provider.generate_completion(messages)
        
        assert result == "This is a test response from Groq API"
        mock_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_groq_provider_no_api_key(self):
        """Test Groq provider without API key"""
        config = SyntheticConfig(api_key=None)
        provider = GroqProvider(config)
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(ValueError, match="Groq API key required"):
            await provider.generate_completion(messages)
    
    @pytest.mark.asyncio
    async def test_groq_provider_api_error(self, mock_session, config_with_key):
        """Test Groq API error handling"""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text.return_value = "Unauthorized"
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        provider = GroqProvider(config_with_key)
        provider.session = mock_session
        
        messages = [{"role": "user", "content": "Test"}]
        result = await provider.generate_completion(messages)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_together_provider_success(self, mock_session, config_with_key):
        """Test successful Together AI call"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Together AI response"
                    }
                }
            ]
        }
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        provider = TogetherProvider(config_with_key)
        provider.session = mock_session
        
        messages = [{"role": "user", "content": "Test"}]
        result = await provider.generate_completion(messages)
        
        assert result == "Together AI response"
    
    @pytest.mark.asyncio
    async def test_huggingface_provider_success(self, mock_session, config_with_key):
        """Test successful HuggingFace API call"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = [
            {"generated_text": "HuggingFace response"}
        ]
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        config_with_key.model_name = "microsoft/DialoGPT-medium"
        provider = HuggingFaceProvider(config_with_key)
        provider.session = mock_session
        
        messages = [{"role": "user", "content": "Test"}]
        result = await provider.generate_completion(messages)
        
        assert result == "HuggingFace response"
    
    def test_huggingface_messages_to_prompt(self, config_with_key):
        """Test HuggingFace message conversion"""
        provider = HuggingFaceProvider(config_with_key)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        prompt = provider._messages_to_prompt(messages)
        
        assert "System: You are a helpful assistant" in prompt
        assert "Human: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "Human: How are you?" in prompt
        assert prompt.endswith("Assistant: ")


class TestSyntheticDataGenerator:
    """Test synthetic data generator"""
    
    @pytest.fixture
    def collection_config(self):
        """Basic collection config"""
        return CollectionConfig(
            output_dir="test_output",
            quality_threshold=0.3
        )
    
    @pytest.fixture
    def synthetic_config(self):
        """Basic synthetic config"""
        return SyntheticConfig(
            provider="groq",
            api_key="test_key",
            scenario_types=["tool_usage", "error_handling"]
        )
    
    @pytest.fixture
    def mock_provider(self):
        """Mock API provider"""
        provider = AsyncMock()
        provider.generate_completion.return_value = (
            "I'll help you with this task step by step. First, I'll use the web_search tool "
            "to find relevant information. Then I'll analyze the data using data_processor."
        )
        return provider
    
    def test_generator_initialization(self, collection_config, synthetic_config):
        """Test generator initialization"""
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        
        assert generator.source_name == "synthetic"
        assert generator.synthetic_config == synthetic_config
        assert len(generator.SCENARIO_TEMPLATES) == 8
        assert len(generator.AVAILABLE_TOOLS) == 8
    
    @pytest.mark.asyncio
    async def test_get_item_ids(self, collection_config, synthetic_config):
        """Test item ID generation"""
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        
        item_ids = list(await generator.get_item_ids())
        
        # 2 scenario types * 8 scenarios per type * 3 variations = 48 IDs
        assert len(item_ids) == 48
        
        # Check format
        for item_id in item_ids[:6]:  # Check first 6
            parts = item_id.split('_')
            assert len(parts) == 3
            assert parts[0] in ["tool_usage", "error_handling"]
            assert parts[1].isdigit()
            assert parts[2].isdigit()
    
    @pytest.mark.asyncio
    async def test_fetch_item_success(self, collection_config, synthetic_config, mock_provider):
        """Test successful item fetching"""
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        generator.provider = mock_provider
        
        # Mock provider context manager
        mock_provider.__aenter__.return_value = mock_provider
        mock_provider.__aexit__.return_value = None
        
        item = await generator.fetch_item("tool_usage_0_1")
        
        assert item is not None
        assert item.source == "synthetic"
        assert item.item_id == "tool_usage_0_1"
        assert item.content["conversation_type"] == "synthetic_tool_usage"
        assert "turns" in item.content
        assert "agentic_patterns" in item.content
        assert item.metadata["scenario_type"] == "tool_usage"
        assert item.metadata["scenario_index"] == 0
        assert item.metadata["variation"] == 1
    
    @pytest.mark.asyncio
    async def test_fetch_item_invalid_id(self, collection_config, synthetic_config):
        """Test fetching with invalid item ID"""
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        
        # Invalid format
        assert await generator.fetch_item("invalid_id") is None
        assert await generator.fetch_item("tool_usage_0") is None
        assert await generator.fetch_item("unknown_0_0") is None
        assert await generator.fetch_item("tool_usage_999_0") is None
    
    def test_create_scenario_variation(self, collection_config, synthetic_config):
        """Test scenario variation creation"""
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        
        base_scenario = "Debug a Python script"
        
        var0 = generator._create_scenario_variation(base_scenario, "code_debugging", 0)
        var1 = generator._create_scenario_variation(base_scenario, "code_debugging", 1)
        var2 = generator._create_scenario_variation(base_scenario, "code_debugging", 2)
        
        assert var0 == base_scenario
        assert "trouble" in var1.lower()
        assert "team" in var2.lower()
    
    def test_process_assistant_response(self, collection_config, synthetic_config):
        """Test assistant response processing"""
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        
        response = (
            "I'll help you step by step. First, I'll use web_search to find information. "
            "Then I'll analyze the error using code_analyzer. This approach will break down "
            "the problem systematically."
        )
        
        agentic_patterns = []
        processed = generator._process_assistant_response(response, "tool_usage", agentic_patterns)
        
        assert processed["role"] == "assistant"
        assert processed["content"] == response
        assert "tool_calls" in processed
        assert len(processed["tool_calls"]) == 2  # web_search and code_analyzer
        
        # Check detected patterns
        assert "tool_calling" in agentic_patterns
        assert "step_by_step_reasoning" in agentic_patterns
        assert "problem_decomposition" in agentic_patterns
    
    def test_extract_tool_calls(self, collection_config, synthetic_config):
        """Test tool call extraction"""
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        
        response = "I'll use web_search to find data, then data_processor to analyze it."
        tool_calls = generator._extract_tool_calls(response)
        
        assert len(tool_calls) == 2
        
        # Check first tool call
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "web_search"
        assert "arguments" in tool_calls[0]["function"]
        
        # Check second tool call
        assert tool_calls[1]["function"]["name"] == "data_processor"
    
    def test_assess_complexity(self, collection_config, synthetic_config):
        """Test complexity assessment"""
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        
        # High complexity scenario
        high_turns = [
            {"role": "user", "content": "Long user message " * 50},
            {
                "role": "assistant", 
                "content": "Long assistant response " * 50,
                "tool_calls": [{"tool": "web_search"}]
            },
            {"role": "user", "content": "Follow up"},
            {
                "role": "assistant",
                "content": "Another long response " * 50,
                "tool_calls": [{"tool": "data_processor"}]
            }
        ]
        high_patterns = ["step_by_step_reasoning", "problem_decomposition", "tool_calling"]
        
        complexity = generator._assess_complexity(high_turns, high_patterns)
        assert complexity == "high"
        
        # Low complexity scenario
        low_turns = [
            {"role": "user", "content": "Short question"},
            {"role": "assistant", "content": "Short answer"}
        ]
        low_patterns = []
        
        complexity = generator._assess_complexity(low_turns, low_patterns)
        assert complexity == "low"
    
    def test_assess_quality(self, collection_config, synthetic_config):
        """Test quality assessment"""
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        
        # High quality conversation
        high_quality_content = {
            "turns": [
                {"role": "user", "content": "User message " * 20},
                {
                    "role": "assistant", 
                    "content": "Assistant response " * 30,
                    "tool_calls": [{"tool": "web_search"}]
                },
                {"role": "user", "content": "Follow up"},
                {"role": "assistant", "content": "Final response " * 25}
            ],
            "agentic_patterns": ["tool_calling", "step_by_step_reasoning", "planning"],
            "complexity": "high"
        }
        
        high_item = DataItem(
            source="synthetic",
            item_id="test",
            content=high_quality_content,
            quality_score=0.0,
            timestamp=datetime.now(),
            metadata={}
        )
        
        quality_score = asyncio.run(generator.assess_quality(high_item))
        assert quality_score > 0.8
        
        # Low quality conversation
        low_quality_content = {
            "turns": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}
            ],
            "agentic_patterns": [],
            "complexity": "low"
        }
        
        low_item = DataItem(
            source="synthetic",
            item_id="test",
            content=low_quality_content,
            quality_score=0.0,
            timestamp=datetime.now(),
            metadata={}
        )
        
        quality_score = asyncio.run(generator.assess_quality(low_item))
        assert quality_score < 0.5


class TestCollectSyntheticData:
    """Test main collection function"""
    
    @pytest.mark.asyncio
    async def test_collect_synthetic_data(self):
        """Test main collection function"""
        collection_config = CollectionConfig(
            output_dir="test_output",
            quality_threshold=0.3
        )
        
        synthetic_config = SyntheticConfig(
            provider="groq",
            api_key="test_key",
            scenario_types=["tool_usage"]  # Just one for testing
        )
        
        # Mock the generator
        with patch('data.collection.synthetic_generator.SyntheticDataGenerator') as MockGenerator:
            mock_generator_instance = AsyncMock()
            mock_metrics = MagicMock()
            mock_metrics.total_collected = 5
            mock_generator_instance.collect.return_value = mock_metrics
            mock_generator_instance.output_dir = Path("test_output/synthetic")
            
            MockGenerator.return_value = mock_generator_instance
            
            result = await collect_synthetic_data(
                collection_config,
                synthetic_config,
                max_items=10
            )
            
            assert result["source"] == "synthetic"
            assert result["provider"] == "groq"
            assert result["model"] == "llama3-8b-8192"
            assert result["scenario_types"] == ["tool_usage"]
            assert "output_directory" in result


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_provider_creation_error(self):
        """Test provider creation with invalid provider"""
        collection_config = CollectionConfig()
        synthetic_config = SyntheticConfig(provider="invalid_provider")
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            SyntheticDataGenerator(collection_config, synthetic_config)
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, collection_config, synthetic_config):
        """Test network error handling"""
        mock_provider = AsyncMock()
        mock_provider.generate_completion.side_effect = Exception("Network error")
        mock_provider.__aenter__.return_value = mock_provider
        mock_provider.__aexit__.return_value = None
        
        generator = SyntheticDataGenerator(collection_config, synthetic_config)
        generator.provider = mock_provider
        
        item = await generator.fetch_item("tool_usage_0_0")
        assert item is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 