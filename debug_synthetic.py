#!/usr/bin/env python3
"""
Debug script for synthetic data generation
"""
import os
import asyncio
import logging
from data.collection.synthetic_generator import SyntheticConfig, SyntheticDataGenerator, GroqProvider
from data.collection.base_collector import CollectionConfig

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

async def test_groq_provider():
    """Test the Groq provider directly"""
    print("🔧 Testing Groq Provider Directly")
    print("=" * 40)
    
    # Load environment
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except:
        pass
    
    api_key = os.getenv('GROQ_API_KEY')
    
    config = SyntheticConfig(
        provider="groq",
        api_key=api_key,
        model_name="llama3-8b-8192"
    )
    
    provider = GroqProvider(config)
    
    async with provider:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Say hello and introduce yourself."}
        ]
        
        print(f"📤 Sending request to Groq...")
        response = await provider.generate_completion(messages)
        
        if response:
            print(f"✅ Groq response: {response[:100]}...")
            return True
        else:
            print(f"❌ No response from Groq")
            return False

async def test_synthetic_generation():
    print("🔍 Testing Synthetic Data Generation")
    print("=" * 50)
    
    # Load environment
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except:
        pass
    
    # Check API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ No GROQ_API_KEY found")
        return
    
    print(f"✅ Found API key: {api_key[:10]}...")
    
    # Test Groq provider first
    groq_works = await test_groq_provider()
    if not groq_works:
        print("❌ Groq provider not working, stopping")
        return False
    
    # Create configs with working model
    collection_config = CollectionConfig(
        output_dir="debug_output",
        quality_threshold=0.3
    )
    
    synthetic_config = SyntheticConfig(
        provider="groq",
        api_key=api_key,
        model_name="llama3-8b-8192",  # Use known working Groq model
        scenario_types=["tool_usage"],
        scenarios_per_batch=1,
        max_turns_per_conversation=4,
        temperature=0.7,
        require_tool_usage=False,
        require_reasoning=False
    )
    
    print(f"📝 Config: provider={synthetic_config.provider}, model={synthetic_config.model_name}")
    
    # Test generation
    generator = SyntheticDataGenerator(collection_config, synthetic_config)
    
    try:
        print("🚀 Testing item generation...")
        
        # Get item IDs properly 
        item_ids = await generator.get_item_ids()
        
        # Test with first item ID
        first_item_id = next(item_ids)
        print(f"📋 Testing item: {first_item_id}")
        
        item = await generator.fetch_item(first_item_id)
        if item:
            print(f"✅ Generated item successfully!")
            print(f"   Content length: {len(str(item.content))}")
            print(f"   Conversation turns: {len(item.content.get('turns', []))}")
            print(f"   Agentic patterns: {item.content.get('agentic_patterns', [])}")
            return True
        else:
            print(f"❌ Failed to generate item")
            return False
    
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_synthetic_generation())
    print(f"\n{'🎉 Success!' if success else '❌ Failed'}") 