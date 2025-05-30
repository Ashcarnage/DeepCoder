#!/usr/bin/env python3
"""
Test collector context management
"""
import os
import asyncio
import logging
from data.collection.synthetic_generator import SyntheticConfig, SyntheticDataGenerator
from data.collection.base_collector import CollectionConfig

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def test():
    # Load env
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except:
        pass
    
    api_key = os.getenv('GROQ_API_KEY')
    
    config = CollectionConfig(output_dir='debug_output', quality_threshold=0.3)
    synthetic_config = SyntheticConfig(
        provider='groq', 
        api_key=api_key, 
        model_name='llama3-8b-8192',
        scenario_types=["tool_usage"],
        max_turns_per_conversation=2
    )
    
    generator = SyntheticDataGenerator(config, synthetic_config)
    
    print("🧪 Testing with collector async context...")
    
    # Test using the collector's async context
    async with generator:
        item_ids = await generator.get_item_ids()
        first_item_id = next(item_ids)
        print(f'📋 Testing item: {first_item_id}')
        
        # Test fetch_item step by step
        print("🔍 Testing fetch_item step by step...")
        
        # Parse item ID (using pipe separator)
        parts = first_item_id.split('|')
        print(f"📝 Item ID parts: {parts}")
        
        if len(parts) != 3:
            print("❌ Invalid item ID format")
            return
        
        scenario_type, scenario_idx, variation = parts
        print(f"📝 Parsed: type={scenario_type}, idx={scenario_idx}, var={variation}")
        
        if scenario_type not in generator.SCENARIO_TEMPLATES:
            print(f"❌ Scenario type {scenario_type} not found")
            return
        
        try:
            scenario_idx = int(scenario_idx)
            variation = int(variation)
            print(f"📝 Converted: idx={scenario_idx}, var={variation}")
        except ValueError as e:
            print(f"❌ Failed to convert indices: {e}")
            return
        
        template = generator.SCENARIO_TEMPLATES[scenario_type]
        scenarios = template["scenarios"]
        
        if scenario_idx >= len(scenarios):
            print(f"❌ Scenario index {scenario_idx} out of range (max: {len(scenarios)-1})")
            return
        
        base_scenario = scenarios[scenario_idx]
        print(f"📝 Base scenario: {base_scenario}")
        
        # Now test the actual fetch_item
        print("🚀 Testing full fetch_item...")
        try:
            item = await generator.fetch_item(first_item_id)
            if item:
                print(f"✅ fetch_item succeeded!")
                print(f"   Content type: {type(item.content)}")
                print(f"   Content keys: {list(item.content.keys()) if isinstance(item.content, dict) else 'Not a dict'}")
            else:
                print(f"❌ fetch_item returned None")
        except Exception as e:
            print(f"❌ Exception in fetch_item: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test()) 