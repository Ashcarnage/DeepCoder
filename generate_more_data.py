#!/usr/bin/env python3
"""
Generate more synthetic agentic training data
"""
import asyncio
import os
from data.collection.synthetic_generator import SyntheticConfig, collect_synthetic_data
from data.collection.base_collector import CollectionConfig

async def generate_more_data():
    # Load environment
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except:
        pass
    
    # Create config with ALL scenarios for maximum diversity
    collection_config = CollectionConfig(
        output_dir='data/collected',
        quality_threshold=0.4,  # Lower threshold for more data
        max_concurrent_requests=2,  # Very conservative to avoid rate limits
        rate_limit_per_second=0.5,  # Very slow rate limiting
        retry_attempts=2,
        retry_delay=3.0
    )
    
    synthetic_config = SyntheticConfig(
        provider='groq',
        api_key=os.environ['GROQ_API_KEY'],
        model_name='llama3-8b-8192',
        scenario_types=[
            'tool_usage', 'multi_step_reasoning', 'error_handling', 'code_debugging',
            'data_analysis', 'web_research', 'api_integration', 'problem_solving'
        ],  # ALL scenario types
        scenarios_per_batch=2,
        max_turns_per_conversation=3,  # Keep it efficient
        temperature=0.6,
        max_tokens=1200,  # Even smaller tokens per response
        require_tool_usage=True,
        require_reasoning=True
    )
    
    print('🚀 Generating final batch with ALL scenario types...')
    print(f'📝 Scenarios: {len(synthetic_config.scenario_types)} types')
    print(f'⚙️ Settings: {synthetic_config.max_turns_per_conversation} turns, {synthetic_config.max_tokens} max tokens')
    print(f'🐌 Rate limit: {collection_config.rate_limit_per_second} requests/sec')
    
    result = await collect_synthetic_data(collection_config, synthetic_config, max_items=15)
    
    if 'metrics' in result:
        print(f'✅ Generated {result["metrics"].total_collected} new conversations')
        print(f'📊 Success rate: {result["metrics"].total_collected}/{result["metrics"].total_requested}')
    else:
        print(f'📋 Result: {result}')

if __name__ == "__main__":
    asyncio.run(generate_more_data()) 