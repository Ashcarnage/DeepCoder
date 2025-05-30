#!/usr/bin/env python3
"""
Test script for GitHub Scraper - collect real-world agentic tool usage examples
"""

import asyncio
import os
import json
from datetime import datetime
from data.collection.base_collector import CollectionConfig
from data.collection.github_scraper import GitHubScraper, GitHubConfig, collect_github_data


async def test_github_scraper():
    """Test the GitHub scraper functionality"""
    
    print("🔍 Testing GitHub Scraper for Agentic Tool Usage Examples")
    print("=" * 60)
    
    # Collection configuration
    collection_config = CollectionConfig(
        output_dir="data/conversations/github",
        max_concurrent_requests=3,  # Conservative for GitHub API
        rate_limit_per_second=0.5,  # 0.5 requests per second (2 second delay)
        retry_attempts=3,
        retry_delay=2.0,
        timeout_seconds=30,
        checkpoint_interval=10,
        quality_threshold=0.4,  # Higher threshold for quality
        enable_deduplication=True,
        log_level="INFO"
    )
    
    # GitHub configuration
    github_config = GitHubConfig(
        api_token=os.getenv("GITHUB_TOKEN"),  # Optional: set for higher rate limits
        max_items_per_query=50,  # Reasonable limit
        max_age_days=60,  # Last 60 days
        min_content_length=300,  # Require substantial content
        require_code_examples=False,  # Allow non-code discussions too
        require_tool_mentions=True   # Must mention tools
    )
    
    print(f"📊 Configuration:")
    print(f"   • API Token: {'✓ Set' if github_config.api_token else '✗ Not set (rate limited)'}")
    print(f"   • Max items per query: {github_config.max_items_per_query}")
    print(f"   • Content age limit: {github_config.max_age_days} days")
    print(f"   • Min content length: {github_config.min_content_length} chars")
    print(f"   • Quality threshold: {collection_config.quality_threshold}")
    print(f"   • Rate limit: {collection_config.rate_limit_per_second} req/sec")
    print(f"   • Max concurrent: {collection_config.max_concurrent_requests}")
    print()
    
    print("🔍 Target Search Queries:")
    for i, query in enumerate(github_config.search_queries[:5], 1):
        print(f"   {i}. {query}")
    print(f"   ... and {len(github_config.search_queries) - 5} more")
    print()
    
    print("📂 Target Repositories:")
    for i, repo in enumerate(github_config.repositories[:5], 1):
        print(f"   {i}. {repo}")
    print(f"   ... and {len(github_config.repositories) - 5} more")
    print()
    
    # Test GitHub API connectivity
    print("🔌 Testing GitHub API connectivity...")
    scraper = GitHubScraper(collection_config, github_config)
    
    async with scraper:
        # Test a simple API call
        try:
            url = "https://api.github.com/rate_limit"
            async with scraper.session.get(url, headers=scraper.headers) as response:
                if response.status == 200:
                    rate_limit_info = await response.json()
                    core_limit = rate_limit_info.get("resources", {}).get("core", {})
                    search_limit = rate_limit_info.get("resources", {}).get("search", {})
                    
                    print("✅ GitHub API connectivity successful!")
                    print(f"   • Core API: {core_limit.get('remaining', 0)}/{core_limit.get('limit', 0)} requests remaining")
                    print(f"   • Search API: {search_limit.get('remaining', 0)}/{search_limit.get('limit', 0)} requests remaining")
                    print()
                else:
                    print(f"❌ GitHub API error: {response.status}")
                    return
        except Exception as e:
            print(f"❌ GitHub API connection failed: {e}")
            return
    
    # Collect sample data
    print("📥 Starting GitHub data collection...")
    print("   Targeting: AI coding assistants, development workflows, tool usage patterns")
    print()
    
    try:
        # Collect a limited set for testing
        result = await collect_github_data(
            collection_config,
            github_config,
            max_items=20  # Start with 20 items for testing
        )
        
        print("\n📊 Collection Results:")
        print("=" * 40)
        
        metrics = result["metrics"]
        print(f"Items attempted: {metrics.total_requested}")
        print(f"Items collected: {metrics.total_collected}")
        print(f"Items filtered: {metrics.total_filtered}")
        print(f"Items duplicates: {metrics.total_duplicates}")
        print(f"Items errors: {metrics.total_errors}")
        print(f"Success rate: {metrics.success_rate:.1%}")
        print(f"Collection rate: {metrics.collection_rate:.2f} items/sec")
        
        if metrics.start_time:
            duration = (datetime.now() - metrics.start_time).total_seconds()
            print(f"Duration: {duration:.1f}s")
        print()
        
        # Show sample of collected data
        output_dir = result["output_directory"]
        print(f"📁 Output directory: {output_dir}")
        
        # Look for the JSON files
        from pathlib import Path
        
        output_path = Path(output_dir)
        if output_path.exists():
            jsonl_files = list(output_path.glob("*.jsonl"))
            print(f"📄 Generated {len(jsonl_files)} conversation files")
            
            if jsonl_files:
                # Show sample conversation from JSONL file
                sample_file = jsonl_files[0]
                print(f"\n🔍 Sample conversations from: {sample_file.name}")
                print("-" * 50)
                
                # Read first few lines from JSONL
                with open(sample_file, 'r') as f:
                    conversations = []
                    for i, line in enumerate(f):
                        if i >= 3:  # Show first 3 conversations
                            break
                        conversations.append(json.loads(line.strip()))
                
                for i, conv_data in enumerate(conversations, 1):
                    sample_conv = conv_data.get("content", {})
                    print(f"\nConversation {i}:")
                    print(f"Type: {sample_conv.get('conversation_type', 'unknown')}")
                    print(f"Title: {sample_conv.get('title', 'No title')[:80]}...")
                    print(f"Repository: {sample_conv.get('repository', 'unknown')}")
                    print(f"Complexity: {sample_conv.get('complexity', 'unknown')}")
                    print(f"Languages: {', '.join(sample_conv.get('programming_languages', []))}")
                    print(f"Agentic patterns: {', '.join(sample_conv.get('agentic_patterns', []))}")
                    print(f"Turns: {len(sample_conv.get('turns', []))}")
                    print(f"Quality score: {conv_data.get('quality_score', 0):.3f}")
                    
                    # Show first turn content (truncated)
                    turns = sample_conv.get("turns", [])
                    if turns:
                        first_turn = turns[0].get("content", "")
                        print(f"First turn preview: '{first_turn[:150]}{'...' if len(first_turn) > 150 else ''}'")
                
                print("-" * 50)
        
        print("\n✅ GitHub scraper test completed successfully!")
        print(f"🎯 Focus areas detected: AI assistants, development workflows, tool usage")
        print(f"📈 Ready for production data collection")
        
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_github_scraper()) 