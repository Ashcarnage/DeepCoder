#!/usr/bin/env python3
"""
Production GitHub Agentic Data Collection Script

Collects real-world examples of agentic tool usage from GitHub repositories,
issues, and discussions. Focuses on AI coding assistants, development workflows,
and tool-calling patterns similar to Cursor and other AI development tools.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
import logging

from data.collection.base_collector import CollectionConfig
from data.collection.github_scraper import GitHubScraper, GitHubConfig, collect_github_data


def setup_logging():
    """Setup production logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('github_collection.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


async def collect_production_github_data():
    """Collect production-scale GitHub agentic data"""
    
    setup_logging()
    logger = logging.getLogger("github_production")
    
    print("🚀 Production GitHub Agentic Data Collection")
    print("=" * 50)
    
    # Production collection configuration
    collection_config = CollectionConfig(
        output_dir="data/conversations/github_production",
        max_concurrent_requests=2,  # Conservative for API limits
        rate_limit_per_second=0.3,  # 3.3 second delay between requests
        retry_attempts=3,
        retry_delay=3.0,
        timeout_seconds=45,
        checkpoint_interval=5,       # Save progress frequently
        quality_threshold=0.5,       # Higher quality threshold
        enable_deduplication=True,
        log_level="INFO"
    )
    
    # Enhanced GitHub configuration for production
    github_config = GitHubConfig(
        api_token=os.getenv("GITHUB_TOKEN"),
        max_items_per_query=30,  # Reasonable limit per query
        max_age_days=120,        # Look back 4 months
        min_content_length=400,  # Require substantial content
        require_code_examples=False,
        require_tool_mentions=True
    )
    
    # Enhanced search queries for agentic patterns
    github_config.search_queries = [
        # AI Coding Assistants & Workflows
        "cursor AI assistant workflow",
        "github copilot integration tutorial", 
        "AI pair programming setup",
        "code assistant configuration",
        "automated coding workflow setup",
        "AI code completion integration",
        "intelligent code suggestions setup",
        
        # Development Tool Integration
        "API integration step by step tutorial",
        "debugging workflow with tools guide",
        "automated testing setup tutorial",
        "CI/CD pipeline configuration",
        "development environment setup",
        "IDE integration tutorial",
        "linter configuration guide",
        
        # Advanced Agentic Patterns
        "multi-step debugging process",
        "error handling workflow automation",
        "code review automation setup",
        "deployment pipeline automation",
        "monitoring setup tutorial",
        "performance optimization workflow",
        
        # Tool Usage Examples
        "VSCode extension development tutorial",
        "development tools integration",
        "command line automation tutorial",
        "script automation workflow",
        "build system configuration"
    ]
    
    # Enhanced repository targets
    github_config.repositories = [
        # AI Development Tools
        "microsoft/vscode",
        "github/copilot-docs",
        "cursor-ai/cursor", 
        "sourcegraph/sourcegraph",
        "tabnine/tabnine-vscode",
        "codeium/codeium",
        
        # Major Development Frameworks
        "microsoft/TypeScript",
        "facebook/react",
        "python/cpython",
        "rust-lang/rust", 
        "golang/go",
        "nodejs/node",
        
        # Development Tools & Automation
        "docker/docker",
        "kubernetes/kubernetes",
        "hashicorp/terraform",
        "ansible/ansible",
        "jenkins-x/jx",
        "argoproj/argo-cd",
        
        # Popular Open Source Projects
        "tensorflow/tensorflow",
        "pytorch/pytorch",
        "huggingface/transformers",
        "openai/gpt-4",
        "anthropic/claude-api"
    ]
    
    print(f"📊 Production Configuration:")
    print(f"   • API Token: {'✓ Set' if github_config.api_token else '✗ Not set (rate limited)'}")
    print(f"   • Search queries: {len(github_config.search_queries)}")
    print(f"   • Target repositories: {len(github_config.repositories)}")
    print(f"   • Max age: {github_config.max_age_days} days")
    print(f"   • Min content length: {github_config.min_content_length} chars")
    print(f"   • Quality threshold: {collection_config.quality_threshold}")
    print(f"   • Rate limit: {collection_config.rate_limit_per_second} req/sec")
    print()
    
    # Calculate total potential items
    total_search_items = len(github_config.search_queries) * 2  # issues + discussions
    total_repo_items = len(github_config.repositories) * 2      # issues + PRs
    total_potential = total_search_items + total_repo_items
    
    print(f"🎯 Collection Scope:")
    print(f"   • Search-based items: {total_search_items}")
    print(f"   • Repository-based items: {total_repo_items}")
    print(f"   • Total potential items: {total_potential}")
    print()
    
    # Confirm before starting
    if github_config.api_token:
        print("⚡ GitHub token detected - higher rate limits available")
    else:
        print("⚠️  No GitHub token - limited to 60 requests/hour")
        print("   Set GITHUB_TOKEN environment variable for production use")
    
    print("🚀 Starting production data collection...")
    print("   Focus: Real-world agentic tool usage patterns")
    print("   Target: AI assistants, development workflows, automation")
    print()
    
    start_time = datetime.now()
    
    try:
        # Collect production data set
        result = await collect_github_data(
            collection_config,
            github_config,
            max_items=100  # Collect up to 100 high-quality items
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n🎉 Production Collection Complete!")
        print("=" * 40)
        
        metrics = result["metrics"]
        print(f"📊 Final Metrics:")
        print(f"   • Items attempted: {metrics.total_requested}")
        print(f"   • Items collected: {metrics.total_collected}")
        print(f"   • Items filtered (low quality): {metrics.total_filtered}")
        print(f"   • Duplicates removed: {metrics.total_duplicates}")
        print(f"   • Errors encountered: {metrics.total_errors}")
        print(f"   • Success rate: {metrics.success_rate:.1%}")
        print(f"   • Collection rate: {metrics.collection_rate:.2f} items/sec")
        print(f"   • Total duration: {duration.total_seconds():.1f}s")
        print()
        
        # Analyze collected data
        output_dir = Path(result["output_directory"])
        if output_dir.exists():
            jsonl_files = list(output_dir.glob("*.jsonl"))
            
            total_conversations = 0
            agentic_patterns_stats = {}
            domain_stats = {}
            complexity_stats = {}
            language_stats = {}
            
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            import json
                            conv_data = json.loads(line.strip())
                            content = conv_data.get("content", {})
                            total_conversations += 1
                            
                            # Track agentic patterns
                            for pattern in content.get("agentic_patterns", []):
                                agentic_patterns_stats[pattern] = agentic_patterns_stats.get(pattern, 0) + 1
                            
                            # Track domains
                            domain = content.get("domain", "unknown")
                            domain_stats[domain] = domain_stats.get(domain, 0) + 1
                            
                            # Track complexity
                            complexity = content.get("complexity", "unknown")
                            complexity_stats[complexity] = complexity_stats.get(complexity, 0) + 1
                            
                            # Track languages
                            for lang in content.get("programming_languages", []):
                                language_stats[lang] = language_stats.get(lang, 0) + 1
            
            print(f"📈 Data Analysis:")
            print(f"   • Total conversations: {total_conversations}")
            print(f"   • JSONL files: {len(jsonl_files)}")
            print()
            
            if agentic_patterns_stats:
                print("🧠 Agentic Patterns Detected:")
                for pattern, count in sorted(agentic_patterns_stats.items(), key=lambda x: x[1], reverse=True):
                    print(f"   • {pattern}: {count}")
                print()
            
            if complexity_stats:
                print("📊 Conversation Complexity:")
                for complexity, count in complexity_stats.items():
                    print(f"   • {complexity}: {count}")
                print()
            
            if language_stats:
                print("💻 Programming Languages (top 10):")
                top_languages = sorted(language_stats.items(), key=lambda x: x[1], reverse=True)[:10]
                for lang, count in top_languages:
                    print(f"   • {lang}: {count}")
                print()
            
            print(f"📁 Output location: {output_dir}")
            print(f"🎯 High-quality agentic training data ready!")
            
            # Calculate training data metrics
            if total_conversations > 0:
                avg_patterns = sum(len(content.get("agentic_patterns", [])) for content in [json.loads(line)["content"] for line in open(jsonl_files[0]).readlines() if line.strip()]) / total_conversations
                print(f"📊 Training Quality Metrics:")
                print(f"   • Average agentic patterns per conversation: {avg_patterns:.1f}")
                print(f"   • Unique agentic patterns: {len(agentic_patterns_stats)}")
                print(f"   • Multi-language conversations: {len([l for l in language_stats.values() if l > 1])}")
        
        return result
        
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
        print("   Progress has been saved to checkpoint")
        return None
        
    except Exception as e:
        logger.error(f"Production collection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🔧 GitHub Agentic Data Collector")
    print("Collecting real-world tool usage patterns for agentic training")
    print()
    
    # Run production collection
    asyncio.run(collect_production_github_data()) 