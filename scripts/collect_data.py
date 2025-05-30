#!/usr/bin/env python3
"""
Data Collection Orchestration Script

This script orchestrates the collection of agentic training data from multiple sources
including HuggingFace datasets, web scraping, and synthetic generation.
"""

import asyncio
import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from data.collection.base_collector import CollectionConfig, create_collector_config
    from data.sources.huggingface_datasets import collect_huggingface_datasets
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run this script from the project root directory")
    sys.exit(1)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'data_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


async def collect_huggingface_data(
    config: CollectionConfig,
    datasets: Optional[List[str]] = None,
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """Collect data from HuggingFace datasets"""
    logger = logging.getLogger("collector.orchestrator")
    logger.info("Starting HuggingFace data collection...")
    
    try:
        result = await collect_huggingface_datasets(config, datasets, max_items)
        logger.info(f"HuggingFace collection completed: {result['metrics'].total_collected} items")
        return result
    except Exception as e:
        logger.error(f"HuggingFace collection failed: {e}")
        return {'source': 'huggingface', 'error': str(e)}


async def collect_synthetic_data(
    config: CollectionConfig,
    scenarios: Optional[List[str]] = None,
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """Collect synthetic agentic data"""
    logger = logging.getLogger("collector.orchestrator")
    logger.info("Starting synthetic data collection...")
    
    try:
        # Import synthetic generator
        from data.collection.synthetic_generator import collect_synthetic_data as collect_synthetic, SyntheticConfig
        
        # Get API configuration from environment or config
        groq_api_key = os.getenv('GROQ_API_KEY')
        together_api_key = os.getenv('TOGETHER_API_KEY')
        hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        # Choose provider based on available API keys
        if groq_api_key:
            provider = "groq"
            api_key = groq_api_key
            model_name = "llama3-8b-8192"
            logger.info("Using Groq API for synthetic generation")
        elif together_api_key:
            provider = "together"
            api_key = together_api_key
            model_name = "meta-llama/Llama-3-8b-chat-hf"
            logger.info("Using Together AI for synthetic generation")
        elif hf_api_key:
            provider = "huggingface"
            api_key = hf_api_key
            model_name = "microsoft/DialoGPT-medium"
            logger.info("Using HuggingFace API for synthetic generation")
        else:
            logger.warning("No API keys found for synthetic generation. Set GROQ_API_KEY, TOGETHER_API_KEY, or HUGGINGFACE_API_KEY")
            return {
                'source': 'synthetic',
                'status': 'no_api_key',
                'message': 'No API keys available. Please set GROQ_API_KEY, TOGETHER_API_KEY, or HUGGINGFACE_API_KEY environment variable.'
            }
        
        # Create synthetic config
        synthetic_config = SyntheticConfig(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            scenario_types=scenarios or [
                "tool_usage",
                "multi_step_reasoning", 
                "error_handling",
                "code_debugging"
            ],
            max_turns_per_conversation=6,
            temperature=0.8,
            require_tool_usage=True,
            require_reasoning=True
        )
        
        logger.info(f"Synthetic config: provider={provider}, model={model_name}, scenarios={len(synthetic_config.scenario_types)}")
        
        # Run synthetic collection
        result = await collect_synthetic(config, synthetic_config, max_items)
        logger.info(f"Synthetic collection completed: {result['metrics'].total_collected} items")
        return result
        
    except ImportError as e:
        logger.error(f"Failed to import synthetic generator: {e}")
        return {
            'source': 'synthetic',
            'status': 'import_error',
            'message': f'Import error: {e}'
        }
    except Exception as e:
        logger.error(f"Synthetic collection failed: {e}")
        return {
            'source': 'synthetic', 
            'error': str(e),
            'status': 'failed'
        }


async def collect_web_data(
    config: CollectionConfig,
    sources: Optional[List[str]] = None,
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """Collect data from web sources (placeholder)"""
    logger = logging.getLogger("collector.orchestrator")
    logger.info("Web data collection not yet implemented")
    
    return {
        'source': 'web',
        'status': 'not_implemented', 
        'message': 'Web scraping will be implemented in future tasks'
    }


async def main_collection_pipeline(
    config_path: str,
    sources: List[str],
    datasets: Optional[List[str]] = None,
    max_items_per_source: Optional[int] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Main data collection pipeline"""
    
    # Load configuration
    config = create_collector_config(config_path)
    
    # Setup logging
    setup_logging(config.log_level)
    logger = logging.getLogger("collector.orchestrator")
    
    logger.info(f"Starting data collection pipeline with sources: {sources}")
    logger.info(f"Configuration: {config}")
    
    if dry_run:
        logger.info("DRY RUN MODE - No data will be collected")
        return {
            'dry_run': True,
            'config': config,
            'sources': sources,
            'datasets': datasets,
            'max_items_per_source': max_items_per_source
        }
    
    # Collection results
    results = {
        'start_time': datetime.now().isoformat(),
        'config': config,
        'sources': sources,
        'results': {}
    }
    
    # Collect from each source
    collection_tasks = []
    
    if 'huggingface' in sources:
        task = collect_huggingface_data(config, datasets, max_items_per_source)
        collection_tasks.append(('huggingface', task))
    
    if 'synthetic' in sources:
        task = collect_synthetic_data(config, datasets, max_items_per_source)
        collection_tasks.append(('synthetic', task))
    
    if 'web' in sources:
        task = collect_web_data(config, datasets, max_items_per_source)
        collection_tasks.append(('web', task))
    
    # Run collections concurrently
    logger.info(f"Running {len(collection_tasks)} collection tasks...")
    
    for source_name, task in collection_tasks:
        try:
            result = await task
            results['results'][source_name] = result
            logger.info(f"Completed collection for {source_name}")
        except Exception as e:
            logger.error(f"Failed collection for {source_name}: {e}")
            results['results'][source_name] = {
                'source': source_name,
                'error': str(e),
                'status': 'failed'
            }
    
    # Summary
    results['end_time'] = datetime.now().isoformat()
    results['summary'] = generate_collection_summary(results)
    
    # Save results
    output_file = Path(config.output_dir) / f"collection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Collection results saved to: {output_file}")
    return results


def generate_collection_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of collection results"""
    
    summary = {
        'total_sources': len(results['results']),
        'successful_sources': 0,
        'failed_sources': 0,
        'total_items_collected': 0,
        'sources_status': {}
    }
    
    for source, result in results['results'].items():
        if 'error' in result:
            summary['failed_sources'] += 1
            summary['sources_status'][source] = 'failed'
        else:
            summary['successful_sources'] += 1
            summary['sources_status'][source] = 'success'
            
            # Extract metrics if available
            if 'metrics' in result and hasattr(result['metrics'], 'total_collected'):
                summary['total_items_collected'] += result['metrics'].total_collected
    
    return summary


def create_default_config() -> None:
    """Create default configuration file"""
    
    config_content = {
        'data_collection': {
            'output_dir': 'data/collected',
            'max_concurrent_requests': 10,
            'rate_limit_per_second': 2.0,
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'timeout_seconds': 30,
            'checkpoint_interval': 100,
            'quality_threshold': 0.5,
            'enable_deduplication': True,
            'log_level': 'INFO'
        },
        'huggingface_datasets': {
            'default_datasets': [
                'toolbench',
                'qwen_agent', 
                'react',
                'hotpotqa'
            ],
            'max_items_per_dataset': 1000
        },
        'synthetic_generation': {
            'preferred_provider': 'groq',  # groq, together, huggingface
            'default_scenarios': [
                'tool_usage',
                'multi_step_reasoning',
                'error_handling',
                'code_debugging',
                'data_analysis',
                'api_integration'
            ],
            'max_items_per_scenario': 100,
            'max_turns_per_conversation': 6,
            'temperature': 0.8,
            'require_tool_usage': True,
            'require_reasoning': True,
            'api_providers': {
                'groq': {
                    'model_name': 'llama3-8b-8192',
                    'rate_limit': 30.0,  # requests per minute
                    'free_tier_info': 'Free tier available with rate limits'
                },
                'together': {
                    'model_name': 'meta-llama/Llama-3-8b-chat-hf',
                    'rate_limit': 60.0,
                    'free_tier_info': '$1 free credit for new users'
                },
                'huggingface': {
                    'model_name': 'microsoft/DialoGPT-medium',
                    'rate_limit': 10.0,
                    'free_tier_info': 'Free tier with usage limits'
                }
            }
        },
        'web_scraping': {
            'github_repos': [
                'microsoft/semantic-kernel',
                'langchain-ai/langchain',
                'openai/swarm'
            ],
            'rate_limit_github': 1.0
        },
        'api_keys': {
            'note': 'Set these as environment variables: GROQ_API_KEY, TOGETHER_API_KEY, HUGGINGFACE_API_KEY',
            'groq_signup': 'https://console.groq.com/',
            'together_signup': 'https://api.together.xyz/',
            'huggingface_signup': 'https://huggingface.co/settings/tokens'
        }
    }
    
    config_path = Path('configs/data_collection.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config_content, f, indent=2)
    
    print(f"Created default configuration at: {config_path}")
    print("\n🚀 SETUP INSTRUCTIONS:")
    print("=" * 50)
    print("1. Get a FREE API key from one of these providers:")
    print("   • Groq (Recommended): https://console.groq.com/")
    print("   • Together AI: https://api.together.xyz/")
    print("   • HuggingFace: https://huggingface.co/settings/tokens")
    print("\n2. Set your API key as an environment variable:")
    print("   export GROQ_API_KEY='your_api_key_here'")
    print("   # OR")
    print("   export TOGETHER_API_KEY='your_api_key_here'")
    print("   # OR") 
    print("   export HUGGINGFACE_API_KEY='your_api_key_here'")
    print("\n3. Run data collection:")
    print("   python scripts/collect_data.py --sources synthetic --max-items 50")
    print("\n💡 Groq offers the fastest inference with generous free limits!")
    print("💡 All providers listed have free tiers suitable for this project.")


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Collect agentic training data from multiple sources")
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_collection.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--sources',
        nargs='+',
        default=['huggingface'],
        choices=['huggingface', 'synthetic', 'web'],
        help='Data sources to collect from'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to collect (for HuggingFace source)'
    )
    
    parser.add_argument(
        '--max-items',
        type=int,
        help='Maximum items to collect per source'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be collected without actually collecting'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file and exit'
    )
    
    parser.add_argument(
        '--validate-setup',
        action='store_true',
        help='Validate collection setup and requirements'
    )
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        create_default_config()
        return
    
    # Validate setup if requested
    if args.validate_setup:
        await validate_collection_setup(args.config)
        return
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        print("Create one with: python scripts/collect_data.py --create-config")
        return
    
    # Run collection pipeline
    try:
        results = await main_collection_pipeline(
            config_path=args.config,
            sources=args.sources,
            datasets=args.datasets,
            max_items_per_source=args.max_items,
            dry_run=args.dry_run
        )
        
        # Print summary
        if not args.dry_run:
            summary = results['summary']
            print("\n" + "="*50)
            print("COLLECTION SUMMARY")
            print("="*50)
            print(f"Total Sources: {summary['total_sources']}")
            print(f"Successful: {summary['successful_sources']}")
            print(f"Failed: {summary['failed_sources']}")
            print(f"Total Items Collected: {summary['total_items_collected']}")
            print("\nSource Status:")
            for source, status in summary['sources_status'].items():
                print(f"  {source}: {status}")
        else:
            print("\nDry run completed - no data collected")
            
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"Collection failed with error: {e}")
        sys.exit(1)


async def validate_collection_setup(config_path: str) -> None:
    """Validate collection setup and requirements"""
    
    print("Validating collection setup...")
    
    # Check config file
    if not Path(config_path).exists():
        print(f"❌ Configuration file not found: {config_path}")
        return
    else:
        print(f"✅ Configuration file found: {config_path}")
    
    # Check required packages
    required_packages = [
        'datasets',
        'huggingface_hub', 
        'aiohttp',
        'aiofiles',
        'tqdm',
        'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ Package {package} available")
        except ImportError:
            print(f"❌ Package {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return
    
    # Check output directory
    try:
        config = create_collector_config(config_path)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory accessible: {output_dir}")
    except Exception as e:
        print(f"❌ Output directory issue: {e}")
        return
    
    # Test HuggingFace access
    try:
        from datasets import load_dataset
        # Try to load a small dataset
        test_dataset = load_dataset("glue", "cola", split="train[:10]")
        print("✅ HuggingFace datasets accessible")
    except Exception as e:
        print(f"❌ HuggingFace access issue: {e}")
        return
    
    print("\n✅ All validation checks passed!")
    print("Ready to collect agentic training data.")


if __name__ == "__main__":
    asyncio.run(main()) 