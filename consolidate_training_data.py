#!/usr/bin/env python3
"""
Consolidate Agentic Training Data

Runs the data consolidation process to prepare all collected agentic conversations
for training the student model with SAD loss.
"""

import asyncio
import logging
import sys
from pathlib import Path

from data.processing.agentic_consolidator import consolidate_agentic_data, ConsolidationConfig


def setup_logging():
    """Setup logging for consolidation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('consolidation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main consolidation process"""
    
    setup_logging()
    logger = logging.getLogger("consolidation_main")
    
    print("🔧 Agentic Training Data Consolidation")
    print("=" * 50)
    
    # Configuration for consolidation
    config = ConsolidationConfig(
        # Source directories (correct paths based on find results)
        synthetic_dir="data/collected/synthetic",
        github_dir="data/conversations/github_production/github", 
        samples_dir="data/sample_trajectories",
        coding_dir="data/coding_problems",
        
        # Output
        output_dir="data/training_data",
        
        # Quality filters
        min_turns=2,
        min_agentic_patterns=1,
        min_quality_score=0.4,
        
        # Splits (80/10/10)
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        
        # Deduplication
        enable_deduplication=True
    )
    
    print(f"📊 Consolidation Configuration:")
    print(f"   • Output directory: {config.output_dir}")
    print(f"   • Quality threshold: {config.min_quality_score}")
    print(f"   • Minimum turns: {config.min_turns}")
    print(f"   • Split ratios: {config.train_ratio:.1f}/{config.val_ratio:.1f}/{config.test_ratio:.1f}")
    print(f"   • Deduplication: {'Enabled' if config.enable_deduplication else 'Disabled'}")
    print()
    
    # Check source directories
    print("📁 Checking source directories:")
    sources = [
        ("Synthetic", config.synthetic_dir),
        ("GitHub", config.github_dir), 
        ("Samples", config.samples_dir),
        ("Coding", config.coding_dir)
    ]
    
    for name, path in sources:
        path_obj = Path(path)
        if path_obj.exists():
            files = list(path_obj.glob("*.*"))
            print(f"   ✅ {name}: {len(files)} files in {path}")
        else:
            print(f"   ⚠️  {name}: Directory not found - {path}")
    print()
    
    try:
        print("🚀 Starting data consolidation...")
        
        # Run consolidation
        report = consolidate_agentic_data(config)
        
        print("\n✅ Consolidation Complete!")
        print("=" * 40)
        
        # Display results
        summary = report["consolidation_summary"]
        print(f"📊 Summary:")
        print(f"   • Total collected: {summary['total_collected']}")
        print(f"   • Quality filtered: {summary['quality_filtered']}")
        print(f"   • Duplicates removed: {summary['duplicates_removed']}")
        print(f"   • Final dataset size: {summary['final_dataset_size']}")
        print()
        
        print(f"📁 Source Breakdown:")
        for source, count in report["source_breakdown"].items():
            print(f"   • {source.capitalize()}: {count}")
        print()
        
        print(f"📈 Dataset Splits:")
        splits = report["dataset_splits"]
        print(f"   • Training: {splits['train']}")
        print(f"   • Validation: {splits['validation']}")
        print(f"   • Test: {splits['test']}")
        print()
        
        print(f"🧠 Agentic Patterns Detected:")
        patterns = report["agentic_patterns"]
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {pattern}: {count}")
        print()
        
        print(f"📊 Complexity Distribution:")
        complexity = report["complexity_distribution"]
        for level, count in complexity.items():
            print(f"   • {level}: {count}")
        print()
        
        if report["language_distribution"]:
            print(f"💻 Programming Languages (top 8):")
            languages = sorted(report["language_distribution"].items(), key=lambda x: x[1], reverse=True)[:8]
            for lang, count in languages:
                print(f"   • {lang}: {count}")
            print()
        
        output_dir = Path(report["output_directory"])
        print(f"📁 Training Data Location: {output_dir}")
        print(f"   📄 Files created:")
        for file in ["agentic_train.jsonl", "agentic_val.jsonl", "agentic_test.jsonl", "dataset_metadata.json"]:
            file_path = output_dir / file
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"      • {file} ({size_kb:.1f} KB)")
        
        print(f"\n🎯 Ready for Agentic Training!")
        print(f"   • Use {output_dir}/agentic_train.jsonl for training")
        print(f"   • Use {output_dir}/agentic_val.jsonl for validation")
        print(f"   • Use {output_dir}/agentic_test.jsonl for final evaluation")
        print(f"   • Dataset metadata in {output_dir}/dataset_metadata.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 DeepCoder Agentic Data Consolidation")
    print("Preparing training data for SAD student model")
    print()
    
    success = main()
    
    if success:
        print("\n✨ All ready for Phase 4 - Enhanced Training Pipeline!")
        sys.exit(0)
    else:
        print("\n❌ Consolidation failed. Check logs for details.")
        sys.exit(1) 