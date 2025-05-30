"""
Agentic Data Consolidator

Consolidates all collected agentic conversation data from multiple sources
into a unified training dataset with proper formatting and train/val/test splits.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random
from dataclasses import dataclass
import hashlib


@dataclass
class ConsolidationConfig:
    """Configuration for data consolidation"""
    # Source directories
    synthetic_dir: str = "data/conversations/synthetic"
    github_dir: str = "data/conversations/github_production/github"
    samples_dir: str = "data/sample_trajectories"
    coding_dir: str = "data/coding_problems"
    
    # Output configuration
    output_dir: str = "data/training_data"
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Quality filters
    min_turns: int = 2
    min_agentic_patterns: int = 1
    min_quality_score: float = 0.4
    
    # Deduplication
    enable_deduplication: bool = True
    similarity_threshold: float = 0.9


class AgenticDataConsolidator:
    """Consolidate and prepare agentic training data"""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
        self.logger = logging.getLogger("agentic_consolidator")
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_collected": 0,
            "source_counts": {},
            "quality_filtered": 0,
            "duplicates_removed": 0,
            "final_dataset_size": 0,
            "agentic_patterns": {},
            "complexity_distribution": {},
            "language_distribution": {}
        }
    
    def consolidate_all_data(self) -> Dict[str, Any]:
        """Main consolidation process"""
        
        self.logger.info("Starting agentic data consolidation")
        
        # Load data from all sources
        all_conversations = []
        
        # 1. Load synthetic data
        synthetic_conversations = self._load_synthetic_data()
        all_conversations.extend(synthetic_conversations)
        self.stats["source_counts"]["synthetic"] = len(synthetic_conversations)
        
        # 2. Load GitHub data  
        github_conversations = self._load_github_data()
        all_conversations.extend(github_conversations)
        self.stats["source_counts"]["github"] = len(github_conversations)
        
        # 3. Load sample trajectories
        sample_conversations = self._load_sample_trajectories()
        all_conversations.extend(sample_conversations)
        self.stats["source_counts"]["samples"] = len(sample_conversations)
        
        # 4. Load coding problems
        coding_conversations = self._load_coding_problems()
        all_conversations.extend(coding_conversations)
        self.stats["source_counts"]["coding"] = len(coding_conversations)
        
        self.stats["total_collected"] = len(all_conversations)
        self.logger.info(f"Loaded {len(all_conversations)} conversations from all sources")
        
        # 5. Quality filtering and validation
        filtered_conversations = self._apply_quality_filters(all_conversations)
        self.stats["quality_filtered"] = len(all_conversations) - len(filtered_conversations)
        
        # 6. Deduplication
        if self.config.enable_deduplication:
            deduplicated_conversations = self._remove_duplicates(filtered_conversations)
            self.stats["duplicates_removed"] = len(filtered_conversations) - len(deduplicated_conversations)
        else:
            deduplicated_conversations = filtered_conversations
        
        # 7. Analyze data statistics
        self._analyze_dataset_stats(deduplicated_conversations)
        
        # 8. Create train/val/test splits
        train_data, val_data, test_data = self._create_splits(deduplicated_conversations)
        
        # 9. Save consolidated datasets
        self._save_datasets(train_data, val_data, test_data)
        
        self.stats["final_dataset_size"] = len(deduplicated_conversations)
        
        # 10. Generate consolidation report
        report = self._generate_report(train_data, val_data, test_data)
        
        self.logger.info("Agentic data consolidation complete")
        return report
    
    def _load_synthetic_data(self) -> List[Dict[str, Any]]:
        """Load synthetic conversation data"""
        
        conversations = []
        synthetic_path = Path(self.config.synthetic_dir)
        
        if not synthetic_path.exists():
            self.logger.warning(f"Synthetic data directory not found: {synthetic_path}")
            return conversations
        
        # Load from JSONL files
        for jsonl_file in synthetic_path.glob("*.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        conversations.append({
                            "source": "synthetic",
                            "content": data.get("content", {}),
                            "quality_score": data.get("quality_score", 0.0),
                            "metadata": data.get("metadata", {})
                        })
        
        self.logger.info(f"Loaded {len(conversations)} synthetic conversations")
        return conversations
    
    def _load_github_data(self) -> List[Dict[str, Any]]:
        """Load GitHub conversation data"""
        
        conversations = []
        github_path = Path(self.config.github_dir)
        
        if not github_path.exists():
            self.logger.warning(f"GitHub data directory not found: {github_path}")
            return conversations
        
        # Load from JSONL files
        for jsonl_file in github_path.glob("*.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        conversations.append({
                            "source": "github",
                            "content": data.get("content", {}),
                            "quality_score": data.get("quality_score", 0.0),
                            "metadata": data.get("metadata", {})
                        })
        
        self.logger.info(f"Loaded {len(conversations)} GitHub conversations")
        return conversations
    
    def _load_sample_trajectories(self) -> List[Dict[str, Any]]:
        """Load sample trajectory data"""
        
        conversations = []
        samples_path = Path(self.config.samples_dir)
        
        if not samples_path.exists():
            self.logger.warning(f"Sample trajectories directory not found: {samples_path}")
            return conversations
        
        # Load JSONL files (they are actually JSONL, not JSON)
        for jsonl_file in samples_path.glob("*.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        
                        # Convert trajectory to conversation format
                        if "trajectory" in data and "steps" in data["trajectory"]:
                            conversations.append({
                                "source": "samples",
                                "content": {
                                    "conversation_type": "sample_trajectory",
                                    "title": data.get("problem", {}).get("title", "Sample"),
                                    "turns": self._convert_trajectory_to_turns(data["trajectory"]),
                                    "agentic_patterns": ["step_by_step", "problem_solving"],
                                    "domain": data.get("problem", {}).get("category", "general"),
                                    "complexity": data.get("problem", {}).get("difficulty", "medium"),
                                    "programming_languages": ["python"]  # Most sample trajectories are coding
                                },
                                "quality_score": 0.8,  # High quality samples
                                "metadata": {"trajectory_id": data.get("id")}
                            })
        
        self.logger.info(f"Loaded {len(conversations)} sample conversations")
        return conversations
    
    def _convert_trajectory_to_turns(self, trajectory: Dict) -> List[Dict]:
        """Convert trajectory steps to conversation turns"""
        
        turns = []
        current_turn = None
        
        for step in trajectory.get("steps", []):
            step_type = step.get("step_type", "")
            content = step.get("content", "")
            
            if step_type == "thought":
                # This is assistant reasoning
                if current_turn is None:
                    current_turn = {
                        "role": "assistant",
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    current_turn["content"] += "\n\n" + content
            else:
                # Any other step type, finish current turn and start new
                if current_turn:
                    turns.append(current_turn)
                    current_turn = None
                
                turns.append({
                    "role": "assistant",
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Add final turn if exists
        if current_turn:
            turns.append(current_turn)
        
        # If we only have assistant turns, add a user question at the beginning
        if turns and all(t["role"] == "assistant" for t in turns):
            user_turn = {
                "role": "user",
                "content": "Can you help me solve this problem step by step?",
                "timestamp": datetime.now().isoformat()
            }
            turns.insert(0, user_turn)
        
        return turns
    
    def _load_coding_problems(self) -> List[Dict[str, Any]]:
        """Load coding problem data"""
        
        conversations = []
        coding_path = Path(self.config.coding_dir)
        
        if not coding_path.exists():
            self.logger.warning(f"Coding problems directory not found: {coding_path}")
            return conversations
        
        # Load JSON files
        for json_file in coding_path.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Convert coding problem to conversation format
                conversations.append({
                    "source": "coding",
                    "content": {
                        "conversation_type": "coding_problem",
                        "title": data.get("title", "Coding Problem"),
                        "turns": [
                            {
                                "role": "user",
                                "content": data.get("problem_statement", ""),
                                "timestamp": datetime.now().isoformat()
                            },
                            {
                                "role": "assistant", 
                                "content": data.get("solution", ""),
                                "timestamp": datetime.now().isoformat()
                            }
                        ],
                        "agentic_patterns": ["problem_solving", "step_by_step"],
                        "domain": "coding",
                        "complexity": data.get("difficulty", "medium"),
                        "programming_languages": [data.get("language", "python")]
                    },
                    "quality_score": 0.7,  # Good quality coding problems
                    "metadata": {"problem_id": data.get("id")}
                })
        
        self.logger.info(f"Loaded {len(conversations)} coding conversations")
        return conversations
    
    def _apply_quality_filters(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply quality filters to conversations"""
        
        filtered = []
        
        for conv in conversations:
            content = conv.get("content", {})
            
            # Check minimum turns
            turns = content.get("turns", [])
            if len(turns) < self.config.min_turns:
                continue
            
            # Check agentic patterns
            agentic_patterns = content.get("agentic_patterns", [])
            if len(agentic_patterns) < self.config.min_agentic_patterns:
                continue
            
            # Check quality score
            quality_score = conv.get("quality_score", 0.0)
            if quality_score < self.config.min_quality_score:
                continue
            
            filtered.append(conv)
        
        self.logger.info(f"Quality filtering: {len(conversations)} -> {len(filtered)} conversations")
        return filtered
    
    def _remove_duplicates(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate conversations based on content similarity"""
        
        unique_conversations = []
        seen_hashes = set()
        
        for conv in conversations:
            # Create content hash for deduplication
            content_str = json.dumps(conv.get("content", {}), sort_keys=True)
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_conversations.append(conv)
        
        self.logger.info(f"Deduplication: {len(conversations)} -> {len(unique_conversations)} conversations")
        return unique_conversations
    
    def _analyze_dataset_stats(self, conversations: List[Dict[str, Any]]) -> None:
        """Analyze dataset statistics"""
        
        for conv in conversations:
            content = conv.get("content", {})
            
            # Track agentic patterns
            for pattern in content.get("agentic_patterns", []):
                self.stats["agentic_patterns"][pattern] = self.stats["agentic_patterns"].get(pattern, 0) + 1
            
            # Track complexity
            complexity = content.get("complexity", "unknown")
            self.stats["complexity_distribution"][complexity] = self.stats["complexity_distribution"].get(complexity, 0) + 1
            
            # Track languages
            for lang in content.get("programming_languages", []):
                self.stats["language_distribution"][lang] = self.stats["language_distribution"].get(lang, 0) + 1
    
    def _create_splits(self, conversations: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """Create train/validation/test splits"""
        
        # Shuffle conversations for random split
        random.seed(42)  # Reproducible splits
        shuffled = conversations.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_size = int(total * self.config.train_ratio)
        val_size = int(total * self.config.val_ratio)
        
        train_data = shuffled[:train_size]
        val_data = shuffled[train_size:train_size + val_size]
        test_data = shuffled[train_size + val_size:]
        
        self.logger.info(f"Created splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _save_datasets(self, train_data: List, val_data: List, test_data: List) -> None:
        """Save datasets to files"""
        
        output_path = Path(self.config.output_dir)
        
        # Save as JSONL files
        for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            output_file = output_path / f"agentic_{split_name}.jsonl"
            
            with open(output_file, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_conversations": len(train_data) + len(val_data) + len(test_data),
            "splits": {
                "train": len(train_data),
                "val": len(val_data), 
                "test": len(test_data)
            },
            "sources": self.stats["source_counts"],
            "config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "min_quality_score": self.config.min_quality_score
            }
        }
        
        with open(output_path / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved datasets to {output_path}")
    
    def _generate_report(self, train_data: List, val_data: List, test_data: List) -> Dict[str, Any]:
        """Generate consolidation report"""
        
        return {
            "consolidation_summary": {
                "total_collected": self.stats["total_collected"],
                "final_dataset_size": self.stats["final_dataset_size"],
                "quality_filtered": self.stats["quality_filtered"],
                "duplicates_removed": self.stats["duplicates_removed"]
            },
            "source_breakdown": self.stats["source_counts"],
            "dataset_splits": {
                "train": len(train_data),
                "validation": len(val_data),
                "test": len(test_data)
            },
            "agentic_patterns": self.stats["agentic_patterns"],
            "complexity_distribution": self.stats["complexity_distribution"],
            "language_distribution": self.stats["language_distribution"],
            "output_directory": self.config.output_dir
        }


def consolidate_agentic_data(config: ConsolidationConfig = None) -> Dict[str, Any]:
    """Main function to consolidate all agentic training data"""
    
    if config is None:
        config = ConsolidationConfig()
    
    consolidator = AgenticDataConsolidator(config)
    return consolidator.consolidate_all_data() 