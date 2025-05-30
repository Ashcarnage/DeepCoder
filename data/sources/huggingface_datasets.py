"""
HuggingFace Datasets Collector for Agentic Training Data

This module collects high-quality agentic datasets from HuggingFace Hub,
focusing on tool use, function calling, and agent interaction patterns.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

from data.collection.base_collector import BaseCollector, DataItem, CollectionConfig


class HuggingFaceCollector(BaseCollector):
    """Collector for HuggingFace agentic datasets"""
    
    # High-quality agentic datasets
    AGENTIC_DATASETS = {
        'toolbench': {
            'path': 'ShishirPatil/gorilla-openfunctions-v1',
            'config': None,
            'split': 'train',
            'description': 'Function calling examples with API documentation',
            'quality_indicators': ['function_call', 'api_usage', 'tool_selection']
        },
        'qwen_agent': {
            'path': 'qwenlm/qwen-agent-data', 
            'config': None,
            'split': 'train',
            'description': 'Tool interaction examples from Qwen team',
            'quality_indicators': ['tool_use', 'reasoning', 'execution']
        },
        'webarena': {
            'path': 'webarena/webarena',
            'config': None, 
            'split': 'test',
            'description': 'Web browsing and interaction tasks',
            'quality_indicators': ['web_navigation', 'multi_step', 'task_completion']
        },
        'agentbench': {
            'path': 'THUDM/AgentBench',
            'config': None,
            'split': 'test', 
            'description': 'Diverse agent evaluation benchmarks',
            'quality_indicators': ['multi_domain', 'agent_evaluation', 'task_variety']
        },
        'react': {
            'path': 'chenxwh/ReAct',
            'config': None,
            'split': 'train',
            'description': 'Reasoning and acting sequences',
            'quality_indicators': ['reasoning_chain', 'action_execution', 'observation']
        },
        'hotpotqa': {
            'path': 'hotpot_qa',
            'config': 'distractor',
            'split': 'train',
            'description': 'Multi-hop reasoning questions requiring tool use',
            'quality_indicators': ['multi_hop', 'reasoning', 'evidence_collection']
        },
        'natural_questions': {
            'path': 'natural_questions',
            'config': None,
            'split': 'train',
            'description': 'Questions requiring search and synthesis',
            'quality_indicators': ['search_query', 'information_synthesis', 'answer_generation']
        },
        'toolllama': {
            'path': 'ToolBench/ToolBench',
            'config': None,
            'split': 'train',
            'description': 'Tool learning with real APIs',
            'quality_indicators': ['api_calling', 'tool_chain', 'real_apis']
        }
    }
    
    def __init__(self, config: CollectionConfig, target_datasets: Optional[List[str]] = None):
        super().__init__(config, "huggingface")
        self.hf_api = HfApi()
        self.target_datasets = target_datasets or list(self.AGENTIC_DATASETS.keys())
        self.current_dataset = None
        self.current_data = None
    
    async def get_item_ids(self) -> Iterator[str]:
        """Get iterator of dataset item IDs"""
        item_ids = []
        
        for dataset_name in self.target_datasets:
            if dataset_name not in self.AGENTIC_DATASETS:
                self.logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            dataset_info = self.AGENTIC_DATASETS[dataset_name]
            
            try:
                # Load dataset metadata to get size
                dataset_path = dataset_info['path']
                config_name = dataset_info.get('config')
                split = dataset_info['split']
                
                self.logger.info(f"Loading dataset {dataset_name}: {dataset_path}")
                
                # Check if dataset exists and is accessible
                try:
                    dataset = load_dataset(
                        dataset_path, 
                        config_name, 
                        split=split,
                        streaming=True  # Use streaming to avoid loading everything
                    )
                    
                    # Get first few items to determine structure
                    sample_items = list(dataset.take(10))
                    if not sample_items:
                        self.logger.warning(f"Dataset {dataset_name} is empty")
                        continue
                    
                    # Estimate dataset size (for non-streaming datasets)
                    try:
                        full_dataset = load_dataset(dataset_path, config_name, split=split)
                        dataset_size = len(full_dataset)
                        self.logger.info(f"Dataset {dataset_name} has {dataset_size} items")
                    except:
                        dataset_size = 10000  # Default estimate for streaming datasets
                        self.logger.info(f"Dataset {dataset_name} size unknown, estimating {dataset_size}")
                    
                    # Generate item IDs
                    for i in range(min(dataset_size, 10000)):  # Limit to 10k items per dataset
                        item_ids.append(f"{dataset_name}_{i}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue
        
        self.logger.info(f"Generated {len(item_ids)} item IDs from {len(self.target_datasets)} datasets")
        return iter(item_ids)
    
    async def fetch_item(self, item_id: str) -> Optional[DataItem]:
        """Fetch a single item from HuggingFace dataset"""
        
        # Parse item ID
        parts = item_id.split('_', 1)
        if len(parts) != 2:
            return None
        
        dataset_name, index_str = parts
        try:
            index = int(index_str)
        except ValueError:
            return None
        
        if dataset_name not in self.AGENTIC_DATASETS:
            return None
        
        dataset_info = self.AGENTIC_DATASETS[dataset_name]
        
        try:
            # Load dataset if not cached
            if self.current_dataset != dataset_name:
                self.logger.info(f"Loading dataset {dataset_name}")
                
                dataset_path = dataset_info['path']
                config_name = dataset_info.get('config')
                split = dataset_info['split']
                
                self.current_data = load_dataset(dataset_path, config_name, split=split)
                self.current_dataset = dataset_name
            
            # Get item by index
            if index >= len(self.current_data):
                return None
            
            raw_item = self.current_data[index]
            
            # Convert to standardized format
            processed_content = await self._process_dataset_item(dataset_name, raw_item, dataset_info)
            if not processed_content:
                return None
            
            # Create DataItem
            item = DataItem(
                source=f"huggingface_{dataset_name}",
                item_id=item_id,
                content=processed_content,
                quality_score=0.0,  # Will be assessed later
                timestamp=datetime.now(),
                metadata={
                    'dataset_name': dataset_name,
                    'dataset_path': dataset_info['path'],
                    'original_index': index,
                    'description': dataset_info['description'],
                    'quality_indicators': dataset_info['quality_indicators']
                }
            )
            
            return item
            
        except Exception as e:
            self.logger.error(f"Error fetching item {item_id}: {e}")
            return None
    
    async def _process_dataset_item(self, dataset_name: str, raw_item: Dict[str, Any], dataset_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process raw dataset item into standardized format"""
        
        try:
            if dataset_name == 'toolbench':
                return await self._process_toolbench_item(raw_item)
            elif dataset_name == 'qwen_agent':
                return await self._process_qwen_agent_item(raw_item)
            elif dataset_name == 'webarena':
                return await self._process_webarena_item(raw_item)
            elif dataset_name == 'agentbench':
                return await self._process_agentbench_item(raw_item)
            elif dataset_name == 'react':
                return await self._process_react_item(raw_item)
            elif dataset_name == 'hotpotqa':
                return await self._process_hotpotqa_item(raw_item)
            elif dataset_name == 'natural_questions':
                return await self._process_natural_questions_item(raw_item)
            elif dataset_name == 'toolllama':
                return await self._process_toolllama_item(raw_item)
            else:
                # Generic processing
                return await self._process_generic_item(raw_item)
                
        except Exception as e:
            self.logger.error(f"Error processing {dataset_name} item: {e}")
            return None
    
    async def _process_toolbench_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process ToolBench/Gorilla function calling data"""
        
        # Extract key fields
        function_call = item.get('function_call', '')
        api_call = item.get('api_call', '')
        description = item.get('description', '')
        
        if not function_call and not api_call:
            return None
        
        # Create conversation format
        turns = []
        
        # User request (if available)
        if description:
            turns.append({
                'role': 'user',
                'content': description
            })
        
        # Assistant response with function call
        assistant_content = f"I'll help you with that. Let me call the appropriate function."
        tool_calls = []
        
        if function_call:
            try:
                # Parse function call
                if isinstance(function_call, str):
                    func_data = json.loads(function_call) if function_call.startswith('{') else {'function': function_call}
                else:
                    func_data = function_call
                
                tool_calls.append({
                    'type': 'function',
                    'function': func_data
                })
            except:
                pass
        
        if api_call:
            tool_calls.append({
                'type': 'api_call',
                'call': api_call
            })
        
        turns.append({
            'role': 'assistant',
            'content': assistant_content,
            'tool_calls': tool_calls
        })
        
        return {
            'conversation_type': 'function_calling',
            'turns': turns,
            'agentic_patterns': ['function_calling', 'api_usage', 'tool_selection'],
            'domain': 'general',
            'complexity': 'medium',
            'original_data': item
        }
    
    async def _process_qwen_agent_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process Qwen agent data"""
        
        conversations = item.get('conversations', [])
        if not conversations:
            return None
        
        turns = []
        agentic_patterns = []
        
        for conv in conversations:
            role = conv.get('from', '').lower()
            content = conv.get('value', '')
            
            if role in ['human', 'user']:
                turns.append({
                    'role': 'user',
                    'content': content
                })
            elif role in ['gpt', 'assistant', 'agent']:
                turn = {
                    'role': 'assistant',
                    'content': content
                }
                
                # Extract tool usage patterns
                if 'tool_call' in content.lower() or 'function' in content.lower():
                    agentic_patterns.append('tool_calling')
                if 'search' in content.lower() or 'browse' in content.lower():
                    agentic_patterns.append('information_retrieval')
                if 'step' in content.lower() or 'first' in content.lower():
                    agentic_patterns.append('step_by_step_reasoning')
                
                turns.append(turn)
        
        if not turns:
            return None
        
        return {
            'conversation_type': 'agent_interaction',
            'turns': turns,
            'agentic_patterns': list(set(agentic_patterns)) or ['general_assistance'],
            'domain': 'general',
            'complexity': 'medium',
            'original_data': item
        }
    
    async def _process_webarena_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process WebArena web interaction data"""
        
        task = item.get('task', '')
        website = item.get('website', '')
        actions = item.get('actions', [])
        
        if not task:
            return None
        
        turns = []
        
        # User task
        turns.append({
            'role': 'user',
            'content': f"Please help me complete this web task: {task} on website: {website}"
        })
        
        # Agent response with actions
        action_descriptions = []
        tool_calls = []
        
        for action in actions:
            if isinstance(action, dict):
                action_type = action.get('type', '')
                target = action.get('target', '')
                value = action.get('value', '')
                
                action_descriptions.append(f"{action_type} on {target} with value {value}")
                tool_calls.append({
                    'type': 'web_action',
                    'action': action
                })
        
        assistant_content = f"I'll complete this task by performing these web actions: {'; '.join(action_descriptions)}"
        
        turns.append({
            'role': 'assistant',
            'content': assistant_content,
            'tool_calls': tool_calls
        })
        
        return {
            'conversation_type': 'web_interaction',
            'turns': turns,
            'agentic_patterns': ['web_navigation', 'multi_step_execution', 'task_completion'],
            'domain': 'web_browsing',
            'complexity': 'high',
            'original_data': item
        }
    
    async def _process_agentbench_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process AgentBench evaluation data"""
        
        task_description = item.get('task', item.get('description', ''))
        expected_output = item.get('expected_output', '')
        category = item.get('category', 'general')
        
        if not task_description:
            return None
        
        turns = [
            {
                'role': 'user',
                'content': task_description
            }
        ]
        
        if expected_output:
            turns.append({
                'role': 'assistant',
                'content': expected_output
            })
        
        # Determine complexity based on task description
        complexity = 'low'
        if any(word in task_description.lower() for word in ['multiple', 'steps', 'complex', 'analyze']):
            complexity = 'high'
        elif any(word in task_description.lower() for word in ['use', 'tool', 'search', 'find']):
            complexity = 'medium'
        
        return {
            'conversation_type': 'benchmark_task',
            'turns': turns,
            'agentic_patterns': ['task_execution', 'problem_solving'],
            'domain': category,
            'complexity': complexity,
            'original_data': item
        }
    
    async def _process_react_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process ReAct reasoning and acting data"""
        
        question = item.get('question', '')
        thought = item.get('thought', '')
        action = item.get('action', '')
        observation = item.get('observation', '')
        
        if not question:
            return None
        
        turns = [
            {
                'role': 'user',
                'content': question
            }
        ]
        
        # Build reasoning chain
        assistant_content = ""
        if thought:
            assistant_content += f"Thought: {thought}\n"
        if action:
            assistant_content += f"Action: {action}\n"
        if observation:
            assistant_content += f"Observation: {observation}"
        
        if assistant_content:
            turns.append({
                'role': 'assistant',
                'content': assistant_content.strip(),
                'reasoning_chain': {
                    'thought': thought,
                    'action': action,
                    'observation': observation
                }
            })
        
        return {
            'conversation_type': 'reasoning_acting',
            'turns': turns,
            'agentic_patterns': ['reasoning_chain', 'action_execution', 'observation_processing'],
            'domain': 'question_answering',
            'complexity': 'medium',
            'original_data': item
        }
    
    async def _process_hotpotqa_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process HotpotQA multi-hop reasoning data"""
        
        question = item.get('question', '')
        answer = item.get('answer', '')
        supporting_facts = item.get('supporting_facts', [])
        
        if not question or not answer:
            return None
        
        turns = [
            {
                'role': 'user', 
                'content': question
            }
        ]
        
        # Build multi-hop reasoning response
        assistant_content = f"To answer this question, I need to find information from multiple sources.\n\n"
        
        if supporting_facts:
            assistant_content += "Let me search for the relevant information:\n"
            for i, fact in enumerate(supporting_facts[:3]):  # Limit to 3 facts
                if isinstance(fact, list) and len(fact) >= 2:
                    title, sentence_id = fact[0], fact[1]
                    assistant_content += f"{i+1}. Search for information about {title}\n"
        
        assistant_content += f"\nBased on the information gathered: {answer}"
        
        turns.append({
            'role': 'assistant',
            'content': assistant_content,
            'supporting_facts': supporting_facts
        })
        
        return {
            'conversation_type': 'multi_hop_qa',
            'turns': turns,
            'agentic_patterns': ['multi_hop_reasoning', 'information_synthesis', 'evidence_collection'],
            'domain': 'question_answering',
            'complexity': 'high',
            'original_data': item
        }
    
    async def _process_natural_questions_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process Natural Questions data"""
        
        question = item.get('question', {}).get('text', '')
        annotations = item.get('annotations', [])
        
        if not question:
            return None
        
        turns = [
            {
                'role': 'user',
                'content': question
            }
        ]
        
        # Extract answer from annotations
        if annotations:
            annotation = annotations[0]
            short_answers = annotation.get('short_answers', [])
            yes_no_answer = annotation.get('yes_no_answer', '')
            
            if short_answers:
                answer_text = short_answers[0].get('text', '')
                if answer_text:
                    turns.append({
                        'role': 'assistant',
                        'content': f"Based on my search, the answer is: {answer_text}"
                    })
            elif yes_no_answer:
                turns.append({
                    'role': 'assistant',
                    'content': f"The answer is: {yes_no_answer}"
                })
        
        if len(turns) == 1:  # No answer found
            return None
        
        return {
            'conversation_type': 'natural_qa',
            'turns': turns,
            'agentic_patterns': ['search_query', 'information_retrieval', 'answer_generation'],
            'domain': 'question_answering',
            'complexity': 'medium',
            'original_data': item
        }
    
    async def _process_toolllama_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process ToolLLaMA data"""
        
        conversations = item.get('conversations', [])
        tools = item.get('tools', [])
        
        if not conversations:
            return None
        
        turns = []
        agentic_patterns = ['tool_calling', 'api_usage']
        
        for conv in conversations:
            role = conv.get('from', '').lower()
            content = conv.get('value', '')
            
            if role in ['human', 'user']:
                turns.append({
                    'role': 'user',
                    'content': content
                })
            elif role in ['gpt', 'assistant']:
                turn = {
                    'role': 'assistant',
                    'content': content
                }
                
                # Extract tool calls from content
                if tools and any(tool_name in content for tool_name in [t.get('name', '') for t in tools]):
                    agentic_patterns.append('real_api_usage')
                    turn['available_tools'] = tools
                
                turns.append(turn)
        
        return {
            'conversation_type': 'tool_usage',
            'turns': turns,
            'agentic_patterns': agentic_patterns,
            'domain': 'api_interaction',
            'complexity': 'high',
            'original_data': item
        }
    
    async def _process_generic_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generic processing for unknown dataset formats"""
        
        # Try to extract text content
        text_fields = ['text', 'content', 'input', 'output', 'question', 'answer']
        content = {}
        
        for field in text_fields:
            if field in item and item[field]:
                content[field] = item[field]
        
        if not content:
            return None
        
        # Create basic conversation
        turns = []
        if 'question' in content:
            turns.append({
                'role': 'user',
                'content': content['question']
            })
            if 'answer' in content:
                turns.append({
                    'role': 'assistant',
                    'content': content['answer']
                })
        elif 'input' in content:
            turns.append({
                'role': 'user',
                'content': content['input']
            })
            if 'output' in content:
                turns.append({
                    'role': 'assistant',
                    'content': content['output']
                })
        elif 'text' in content:
            turns.append({
                'role': 'user',
                'content': content['text']
            })
        
        return {
            'conversation_type': 'generic',
            'turns': turns,
            'agentic_patterns': ['general_assistance'],
            'domain': 'general',
            'complexity': 'low',
            'original_data': item
        }
    
    async def assess_quality(self, item: DataItem) -> float:
        """Assess quality of HuggingFace dataset item"""
        
        content = item.content
        turns = content.get('turns', [])
        agentic_patterns = content.get('agentic_patterns', [])
        
        score = 0.0
        
        # Basic quality checks
        if not turns:
            return 0.0
        
        # Multi-turn bonus
        if len(turns) >= 2:
            score += 0.3
        
        # Agentic patterns bonus
        high_value_patterns = [
            'function_calling', 'tool_calling', 'api_usage', 'reasoning_chain',
            'multi_step_execution', 'web_navigation', 'multi_hop_reasoning'
        ]
        
        pattern_score = len([p for p in agentic_patterns if p in high_value_patterns]) * 0.15
        score += min(pattern_score, 0.4)
        
        # Content quality
        total_content_length = sum(len(turn.get('content', '')) for turn in turns)
        if total_content_length > 100:
            score += 0.2
        if total_content_length > 500:
            score += 0.1
        
        # Tool usage bonus
        has_tool_calls = any('tool_calls' in turn for turn in turns)
        if has_tool_calls:
            score += 0.2
        
        # Reasoning bonus
        has_reasoning = any('reasoning' in turn.get('content', '').lower() for turn in turns)
        if has_reasoning:
            score += 0.1
        
        return min(score, 1.0)


async def collect_huggingface_datasets(
    config: CollectionConfig,
    target_datasets: Optional[List[str]] = None,
    max_items_per_dataset: Optional[int] = None
) -> Dict[str, Any]:
    """Collect data from HuggingFace datasets"""
    
    collector = HuggingFaceCollector(config, target_datasets)
    
    async with collector:
        metrics = await collector.collect(max_items_per_dataset)
        
        return {
            'source': 'huggingface',
            'metrics': metrics,
            'datasets_processed': collector.target_datasets,
            'output_directory': str(collector.output_dir)
        } 