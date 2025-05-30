"""
Base Data Collection Framework for Agentic Training Data

This module provides the foundational infrastructure for collecting agentic training data
from multiple sources with proper rate limiting, error handling, and progress tracking.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Callable
import hashlib
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm
import yaml


@dataclass
class CollectionConfig:
    """Configuration for data collection"""
    output_dir: str = "data/collected"
    max_concurrent_requests: int = 10
    rate_limit_per_second: float = 2.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout_seconds: int = 30
    checkpoint_interval: int = 100
    quality_threshold: float = 0.5
    enable_deduplication: bool = True
    log_level: str = "INFO"


@dataclass
class CollectionMetrics:
    """Metrics for tracking collection progress"""
    total_requested: int = 0
    total_collected: int = 0
    total_filtered: int = 0
    total_duplicates: int = 0
    total_errors: int = 0
    start_time: Optional[datetime] = None
    last_checkpoint: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_requested == 0:
            return 0.0
        return self.total_collected / self.total_requested
    
    @property
    def collection_rate(self) -> float:
        if not self.start_time:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.total_collected / elapsed if elapsed > 0 else 0.0


@dataclass
class DataItem:
    """Base data item structure"""
    source: str
    item_id: str
    content: Dict[str, Any]
    quality_score: float
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataItem':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    @property
    def content_hash(self) -> str:
        """Generate hash for deduplication"""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()


class RateLimiter:
    """Rate limiter with token bucket algorithm"""
    
    def __init__(self, rate_per_second: float):
        self.rate = rate_per_second
        self.tokens = rate_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
            else:
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 0


class ProgressTracker:
    """Progress tracking with persistence"""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.metrics = CollectionMetrics()
        self.processed_ids = set()
        self.lock = asyncio.Lock()
        self._load_checkpoint()
    
    def _load_checkpoint(self) -> None:
        """Load progress from checkpoint file"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    
                # Load metrics
                metrics_data = data.get('metrics', {})
                for key, value in metrics_data.items():
                    if key in ['start_time', 'last_checkpoint'] and value:
                        value = datetime.fromisoformat(value)
                    setattr(self.metrics, key, value)
                
                # Load processed IDs
                self.processed_ids = set(data.get('processed_ids', []))
                
                logging.info(f"Loaded checkpoint: {len(self.processed_ids)} processed items")
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}")
    
    async def save_checkpoint(self) -> None:
        """Save progress to checkpoint file"""
        async with self.lock:
            try:
                # Prepare metrics data
                metrics_data = asdict(self.metrics)
                for key, value in metrics_data.items():
                    if isinstance(value, datetime):
                        metrics_data[key] = value.isoformat()
                
                data = {
                    'metrics': metrics_data,
                    'processed_ids': list(self.processed_ids)
                }
                
                # Ensure directory exists
                self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write atomically
                temp_file = self.checkpoint_file.with_suffix('.tmp')
                async with aiofiles.open(temp_file, 'w') as f:
                    await f.write(json.dumps(data, indent=2))
                
                temp_file.replace(self.checkpoint_file)
                self.metrics.last_checkpoint = datetime.now()
                
            except Exception as e:
                logging.error(f"Failed to save checkpoint: {e}")
    
    async def update_metrics(self, **kwargs) -> None:
        """Update metrics"""
        async with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            
            if not self.metrics.start_time:
                self.metrics.start_time = datetime.now()
    
    async def mark_processed(self, item_id: str) -> None:
        """Mark item as processed"""
        async with self.lock:
            self.processed_ids.add(item_id)
    
    def is_processed(self, item_id: str) -> bool:
        """Check if item was already processed"""
        return item_id in self.processed_ids


class DeduplicationManager:
    """Manage deduplication of collected data"""
    
    def __init__(self, storage_file: Path):
        self.storage_file = storage_file
        self.content_hashes = set()
        self.lock = asyncio.Lock()
        self._load_hashes()
    
    def _load_hashes(self) -> None:
        """Load existing content hashes"""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    self.content_hashes = set(json.load(f))
                logging.info(f"Loaded {len(self.content_hashes)} content hashes")
            except Exception as e:
                logging.warning(f"Failed to load deduplication data: {e}")
    
    async def save_hashes(self) -> None:
        """Save content hashes to storage"""
        async with self.lock:
            try:
                self.storage_file.parent.mkdir(parents=True, exist_ok=True)
                temp_file = self.storage_file.with_suffix('.tmp')
                
                async with aiofiles.open(temp_file, 'w') as f:
                    await f.write(json.dumps(list(self.content_hashes)))
                
                temp_file.replace(self.storage_file)
            except Exception as e:
                logging.error(f"Failed to save deduplication data: {e}")
    
    async def is_duplicate(self, item: DataItem) -> bool:
        """Check if item is duplicate"""
        content_hash = item.content_hash
        async with self.lock:
            return content_hash in self.content_hashes
    
    async def add_hash(self, item: DataItem) -> None:
        """Add content hash"""
        async with self.lock:
            self.content_hashes.add(item.content_hash)


class BaseCollector(ABC):
    """Base class for all data collectors"""
    
    def __init__(self, config: CollectionConfig, source_name: str):
        self.config = config
        self.source_name = source_name
        
        # Setup paths
        self.output_dir = Path(config.output_dir) / source_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.rate_limiter = RateLimiter(config.rate_limit_per_second)
        self.progress = ProgressTracker(self.output_dir / "checkpoint.json")
        self.deduplication = DeduplicationManager(self.output_dir / "dedup.json") if config.enable_deduplication else None
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"collector.{source_name}")
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        
        # Save final state
        await self.progress.save_checkpoint()
        if self.deduplication:
            await self.deduplication.save_hashes()
    
    @abstractmethod
    async def get_item_ids(self) -> Iterator[str]:
        """Get iterator of item IDs to collect"""
        pass
    
    @abstractmethod
    async def fetch_item(self, item_id: str) -> Optional[DataItem]:
        """Fetch a single item by ID"""
        pass
    
    async def assess_quality(self, item: DataItem) -> float:
        """Assess quality of collected item (override for custom logic)"""
        return 1.0  # Default: all items are high quality
    
    async def process_item(self, item_id: str) -> Optional[DataItem]:
        """Process a single item with error handling and rate limiting"""
        
        # Check if already processed
        if self.progress.is_processed(item_id):
            return None
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Update metrics
        await self.progress.update_metrics(total_requested=self.progress.metrics.total_requested + 1)
        
        try:
            # Fetch item with retry logic
            item = await self._fetch_with_retry(item_id)
            if not item:
                await self.progress.update_metrics(total_errors=self.progress.metrics.total_errors + 1)
                return None
            
            # Assess quality
            quality_score = await self.assess_quality(item)
            item.quality_score = quality_score
            
            # Filter by quality
            if quality_score < self.config.quality_threshold:
                await self.progress.update_metrics(total_filtered=self.progress.metrics.total_filtered + 1)
                await self.progress.mark_processed(item_id)
                return None
            
            # Check for duplicates
            if self.deduplication and await self.deduplication.is_duplicate(item):
                await self.progress.update_metrics(total_duplicates=self.progress.metrics.total_duplicates + 1)
                await self.progress.mark_processed(item_id)
                return None
            
            # Save item
            await self._save_item(item)
            
            # Update tracking
            await self.progress.update_metrics(total_collected=self.progress.metrics.total_collected + 1)
            await self.progress.mark_processed(item_id)
            
            if self.deduplication:
                await self.deduplication.add_hash(item)
            
            return item
            
        except Exception as e:
            self.logger.error(f"Error processing item {item_id}: {e}")
            await self.progress.update_metrics(total_errors=self.progress.metrics.total_errors + 1)
            return None
    
    async def _fetch_with_retry(self, item_id: str) -> Optional[DataItem]:
        """Fetch item with retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                return await self.fetch_item(item_id)
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise e
                
                delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                self.logger.warning(f"Retry {attempt + 1} for {item_id} after {delay}s: {e}")
                await asyncio.sleep(delay)
        
        return None
    
    async def _save_item(self, item: DataItem) -> None:
        """Save item to storage"""
        timestamp = datetime.now().strftime("%Y%m%d")
        output_file = self.output_dir / f"{timestamp}.jsonl"
        
        async with aiofiles.open(output_file, 'a') as f:
            await f.write(json.dumps(item.to_dict()) + '\n')
    
    async def collect(self, max_items: Optional[int] = None) -> CollectionMetrics:
        """Main collection method"""
        self.logger.info(f"Starting collection for {self.source_name}")
        
        # Get item IDs
        item_ids = await self.get_item_ids()
        if max_items:
            item_ids = list(item_ids)[:max_items]
        else:
            item_ids = list(item_ids)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_with_semaphore(item_id: str) -> Optional[DataItem]:
            async with semaphore:
                return await self.process_item(item_id)
        
        # Process items with progress bar
        tasks = [process_with_semaphore(item_id) for item_id in item_ids]
        
        checkpoint_counter = 0
        results = []
        
        async for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Collecting {self.source_name}"):
            item = await result
            if item:
                results.append(item)
            
            checkpoint_counter += 1
            if checkpoint_counter % self.config.checkpoint_interval == 0:
                await self.progress.save_checkpoint()
                if self.deduplication:
                    await self.deduplication.save_hashes()
        
        # Final save
        await self.progress.save_checkpoint()
        if self.deduplication:
            await self.deduplication.save_hashes()
        
        self.logger.info(f"Collection complete: {len(results)} items collected")
        return self.progress.metrics


def create_collector_config(config_path: str) -> CollectionConfig:
    """Create collector config from YAML file"""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    collection_config = config_data.get('data_collection', {})
    return CollectionConfig(**collection_config) 