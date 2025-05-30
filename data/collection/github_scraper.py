"""
GitHub Scraper for Agentic Tool Usage Examples

This module scrapes GitHub repositories, issues, and discussions to find real-world
examples of agentic tool usage, particularly AI coding assistants, development workflows,
and tool-calling patterns similar to Cursor, GitHub Copilot, and other AI development tools.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
from dataclasses import dataclass
import aiohttp
from urllib.parse import urlencode

from data.collection.base_collector import BaseCollector, DataItem, CollectionConfig


@dataclass
class GitHubConfig:
    """Configuration for GitHub API scraping"""
    # API settings
    api_token: Optional[str] = None
    base_url: str = "https://api.github.com"
    
    # Search settings
    search_queries: List[str] = None
    repositories: List[str] = None
    max_items_per_query: int = 100
    max_age_days: int = 90
    
    # Content filters
    min_content_length: int = 200
    require_code_examples: bool = True
    require_tool_mentions: bool = True
    
    def __post_init__(self):
        if self.search_queries is None:
            self.search_queries = [
                # AI Coding Assistants
                "cursor AI assistant code generation",
                "github copilot workflow examples", 
                "AI pair programming tutorial",
                "code assistant tool usage",
                "automated coding workflow",
                
                # Tool Usage Patterns
                "API integration step by step",
                "debugging workflow with tools",
                "code review AI assistant",
                "automated testing setup",
                "deployment pipeline tutorial",
                
                # Development Workflows
                "AI code completion workflow",
                "intelligent code suggestions",
                "automated refactoring examples",
                "code analysis tool usage",
                "development productivity tools"
            ]
        
        if self.repositories is None:
            self.repositories = [
                # AI Development Tools
                "microsoft/vscode",
                "github/copilot-docs", 
                "cursor-ai/cursor",
                "sourcegraph/sourcegraph",
                "tabnine/tabnine-vscode",
                
                # Development Workflow Examples
                "microsoft/TypeScript",
                "facebook/react",
                "python/cpython",
                "rust-lang/rust",
                "tensorflow/tensorflow"
            ]


class GitHubScraper(BaseCollector):
    """Scrape GitHub for real-world agentic tool usage examples"""
    
    # AI tool keywords to look for in content
    AI_TOOL_KEYWORDS = [
        # AI Assistants
        "cursor", "copilot", "tabnine", "codewhisperer", "kite",
        "ai assistant", "code completion", "ai pair programming",
        "intelligent suggestions", "automated refactoring",
        
        # Development Tools
        "vscode", "intellij", "pycharm", "visual studio",
        "debugger", "linter", "formatter", "type checker",
        "git integration", "api client", "test runner",
        
        # Workflow Patterns
        "step by step", "workflow", "automation", "pipeline",
        "integration", "deployment", "ci/cd", "testing",
        "code review", "pull request", "issue tracking"
    ]
    
    # Patterns that indicate agentic behavior
    AGENTIC_PATTERNS = {
        "tool_selection": [
            r"use\s+(\w+)\s+to\s+",
            r"run\s+(\w+)\s+command",
            r"execute\s+(\w+)",
            r"invoke\s+(\w+)",
            r"call\s+(\w+)\s+api"
        ],
        "step_by_step": [
            r"step\s+\d+",
            r"first,?\s+",
            r"then,?\s+", 
            r"next,?\s+",
            r"finally,?\s+"
        ],
        "reasoning": [
            r"because\s+",
            r"since\s+",
            r"therefore\s+",
            r"as\s+a\s+result",
            r"this\s+will\s+"
        ],
        "error_handling": [
            r"if\s+.+\s+fails",
            r"in\s+case\s+of\s+error",
            r"troubleshoot",
            r"debug",
            r"fix\s+the\s+issue"
        ]
    }
    
    def __init__(self, config: CollectionConfig, github_config: GitHubConfig):
        super().__init__(config, "github")
        self.github_config = github_config
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "DeepCoder-DataCollector/1.0"
        }
        if github_config.api_token:
            self.headers["Authorization"] = f"token {github_config.api_token}"
    
    async def get_item_ids(self) -> Iterator[str]:
        """Generate GitHub item IDs from various sources"""
        item_ids = []
        
        # Search queries
        for i, query in enumerate(self.github_config.search_queries):
            # Issues
            item_ids.append(f"search_issues|{i}|{query}")
            # Discussions  
            item_ids.append(f"search_discussions|{i}|{query}")
        
        # Repository-specific searches
        for i, repo in enumerate(self.github_config.repositories):
            # Recent issues
            item_ids.append(f"repo_issues|{i}|{repo}")
            # Pull requests with discussions
            item_ids.append(f"repo_pulls|{i}|{repo}")
        
        self.logger.info(f"Generated {len(item_ids)} GitHub item IDs")
        return iter(item_ids)
    
    async def fetch_item(self, item_id: str) -> Optional[DataItem]:
        """Fetch and process a GitHub item"""
        
        # Parse item ID
        parts = item_id.split('|')
        if len(parts) != 3:
            return None
        
        search_type, index, query_or_repo = parts
        
        try:
            if search_type == "search_issues":
                return await self._fetch_search_issues(query_or_repo, int(index))
            elif search_type == "search_discussions": 
                return await self._fetch_search_discussions(query_or_repo, int(index))
            elif search_type == "repo_issues":
                return await self._fetch_repo_issues(query_or_repo, int(index))
            elif search_type == "repo_pulls":
                return await self._fetch_repo_pulls(query_or_repo, int(index))
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching GitHub item {item_id}: {e}")
            return None
    
    async def _fetch_search_issues(self, query: str, index: int) -> Optional[DataItem]:
        """Search GitHub issues for agentic tool usage examples"""
        
        # Build search query
        since_date = (datetime.now() - timedelta(days=self.github_config.max_age_days)).strftime("%Y-%m-%d")
        search_params = {
            "q": f"{query} is:issue created:>{since_date}",
            "sort": "updated",
            "order": "desc",
            "per_page": min(self.github_config.max_items_per_query, 30)
        }
        
        url = f"{self.github_config.base_url}/search/issues?" + urlencode(search_params)
        
        async with self.session.get(url, headers=self.headers) as response:
            if response.status != 200:
                self.logger.error(f"GitHub API error {response.status} for query: {query}")
                return None
            
            data = await response.json()
            
            if not data.get("items"):
                return None
            
            # Find the best issue with agentic patterns
            best_issue = await self._select_best_issue(data["items"])
            if not best_issue:
                return None
            
            # Extract conversation
            conversation = await self._extract_issue_conversation(best_issue)
            if not conversation:
                return None
            
            return DataItem(
                source="github",
                item_id=f"search_issues|{index}|{query}",
                content=conversation,
                quality_score=0.0,  # Will be assessed later
                timestamp=datetime.now(),
                metadata={
                    "search_type": "issues",
                    "query": query,
                    "repository": best_issue.get("repository_url", "").split("/")[-2:],
                    "issue_number": best_issue.get("number"),
                    "created_at": best_issue.get("created_at"),
                    "url": best_issue.get("html_url")
                }
            )
    
    async def _fetch_search_discussions(self, query: str, index: int) -> Optional[DataItem]:
        """Search GitHub discussions for agentic examples"""
        
        # GitHub Discussions use GraphQL API
        graphql_query = """
        query($searchQuery: String!, $first: Int!) {
          search(query: $searchQuery, type: DISCUSSION, first: $first) {
            edges {
              node {
                ... on Discussion {
                  title
                  body
                  url
                  createdAt
                  repository {
                    nameWithOwner
                  }
                  comments(first: 10) {
                    edges {
                      node {
                        body
                        author {
                          login
                        }
                        createdAt
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        since_date = (datetime.now() - timedelta(days=self.github_config.max_age_days)).strftime("%Y-%m-%d")
        search_query = f"{query} created:>{since_date}"
        
        payload = {
            "query": graphql_query,
            "variables": {
                "searchQuery": search_query,
                "first": min(self.github_config.max_items_per_query, 20)
            }
        }
        
        url = "https://api.github.com/graphql"
        async with self.session.post(url, headers=self.headers, json=payload) as response:
            if response.status != 200:
                self.logger.error(f"GitHub GraphQL API error {response.status} for query: {query}")
                return None
            
            data = await response.json()
            
            if "errors" in data:
                self.logger.error(f"GraphQL errors: {data['errors']}")
                return None
            
            discussions = data.get("data", {}).get("search", {}).get("edges", [])
            if not discussions:
                return None
            
            # Find best discussion
            best_discussion = await self._select_best_discussion([edge["node"] for edge in discussions])
            if not best_discussion:
                return None
            
            # Extract conversation
            conversation = await self._extract_discussion_conversation(best_discussion)
            if not conversation:
                return None
            
            return DataItem(
                source="github",
                item_id=f"search_discussions|{index}|{query}",
                content=conversation,
                quality_score=0.0,
                timestamp=datetime.now(),
                metadata={
                    "search_type": "discussions",
                    "query": query,
                    "repository": best_discussion.get("repository", {}).get("nameWithOwner"),
                    "created_at": best_discussion.get("createdAt"),
                    "url": best_discussion.get("url")
                }
            )
    
    async def _fetch_repo_issues(self, repo: str, index: int) -> Optional[DataItem]:
        """Fetch recent issues from a specific repository"""
        
        since_date = (datetime.now() - timedelta(days=self.github_config.max_age_days)).isoformat()
        params = {
            "state": "all",
            "since": since_date,
            "sort": "updated",
            "direction": "desc",
            "per_page": 20
        }
        
        url = f"{self.github_config.base_url}/repos/{repo}/issues?" + urlencode(params)
        
        async with self.session.get(url, headers=self.headers) as response:
            if response.status != 200:
                self.logger.error(f"GitHub API error {response.status} for repo: {repo}")
                return None
            
            issues = await response.json()
            if not issues:
                return None
            
            # Filter for issues with tool usage patterns
            tool_issues = [issue for issue in issues if await self._has_tool_usage(issue)]
            if not tool_issues:
                return None
            
            best_issue = tool_issues[0]  # Take the most recent relevant issue
            
            conversation = await self._extract_issue_conversation(best_issue)
            if not conversation:
                return None
            
            return DataItem(
                source="github", 
                item_id=f"repo_issues|{index}|{repo}",
                content=conversation,
                quality_score=0.0,
                timestamp=datetime.now(),
                metadata={
                    "search_type": "repo_issues",
                    "repository": repo,
                    "issue_number": best_issue.get("number"),
                    "created_at": best_issue.get("created_at"),
                    "url": best_issue.get("html_url")
                }
            )
    
    async def _fetch_repo_pulls(self, repo: str, index: int) -> Optional[DataItem]:
        """Fetch pull requests with rich discussions"""
        
        params = {
            "state": "all",
            "sort": "updated", 
            "direction": "desc",
            "per_page": 15
        }
        
        url = f"{self.github_config.base_url}/repos/{repo}/pulls?" + urlencode(params)
        
        async with self.session.get(url, headers=self.headers) as response:
            if response.status != 200:
                self.logger.error(f"GitHub API error {response.status} for repo pulls: {repo}")
                return None
            
            pulls = await response.json()
            if not pulls:
                return None
            
            # Find PRs with substantial discussions
            for pull in pulls:
                if pull.get("comments", 0) > 2:  # Has discussion
                    conversation = await self._extract_pull_conversation(pull)
                    if conversation:
                        return DataItem(
                            source="github",
                            item_id=f"repo_pulls|{index}|{repo}",
                            content=conversation,
                            quality_score=0.0,
                            timestamp=datetime.now(),
                            metadata={
                                "search_type": "repo_pulls",
                                "repository": repo,
                                "pull_number": pull.get("number"),
                                "created_at": pull.get("created_at"),
                                "url": pull.get("html_url")
                            }
                        )
            
            return None
    
    async def _select_best_issue(self, issues: List[Dict]) -> Optional[Dict]:
        """Select the issue with the best agentic patterns"""
        
        scored_issues = []
        
        for issue in issues:
            score = await self._score_content_quality(issue.get("body", "") + issue.get("title", ""))
            if score > 0.3:  # Minimum quality threshold
                scored_issues.append((score, issue))
        
        if not scored_issues:
            return None
        
        # Return highest scoring issue
        return max(scored_issues, key=lambda x: x[0])[1]
    
    async def _select_best_discussion(self, discussions: List[Dict]) -> Optional[Dict]:
        """Select the discussion with the best agentic patterns"""
        
        scored_discussions = []
        
        for discussion in discussions:
            content = discussion.get("body", "") + discussion.get("title", "")
            # Add comments content
            for comment_edge in discussion.get("comments", {}).get("edges", []):
                content += " " + comment_edge.get("node", {}).get("body", "")
            
            score = await self._score_content_quality(content)
            if score > 0.3:
                scored_discussions.append((score, discussion))
        
        if not scored_discussions:
            return None
        
        return max(scored_discussions, key=lambda x: x[0])[1]
    
    async def _has_tool_usage(self, item: Dict) -> bool:
        """Check if an item contains tool usage patterns"""
        
        content = (item.get("body", "") + " " + item.get("title", "")).lower()
        
        # Check for AI tool keywords
        for keyword in self.AI_TOOL_KEYWORDS:
            if keyword.lower() in content:
                return True
        
        return False
    
    async def _score_content_quality(self, content: str) -> float:
        """Score content based on agentic patterns and tool usage"""
        
        if not content or len(content) < self.github_config.min_content_length:
            return 0.0
        
        score = 0.0
        content_lower = content.lower()
        
        # AI tool mentions (40% weight)
        tool_mentions = sum(1 for keyword in self.AI_TOOL_KEYWORDS if keyword.lower() in content_lower)
        score += min(tool_mentions * 0.1, 0.4)
        
        # Agentic patterns (40% weight)
        pattern_score = 0.0
        for pattern_type, patterns in self.AGENTIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    pattern_score += 0.1
        score += min(pattern_score, 0.4)
        
        # Code examples (20% weight)
        if self.github_config.require_code_examples:
            code_blocks = len(re.findall(r'```[\s\S]*?```', content))
            score += min(code_blocks * 0.05, 0.2)
        
        return min(score, 1.0)
    
    async def _extract_issue_conversation(self, issue: Dict) -> Optional[Dict]:
        """Extract conversation from GitHub issue"""
        
        if not issue:
            return None
        
        conversation_turns = []
        
        # Add initial issue
        conversation_turns.append({
            "role": "user",
            "content": f"{issue.get('title', '')}\n\n{issue.get('body', '')}",
            "author": issue.get("user", {}).get("login", "user"),
            "timestamp": issue.get("created_at")
        })
        
        # Fetch and add comments
        comments_url = issue.get("comments_url")
        if comments_url and issue.get("comments", 0) > 0:
            async with self.session.get(comments_url, headers=self.headers) as response:
                if response.status == 200:
                    comments = await response.json()
                    
                    for comment in comments[:10]:  # Limit to first 10 comments
                        if len(comment.get("body", "")) > 50:  # Skip very short comments
                            conversation_turns.append({
                                "role": "assistant" if self._is_helpful_response(comment.get("body", "")) else "user",
                                "content": comment.get("body", ""),
                                "author": comment.get("user", {}).get("login", "user"),
                                "timestamp": comment.get("created_at")
                            })
        
        if len(conversation_turns) < 2:
            return None
        
        # Detect agentic patterns
        agentic_patterns = await self._detect_agentic_patterns(conversation_turns)
        
        return {
            "conversation_type": "github_issue",
            "title": issue.get("title", ""),
            "turns": conversation_turns,
            "agentic_patterns": agentic_patterns,
            "domain": "development_workflow",
            "complexity": self._assess_conversation_complexity(conversation_turns),
            "repository": issue.get("repository_url", "").split("/")[-2:] if issue.get("repository_url") else None,
            "labels": [label.get("name") for label in issue.get("labels", [])],
            "programming_languages": await self._detect_programming_languages(conversation_turns)
        }
    
    async def _extract_discussion_conversation(self, discussion: Dict) -> Optional[Dict]:
        """Extract conversation from GitHub discussion"""
        
        if not discussion:
            return None
        
        conversation_turns = []
        
        # Add initial discussion post
        conversation_turns.append({
            "role": "user",
            "content": f"{discussion.get('title', '')}\n\n{discussion.get('body', '')}",
            "author": "user",
            "timestamp": discussion.get("createdAt")
        })
        
        # Add comments
        for comment_edge in discussion.get("comments", {}).get("edges", [])[:8]:
            comment = comment_edge.get("node", {})
            if len(comment.get("body", "")) > 50:
                conversation_turns.append({
                    "role": "assistant" if self._is_helpful_response(comment.get("body", "")) else "user",
                    "content": comment.get("body", ""),
                    "author": comment.get("author", {}).get("login", "user"),
                    "timestamp": comment.get("createdAt")
                })
        
        if len(conversation_turns) < 2:
            return None
        
        agentic_patterns = await self._detect_agentic_patterns(conversation_turns)
        
        return {
            "conversation_type": "github_discussion",
            "title": discussion.get("title", ""),
            "turns": conversation_turns,
            "agentic_patterns": agentic_patterns,
            "domain": "development_workflow",
            "complexity": self._assess_conversation_complexity(conversation_turns),
            "repository": discussion.get("repository", {}).get("nameWithOwner"),
            "programming_languages": await self._detect_programming_languages(conversation_turns)
        }
    
    async def _extract_pull_conversation(self, pull: Dict) -> Optional[Dict]:
        """Extract conversation from pull request"""
        
        conversation_turns = []
        
        # Add PR description
        if pull.get("body") and len(pull.get("body")) > 100:
            conversation_turns.append({
                "role": "user",
                "content": f"{pull.get('title', '')}\n\n{pull.get('body', '')}",
                "author": pull.get("user", {}).get("login", "user"),
                "timestamp": pull.get("created_at")
            })
        
        # Fetch comments
        comments_url = pull.get("comments_url")
        if comments_url:
            async with self.session.get(comments_url, headers=self.headers) as response:
                if response.status == 200:
                    comments = await response.json()
                    
                    for comment in comments[:6]:  # Limit comments
                        if len(comment.get("body", "")) > 50:
                            conversation_turns.append({
                                "role": "assistant" if self._is_helpful_response(comment.get("body", "")) else "user",
                                "content": comment.get("body", ""),
                                "author": comment.get("user", {}).get("login", "user"),
                                "timestamp": comment.get("created_at")
                            })
        
        if len(conversation_turns) < 2:
            return None
        
        agentic_patterns = await self._detect_agentic_patterns(conversation_turns)
        
        return {
            "conversation_type": "github_pull_request",
            "title": pull.get("title", ""),
            "turns": conversation_turns,
            "agentic_patterns": agentic_patterns,
            "domain": "code_review_workflow",
            "complexity": self._assess_conversation_complexity(conversation_turns),
            "repository": pull.get("base", {}).get("repo", {}).get("full_name"),
            "programming_languages": await self._detect_programming_languages(conversation_turns)
        }
    
    def _is_helpful_response(self, content: str) -> bool:
        """Determine if a response is helpful/instructional"""
        
        helpful_indicators = [
            "you can", "try", "use", "run", "execute", "install",
            "here's how", "solution", "fix", "resolve", "steps",
            "example", "tutorial", "guide", "workflow"
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in helpful_indicators)
    
    async def _detect_agentic_patterns(self, turns: List[Dict]) -> List[str]:
        """Detect agentic patterns in conversation"""
        
        patterns = set()
        full_content = " ".join(turn.get("content", "") for turn in turns).lower()
        
        for pattern_type, regex_patterns in self.AGENTIC_PATTERNS.items():
            for pattern in regex_patterns:
                if re.search(pattern, full_content):
                    patterns.add(pattern_type)
        
        # Additional pattern detection
        if any(keyword in full_content for keyword in ["tool", "command", "api", "function"]):
            patterns.add("tool_usage")
        
        if len(turns) > 3:
            patterns.add("multi_turn_interaction")
        
        return list(patterns)
    
    def _assess_conversation_complexity(self, turns: List[Dict]) -> str:
        """Assess conversation complexity"""
        
        total_length = sum(len(turn.get("content", "")) for turn in turns)
        
        if total_length > 2000 and len(turns) > 4:
            return "high"
        elif total_length > 800 and len(turns) > 2:
            return "medium"
        else:
            return "low"
    
    async def _detect_programming_languages(self, turns: List[Dict]) -> List[str]:
        """Detect programming languages mentioned in conversation"""
        
        languages = set()
        content = " ".join(turn.get("content", "") for turn in turns).lower()
        
        lang_keywords = {
            "python": ["python", "py", "pip", "conda", "jupyter"],
            "javascript": ["javascript", "js", "node", "npm", "yarn"],
            "typescript": ["typescript", "ts"],
            "rust": ["rust", "cargo"],
            "go": ["golang", "go"],
            "java": ["java", "maven", "gradle"],
            "cpp": ["c++", "cpp", "cmake"],
            "csharp": ["c#", "csharp", "dotnet"],
            "php": ["php", "composer"],
            "ruby": ["ruby", "gem"],
            "swift": ["swift", "xcode"],
            "kotlin": ["kotlin"]
        }
        
        for lang, keywords in lang_keywords.items():
            if any(keyword in content for keyword in keywords):
                languages.add(lang)
        
        return list(languages)
    
    async def assess_quality(self, item: DataItem) -> float:
        """Assess quality of GitHub conversation data"""
        
        if not item.content or not isinstance(item.content, dict):
            return 0.0
        
        score = 0.0
        content = item.content
        
        # Agentic patterns (40% weight)
        agentic_patterns = content.get("agentic_patterns", [])
        score += min(len(agentic_patterns) * 0.1, 0.4)
        
        # Conversation length and depth (30% weight)
        turns = content.get("turns", [])
        if len(turns) >= 3:
            score += 0.15
        if len(turns) >= 5:
            score += 0.15
        
        # Technical content quality (20% weight)
        programming_langs = content.get("programming_languages", [])
        score += min(len(programming_langs) * 0.05, 0.2)
        
        # Content complexity (10% weight)
        complexity = content.get("complexity", "low")
        if complexity == "high":
            score += 0.1
        elif complexity == "medium":
            score += 0.05
        
        return min(score, 1.0)


async def collect_github_data(
    config: CollectionConfig,
    github_config: GitHubConfig,
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """Collect GitHub agentic data"""
    
    scraper = GitHubScraper(config, github_config)
    
    async with scraper:
        metrics = await scraper.collect(max_items)
        
        return {
            "source": "github",
            "metrics": metrics,
            "api_token_used": bool(github_config.api_token),
            "search_queries": github_config.search_queries,
            "repositories": github_config.repositories,
            "output_directory": str(scraper.output_dir)
        } 