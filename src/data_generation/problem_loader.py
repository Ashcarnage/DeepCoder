"""
Problem Loader for DeepCoder
Handles loading and managing coding problems for trajectory generation.
"""

import json
import random
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

@dataclass
class Problem:
    """Represents a coding problem"""
    id: str
    title: str
    description: str
    difficulty: str
    category: str
    examples: List[Dict[str, Any]]
    constraints: Optional[str] = None
    hints: Optional[List[str]] = None
    solution: Optional[str] = None
    test_cases: Optional[List[Dict[str, Any]]] = None

class ProblemLoader:
    """
    Loads and manages coding problems for trajectory generation
    """
    
    def __init__(self):
        """Initialize problem loader"""
        self.problems = []
        self.problems_by_category = {}
        self.problems_by_difficulty = {}
        
    def load_problems(self, file_path: str) -> List[Dict[str, Any]]:
        """Load problems from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Problems file not found: {file_path}")
            # Create sample problems
            return self._create_sample_problems()
        
        problems = []
        
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            problem = json.loads(line.strip())
                            problems.append(problem)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        problems = data
                    else:
                        problems = [data]
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return self._create_sample_problems()
                
        except Exception as e:
            logger.error(f"Error loading problems from {file_path}: {e}")
            return self._create_sample_problems()
        
        # Validate and process problems
        valid_problems = []
        for i, problem in enumerate(problems):
            if self._validate_problem(problem):
                valid_problems.append(problem)
            else:
                logger.warning(f"Invalid problem at index {i}, skipping")
        
        self.problems = valid_problems
        self._organize_problems()
        
        logger.info(f"Loaded {len(valid_problems)} valid problems from {file_path}")
        return valid_problems
    
    def _validate_problem(self, problem: Dict[str, Any]) -> bool:
        """Validate problem structure"""
        required_fields = ['id', 'title', 'description']
        
        for field in required_fields:
            if field not in problem:
                logger.warning(f"Problem missing required field: {field}")
                return False
        
        return True
    
    def _organize_problems(self):
        """Organize problems by category and difficulty"""
        self.problems_by_category = {}
        self.problems_by_difficulty = {}
        
        for problem in self.problems:
            # Organize by category
            category = problem.get('category', 'general')
            if category not in self.problems_by_category:
                self.problems_by_category[category] = []
            self.problems_by_category[category].append(problem)
            
            # Organize by difficulty
            difficulty = problem.get('difficulty', 'medium')
            if difficulty not in self.problems_by_difficulty:
                self.problems_by_difficulty[difficulty] = []
            self.problems_by_difficulty[difficulty].append(problem)
    
    def get_problems_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get problems by category"""
        return self.problems_by_category.get(category, [])
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Get problems by difficulty"""
        return self.problems_by_difficulty.get(difficulty, [])
    
    def sample_problems(
        self, 
        n: int, 
        category: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Sample n problems with optional filtering"""
        if category:
            pool = self.get_problems_by_category(category)
        elif difficulty:
            pool = self.get_problems_by_difficulty(difficulty)
        else:
            pool = self.problems
        
        if not pool:
            logger.warning(f"No problems found for category={category}, difficulty={difficulty}")
            return []
        
        return random.choices(pool, k=min(n, len(pool)))
    
    def _create_sample_problems(self) -> List[Dict[str, Any]]:
        """Create sample problems for testing"""
        sample_problems = [
            {
                "id": "two_sum",
                "title": "Two Sum",
                "description": """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]""",
                "difficulty": "easy",
                "category": "array",
                "examples": [
                    {"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"},
                    {"input": "nums = [3,2,4], target = 6", "output": "[1,2]"},
                    {"input": "nums = [3,3], target = 6", "output": "[0,1]"}
                ],
                "constraints": "2 <= nums.length <= 10^4, -10^9 <= nums[i] <= 10^9, -10^9 <= target <= 10^9"
            },
            {
                "id": "reverse_string",
                "title": "Reverse String",
                "description": """Write a function that reverses a string. The input string is given as an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.

Example 1:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Example 2:
Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]""",
                "difficulty": "easy",
                "category": "string",
                "examples": [
                    {"input": 's = ["h","e","l","l","o"]', "output": '["o","l","l","e","h"]'},
                    {"input": 's = ["H","a","n","n","a","h"]', "output": '["h","a","n","n","a","H"]'}
                ],
                "constraints": "1 <= s.length <= 10^5, s[i] is a printable ascii character"
            },
            {
                "id": "fibonacci",
                "title": "Fibonacci Number",
                "description": """The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1.

F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.

Given n, calculate F(n).

Example 1:
Input: n = 2
Output: 1
Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.

Example 2:
Input: n = 3
Output: 2
Explanation: F(3) = F(2) + F(1) = 1 + 1 = 2.

Example 3:
Input: n = 4
Output: 3
Explanation: F(4) = F(3) + F(2) = 2 + 1 = 3.""",
                "difficulty": "easy",
                "category": "dynamic_programming",
                "examples": [
                    {"input": "n = 2", "output": "1"},
                    {"input": "n = 3", "output": "2"},
                    {"input": "n = 4", "output": "3"}
                ],
                "constraints": "0 <= n <= 30"
            },
            {
                "id": "valid_parentheses",
                "title": "Valid Parentheses",
                "description": """Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "()[]{}"
Output: true

Example 3:
Input: s = "(]"
Output: false""",
                "difficulty": "easy",
                "category": "stack",
                "examples": [
                    {"input": 's = "()"', "output": "true"},
                    {"input": 's = "()[]{}"', "output": "true"},
                    {"input": 's = "(]"', "output": "false"}
                ],
                "constraints": "1 <= s.length <= 10^4, s consists of parentheses only '()[]{}'."
            },
            {
                "id": "binary_search",
                "title": "Binary Search",
                "description": """Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

Example 1:
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Example 2:
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1""",
                "difficulty": "easy",
                "category": "binary_search",
                "examples": [
                    {"input": "nums = [-1,0,3,5,9,12], target = 9", "output": "4"},
                    {"input": "nums = [-1,0,3,5,9,12], target = 2", "output": "-1"}
                ],
                "constraints": "1 <= nums.length <= 10^4, -10^4 < nums[i], target < 10^4, All the integers in nums are unique, nums is sorted in ascending order."
            },
            {
                "id": "merge_sorted_arrays",
                "title": "Merge Sorted Array",
                "description": """You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

Example 1:
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.""",
                "difficulty": "easy",
                "category": "array",
                "examples": [
                    {"input": "nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3", "output": "[1,2,2,3,5,6]"},
                    {"input": "nums1 = [1], m = 1, nums2 = [], n = 0", "output": "[1]"}
                ],
                "constraints": "nums1.length == m + n, nums2.length == n, 0 <= m, n <= 200, 1 <= m + n <= 200"
            },
            {
                "id": "longest_substring",
                "title": "Longest Substring Without Repeating Characters",
                "description": """Given a string s, find the length of the longest substring without repeating characters.

Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.""",
                "difficulty": "medium",
                "category": "string",
                "examples": [
                    {"input": 's = "abcabcbb"', "output": "3"},
                    {"input": 's = "bbbbb"', "output": "1"},
                    {"input": 's = "pwwkew"', "output": "3"}
                ],
                "constraints": "0 <= s.length <= 5 * 10^4, s consists of English letters, digits, symbols and spaces."
            },
            {
                "id": "maximum_subarray",
                "title": "Maximum Subarray",
                "description": """Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

A subarray is a contiguous part of an array.

Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Example 2:
Input: nums = [1]
Output: 1

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23""",
                "difficulty": "medium",
                "category": "dynamic_programming",
                "examples": [
                    {"input": "nums = [-2,1,-3,4,-1,2,1,-5,4]", "output": "6"},
                    {"input": "nums = [1]", "output": "1"},
                    {"input": "nums = [5,4,-1,7,8]", "output": "23"}
                ],
                "constraints": "1 <= nums.length <= 10^5, -10^4 <= nums[i] <= 10^4"
            },
            {
                "id": "climbing_stairs",
                "title": "Climbing Stairs",
                "description": """You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Example 1:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Example 2:
Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step""",
                "difficulty": "easy",
                "category": "dynamic_programming",
                "examples": [
                    {"input": "n = 2", "output": "2"},
                    {"input": "n = 3", "output": "3"}
                ],
                "constraints": "1 <= n <= 45"
            },
            {
                "id": "best_time_to_buy_sell_stock",
                "title": "Best Time to Buy and Sell Stock",
                "description": """You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.""",
                "difficulty": "easy",
                "category": "array",
                "examples": [
                    {"input": "prices = [7,1,5,3,6,4]", "output": "5"},
                    {"input": "prices = [7,6,4,3,1]", "output": "0"}
                ],
                "constraints": "1 <= prices.length <= 10^5, 0 <= prices[i] <= 10^4"
            }
        ]
        
        # Save sample problems to file
        problems_dir = Path("data/problems")
        problems_dir.mkdir(parents=True, exist_ok=True)
        
        problems_file = problems_dir / "coding_problems.jsonl"
        with open(problems_file, 'w') as f:
            for problem in sample_problems:
                f.write(json.dumps(problem) + '\n')
        
        console.print(f"[green]Created {len(sample_problems)} sample problems at {problems_file}[/green]")
        
        return sample_problems
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded problems"""
        if not self.problems:
            return {"total": 0}
        
        stats = {
            "total": len(self.problems),
            "by_category": {cat: len(probs) for cat, probs in self.problems_by_category.items()},
            "by_difficulty": {diff: len(probs) for diff, probs in self.problems_by_difficulty.items()},
            "categories": list(self.problems_by_category.keys()),
            "difficulties": list(self.problems_by_difficulty.keys())
        }
        
        return stats
    
    def display_stats(self):
        """Display problem statistics"""
        stats = self.get_stats()
        
        if stats["total"] == 0:
            console.print("[yellow]No problems loaded[/yellow]")
            return
        
        console.print(f"[bold blue]Problem Statistics[/bold blue]")
        console.print(f"Total problems: {stats['total']}")
        
        if stats.get("by_category"):
            console.print("\n[bold]By Category:[/bold]")
            for category, count in stats["by_category"].items():
                console.print(f"  {category}: {count}")
        
        if stats.get("by_difficulty"):
            console.print("\n[bold]By Difficulty:[/bold]")
            for difficulty, count in stats["by_difficulty"].items():
                console.print(f"  {difficulty}: {count}")

if __name__ == "__main__":
    # Test problem loader
    loader = ProblemLoader()
    
    # Test loading (will create sample problems if file doesn't exist)
    problems = loader.load_problems("data/problems/coding_problems.jsonl")
    
    console.print(f"[green]Loaded {len(problems)} problems[/green]")
    loader.display_stats()
    
    # Test sampling
    sample = loader.sample_problems(3, difficulty="easy")
    console.print(f"\n[blue]Sample of 3 easy problems:[/blue]")
    for i, problem in enumerate(sample, 1):
        console.print(f"{i}. {problem['title']} ({problem['category']})") 