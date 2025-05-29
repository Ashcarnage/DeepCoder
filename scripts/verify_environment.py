#!/usr/bin/env python3
"""
Environment Verification Script for DeepCoder Project
Verifies all dependencies, configurations, and basic functionality.
"""

import os
import sys
import importlib
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message: str, status: str, color: str = Colors.GREEN):
    """Print formatted status message"""
    print(f"{Colors.BOLD}{message:<50}{Colors.ENDC} [{color}{status}{Colors.ENDC}]")

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= min_version:
        print_status("Python Version", f"✓ {sys.version.split()[0]}", Colors.GREEN)
        return True
    else:
        print_status("Python Version", f"✗ {sys.version.split()[0]} (requires 3.8+)", Colors.RED)
        return False

def check_package_installation(packages: List[str]) -> Dict[str, bool]:
    """Check if required packages are installed"""
    results = {}
    
    print(f"\n{Colors.BOLD}Checking Package Dependencies:{Colors.ENDC}")
    
    for package in packages:
        try:
            # Handle special cases
            if package == "unsloth":
                import unsloth
            elif package == "groq":
                import groq
            elif package == "langchain-groq":
                import langchain_groq
            else:
                importlib.import_module(package)
            
            print_status(f"  {package}", "✓ Installed", Colors.GREEN)
            results[package] = True
            
        except ImportError:
            print_status(f"  {package}", "✗ Missing", Colors.RED)
            results[package] = False
        except Exception as e:
            print_status(f"  {package}", f"✗ Error: {str(e)[:30]}", Colors.RED)
            results[package] = False
    
    return results

def check_gpu_availability() -> bool:
    """Check GPU availability and CUDA setup"""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_count > 0 else 0
            
            print_status("CUDA Available", f"✓ {gpu_count} GPU(s)", Colors.GREEN)
            print_status("GPU Name", f"✓ {gpu_name}", Colors.GREEN)
            print_status("GPU Memory", f"✓ {memory:.1f}GB", Colors.GREEN)
            return True
        else:
            print_status("CUDA Available", "✗ No CUDA GPU found", Colors.YELLOW)
            return False
            
    except Exception as e:
        print_status("GPU Check", f"✗ Error: {str(e)}", Colors.RED)
        return False

def check_environment_variables() -> bool:
    """Check required environment variables"""
    required_vars = [
        "GROQ_API_KEY",
    ]
    
    optional_vars = [
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "HF_TOKEN"
    ]
    
    print(f"\n{Colors.BOLD}Checking Environment Variables:{Colors.ENDC}")
    
    all_good = True
    
    for var in required_vars:
        if os.getenv(var):
            print_status(f"  {var}", "✓ Set", Colors.GREEN)
        else:
            print_status(f"  {var}", "✗ Missing (Required)", Colors.RED)
            all_good = False
    
    for var in optional_vars:
        if os.getenv(var):
            print_status(f"  {var}", "✓ Set", Colors.GREEN)
        else:
            print_status(f"  {var}", "- Not set (Optional)", Colors.YELLOW)
    
    return all_good

def check_project_structure() -> bool:
    """Check if project directory structure exists"""
    required_dirs = [
        "src",
        "src/data_generation", 
        "src/preprocessing",
        "src/training",
        "src/evaluation",
        "src/api",
        "configs",
        "scripts",
        "tests",
        "docs"
    ]
    
    optional_dirs = [
        "data",
        "models", 
        "logs",
        "checkpoints",
        "notebooks"
    ]
    
    print(f"\n{Colors.BOLD}Checking Project Structure:{Colors.ENDC}")
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_status(f"  {dir_path}/", "✓ Exists", Colors.GREEN)
        else:
            print_status(f"  {dir_path}/", "✗ Missing", Colors.RED)
            all_good = False
    
    for dir_path in optional_dirs:
        if Path(dir_path).exists():
            print_status(f"  {dir_path}/", "✓ Exists", Colors.GREEN)
        else:
            print_status(f"  {dir_path}/", "- Missing (will create)", Colors.YELLOW)
    
    return all_good

def check_config_files() -> bool:
    """Check if configuration files exist and are valid"""
    config_files = [
        ("configs/config.yaml", True),
        ("env.example", True),
        ("requirements.txt", True)
    ]
    
    print(f"\n{Colors.BOLD}Checking Configuration Files:{Colors.ENDC}")
    
    all_good = True
    
    for file_path, required in config_files:
        if Path(file_path).exists():
            print_status(f"  {file_path}", "✓ Exists", Colors.GREEN)
            
            # Validate YAML files
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                try:
                    with open(file_path, 'r') as f:
                        yaml.safe_load(f)
                    print_status(f"  {file_path} (syntax)", "✓ Valid YAML", Colors.GREEN)
                except yaml.YAMLError as e:
                    print_status(f"  {file_path} (syntax)", f"✗ Invalid YAML: {str(e)[:30]}", Colors.RED)
                    if required:
                        all_good = False
        else:
            status = "✗ Missing" if required else "- Missing (Optional)"
            color = Colors.RED if required else Colors.YELLOW
            print_status(f"  {file_path}", status, color)
            if required:
                all_good = False
    
    return all_good

def test_groq_api_connection() -> bool:
    """Test Groq API connection"""
    try:
        import groq
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print_status("Groq API Connection", "✗ No API key", Colors.RED)
            return False
        
        client = Groq(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Hello, this is a test. Please respond with 'API connection successful.'"}
            ],
            max_tokens=50,
            temperature=0
        )
        
        if response and response.choices:
            print_status("Groq API Connection", "✓ Connected", Colors.GREEN)
            return True
        else:
            print_status("Groq API Connection", "✗ No response", Colors.RED)
            return False
            
    except Exception as e:
        print_status("Groq API Connection", f"✗ Error: {str(e)[:30]}", Colors.RED)
        return False

def create_missing_directories():
    """Create missing directories"""
    dirs_to_create = [
        "data", "models", "logs", "checkpoints", "cache"
    ]
    
    print(f"\n{Colors.BOLD}Creating Missing Directories:{Colors.ENDC}")
    
    for dir_path in dirs_to_create:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_status(f"  {dir_path}/", "✓ Created", Colors.GREEN)
            except Exception as e:
                print_status(f"  {dir_path}/", f"✗ Failed: {str(e)[:30]}", Colors.RED)
        else:
            print_status(f"  {dir_path}/", "✓ Already exists", Colors.GREEN)

def main():
    """Main verification function"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("      DeepCoder Environment Verification")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    
    # Core packages to check
    core_packages = [
        "torch", "transformers", "datasets", "accelerate", "peft",
        "groq", "langchain", "langchain_groq",
        "trl", "wandb", "fastapi", "uvicorn", "pydantic",
        "numpy", "pandas", "tqdm", "yaml", "rich"
    ]
    
    # Optional packages
    optional_packages = [
        "unsloth", "matplotlib", "seaborn", "jupyter"
    ]
    
    results = {
        "python_version": check_python_version(),
        "core_packages": all(check_package_installation(core_packages).values()),
        "optional_packages": check_package_installation(optional_packages),
        "gpu_available": check_gpu_availability(),
        "environment_vars": check_environment_variables(), 
        "project_structure": check_project_structure(),
        "config_files": check_config_files(),
    }
    
    # Create missing directories
    create_missing_directories()
    
    # Test API connection if environment vars are set
    if results["environment_vars"]:
        results["groq_api"] = test_groq_api_connection()
    else:
        print_status("Groq API Connection", "- Skipped (no API key)", Colors.YELLOW)
        results["groq_api"] = None
    
    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}Verification Summary:{Colors.ENDC}")
    print("=" * 40)
    
    critical_checks = ["python_version", "core_packages", "environment_vars", "project_structure", "config_files"]
    critical_passed = all(results[check] for check in critical_checks if results[check] is not None)
    
    optional_checks = ["gpu_available", "groq_api"]
    optional_passed = sum(1 for check in optional_checks if results[check] is True)
    optional_total = len([check for check in optional_checks if results[check] is not None])
    
    if critical_passed:
        print_status("Critical Requirements", "✓ All Passed", Colors.GREEN)
    else:
        print_status("Critical Requirements", "✗ Some Failed", Colors.RED)
    
    if optional_total > 0:
        print_status("Optional Requirements", f"✓ {optional_passed}/{optional_total} Passed", Colors.GREEN if optional_passed == optional_total else Colors.YELLOW)
    
    # Recommendations
    print(f"\n{Colors.BOLD}Recommendations:{Colors.ENDC}")
    
    if not results["gpu_available"]:
        print(f"  {Colors.YELLOW}•{Colors.ENDC} GPU not available - training will be slower")
    
    if results["groq_api"] is False:
        print(f"  {Colors.YELLOW}•{Colors.ENDC} Groq API connection failed - check your API key")
    
    missing_optional = [pkg for pkg, installed in results["optional_packages"].items() if not installed]
    if missing_optional:
        print(f"  {Colors.YELLOW}•{Colors.ENDC} Consider installing optional packages: {', '.join(missing_optional)}")
    
    # Final status
    print(f"\n{Colors.BOLD}Overall Status:{Colors.ENDC}")
    if critical_passed:
        print(f"{Colors.GREEN}✓ Environment is ready for DeepCoder development!{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.RED}✗ Environment setup incomplete. Please resolve the issues above.{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 