#!/usr/bin/env python3
"""
Workspace Persistence Checker

This script verifies what's saved in the persistent workspace volume
and provides a checklist for recreating the environment.
"""

import os
import json
from pathlib import Path
from datetime import datetime

def check_file_exists(path, description):
    """Check if a file or directory exists"""
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def get_directory_size(path):
    """Get directory size in human readable format"""
    if not Path(path).exists():
        return "N/A"
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except (OSError, IOError):
                pass
    
    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if total_size < 1024.0:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024.0
    return f"{total_size:.1f} PB"

def main():
    print("🔍 WORKSPACE PERSISTENCE VERIFICATION")
    print("=" * 50)
    print(f"Checked at: {datetime.now().isoformat()}")
    print()
    
    # Check if symlink exists
    print("🔗 SYMLINK STATUS")
    print("-" * 20)
    if Path("/DeepCoder").is_symlink():
        target = os.readlink("/DeepCoder")
        print(f"✅ /DeepCoder -> {target}")
    elif Path("/DeepCoder").exists():
        print("⚠️  /DeepCoder exists but is not a symlink (may be lost on termination)")
    else:
        print("❌ /DeepCoder does not exist")
    print()
    
    # 1. Check project files in PERSISTENT storage
    print("📁 PROJECT FILES (PERSISTENT STORAGE)")
    print("-" * 40)
    project_files = [
        ("/workspace/persistent/DeepCoder", "Main project directory (PERSISTENT)"),
        ("/workspace/persistent/DeepCoder/src", "Source code directory"),
        ("/workspace/persistent/DeepCoder/data", "Data directory"),
        ("/workspace/persistent/DeepCoder/configs", "Configuration files"),
        ("/workspace/persistent/DeepCoder/scripts", "Scripts directory"),
        ("/workspace/persistent/DeepCoder/requirements.txt", "Requirements file"),
        ("/workspace/persistent/DeepCoder/README.md", "Project README"),
    ]
    
    all_files_exist = True
    for path, desc in project_files:
        exists = check_file_exists(path, desc)
        all_files_exist = all_files_exist and exists
    
    # Check project size
    if Path("/workspace/persistent/DeepCoder").exists():
        project_size = get_directory_size("/workspace/persistent/DeepCoder")
        print(f"📊 Project size: {project_size}")
    print()
    
    # 2. Check model files
    print("🤖 MODEL FILES (PERSISTENT STORAGE)")
    print("-" * 40)
    model_files = [
        ("/workspace/persistent/models", "Models directory"),
        ("/workspace/persistent/models/qwen3-30b-a3b", "Qwen model directory"),
        ("/workspace/persistent/models/qwen3-30b-a3b/config.json", "Model config"),
        ("/workspace/persistent/models/qwen3-30b-a3b/tokenizer.json", "Tokenizer"),
        ("/workspace/persistent/models/qwen3-30b-a3b/model.safetensors.index.json", "Model index"),
    ]
    
    models_exist = True
    for path, desc in model_files:
        exists = check_file_exists(path, desc)
        models_exist = models_exist and exists
    
    # Check model size
    model_size = get_directory_size("/workspace/persistent/models/qwen3-30b-a3b")
    print(f"📊 Qwen model size: {model_size}")
    print()
    
    # 3. Check cache directories
    print("💾 CACHE DIRECTORIES")
    print("-" * 20)
    cache_dirs = [
        ("/root/.cache/huggingface", "HuggingFace cache (TEMPORARY)"),
        ("/workspace/persistent/models/qwen3-30b-a3b/.cache", "Model cache"),
    ]
    
    for path, desc in cache_dirs:
        exists = check_file_exists(path, desc)
        if exists:
            size = get_directory_size(path)
            print(f"   Size: {size}")
    
    print()
    
    # 4. Check data collection outputs
    print("📊 DATA COLLECTION OUTPUTS")
    print("-" * 30)
    data_dirs = [
        ("/workspace/persistent/DeepCoder/data/collected", "Collected data directory"),
        ("/workspace/persistent/DeepCoder/data/collection", "Collection modules"),
        ("/workspace/persistent/DeepCoder/data/sources", "Data sources"),
    ]
    
    for path, desc in data_dirs:
        exists = check_file_exists(path, desc)
        if exists:
            size = get_directory_size(path)
            print(f"   Size: {size}")
    
    print()
    
    # 5. Environment check
    print("🔧 ENVIRONMENT VARIABLES")
    print("-" * 20)
    env_vars = [
        "GROQ_API_KEY",
        "TOGETHER_API_KEY", 
        "HUGGINGFACE_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        display_value = f"{value[:10]}..." if value and len(value) > 10 else (value or "Not set")
        print(f"{status} {var}: {display_value}")
    
    print()
    
    # 6. Summary and recommendations
    print("📋 PERSISTENCE SUMMARY")
    print("-" * 20)
    
    if all_files_exist:
        print("✅ All project files are present in PERSISTENT storage")
    else:
        print("❌ Some project files are missing from persistent storage")
    
    if models_exist:
        print("✅ Qwen model files are properly saved")
        print(f"   Total model size: {model_size}")
    else:
        print("❌ Qwen model files are incomplete or missing")
    
    # Critical paths that MUST be in persistent storage
    critical_paths = [
        "/workspace/persistent/models/qwen3-30b-a3b",
        "/workspace/persistent/DeepCoder"
    ]
    
    print()
    print("🚨 CRITICAL PATHS IN PERSISTENT STORAGE")
    print("-" * 40)
    for path in critical_paths:
        exists = Path(path).exists()
        status = "✅" if exists else "❌"
        size = get_directory_size(path) if exists else "N/A"
        print(f"{status} {path} ({size})")
    
    print()
    print("💡 RECREATION INSTRUCTIONS")
    print("-" * 25)
    print("To recreate this environment on a new RunPod:")
    print("1. ✅ Mount /workspace/persistent volume (contains models AND project)")
    print("2. ✅ Run setup script: bash /workspace/persistent/DeepCoder/scripts/setup_environment.sh")
    print("3. ⚠️  Set environment variables (API keys)")
    print("4. ✅ Verify setup: python scripts/collect_data.py --validate-setup")
    
    print()
    total_persistent_size = get_directory_size("/workspace/persistent")
    print(f"💾 Total persistent storage used: {total_persistent_size}")
    
    if models_exist and all_files_exist:
        print("🎉 WORKSPACE IS FULLY PERSISTENT! You can safely recreate RunPod.")
        print("   Both project code and models are in persistent storage.")
    else:
        print("⚠️  WORKSPACE HAS ISSUES - Address missing components before recreating.")

if __name__ == "__main__":
    main() 