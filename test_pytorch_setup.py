#!/usr/bin/env python3
"""
Quick test to verify PyTorch setup for agentic training
"""

import sys
import json

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   📱 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   🔥 GPU count: {torch.cuda.device_count()}")
            print(f"   💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError as e:
        print(f"   ❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"   ✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"   ❌ Transformers: {e}")
        return False
    
    try:
        import peft
        print(f"   ✅ PEFT (LoRA)")
    except ImportError as e:
        print(f"   ❌ PEFT: {e}")
        return False
    
    try:
        import plotly
        print(f"   ✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"   ❌ Plotly: {e}")
        return False
    
    try:
        from groq import Groq
        print(f"   ✅ Groq client")
    except ImportError as e:
        print(f"   ❌ Groq: {e}")
        return False
    
    return True

def test_data_loading():
    """Test if training data exists"""
    print("\n📂 Testing data availability...")
    
    train_data = "data/training_data/agentic_train.jsonl"
    val_data = "data/training_data/agentic_val.jsonl"
    
    try:
        with open(train_data, 'r') as f:
            train_count = sum(1 for line in f if line.strip())
        print(f"   ✅ Training data: {train_count} conversations")
        
        with open(val_data, 'r') as f:
            val_count = sum(1 for line in f if line.strip())
        print(f"   ✅ Validation data: {val_count} conversations")
        
        return True
        
    except FileNotFoundError as e:
        print(f"   ❌ Data not found: {e}")
        return False

def test_model_loading():
    """Test basic model loading"""
    print("\n🤖 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        print(f"   📥 Loading tokenizer: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   ✅ Tokenizer loaded")
        
        # Test tokenization
        test_text = "Hello, how are you?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   ✅ Tokenization test: '{test_text}' -> {tokens['input_ids'].shape[1]} tokens")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

def test_groq_connection():
    """Test Groq API connection"""
    print("\n🌐 Testing Groq API connection...")
    
    try:
        from groq import Groq
        
        groq_api_key = "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"
        client = Groq(api_key=groq_api_key)
        
        # Test simple request
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"   ✅ Groq API working")
        print(f"   💬 Test response: {response.choices[0].message.content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Groq API failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 PyTorch Agentic Training Setup Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_data_loading,
        test_model_loading,
        test_groq_connection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for agentic training!")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 