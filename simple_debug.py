#!/usr/bin/env python3
"""
Simple debug for conversation generation
"""
import os
import asyncio
from data.collection.synthetic_generator import SyntheticConfig, GroqProvider

async def main():
    # Load env
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except:
        pass
    
    api_key = os.getenv('GROQ_API_KEY')
    print(f"🔑 API Key: {api_key[:10]}...")
    
    config = SyntheticConfig(
        provider="groq",
        api_key=api_key,
        model_name="llama3-8b-8192"
    )
    
    provider = GroqProvider(config)
    
    async with provider:
        # Test the exact conversation generation steps
        
        # 1. Initial messages
        system_prompt = "You are an AI assistant that helps users accomplish tasks by using various tools and APIs."
        user_prompt = "Help me analyze sales data from multiple CSV files"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print("📤 Testing step 1: Initial response...")
        response1 = await provider.generate_completion(messages)
        if response1:
            print(f"✅ Response 1: {response1[:100]}...")
        else:
            print("❌ Failed at step 1")
            return
        
        # 2. Add assistant response and generate follow-up
        messages.append({"role": "assistant", "content": response1})
        messages.append({"role": "user", "content": "Please provide a detailed step-by-step solution."})
        
        print("📤 Testing step 2: Follow-up response...")
        response2 = await provider.generate_completion(messages)
        if response2:
            print(f"✅ Response 2: {response2[:100]}...")
        else:
            print("❌ Failed at step 2")
            return
        
        print("🎉 Conversation generation working!")

if __name__ == "__main__":
    asyncio.run(main()) 