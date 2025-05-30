# Synthetic Agentic Data Generation Guide

## 🚀 Overview

The synthetic data generation system creates high-quality agentic conversation data using **free/low-cost API providers**. This system generates realistic conversations that demonstrate:

- ✅ **Tool usage and function calling**
- ✅ **Multi-step reasoning chains** 
- ✅ **Error handling and debugging**
- ✅ **Problem decomposition**
- ✅ **Code analysis and assistance**
- ✅ **Data analysis workflows**
- ✅ **API integration patterns**

## 🆓 Free API Providers Supported

| Provider | Free Tier | Speed | Quality | Best For |
|----------|-----------|-------|---------|----------|
| **Groq** ⭐ | Generous limits | Ultra-fast | High | Recommended choice |
| **Together AI** | $1 free credit | Fast | High | Good alternative |
| **HuggingFace** | Free tier | Medium | Good | Fallback option |

## 📋 Setup Instructions

### 1. Get a Free API Key

Choose one of these providers and sign up for a free account:

#### Option A: Groq (Recommended) ⭐
- **Signup**: https://console.groq.com/
- **Features**: Ultra-fast inference, generous free limits
- **Model**: Llama-3-8B (fast and high-quality)

#### Option B: Together AI
- **Signup**: https://api.together.xyz/
- **Features**: $1 free credit, good performance
- **Model**: Meta-Llama-3-8B-Chat

#### Option C: HuggingFace
- **Signup**: https://huggingface.co/settings/tokens
- **Features**: Free tier with usage limits
- **Model**: DialoGPT-Medium

### 2. Set Environment Variable

Set your API key as an environment variable:

```bash
# For Groq (recommended)
export GROQ_API_KEY='your_groq_api_key_here'

# OR for Together AI
export TOGETHER_API_KEY='your_together_api_key_here'

# OR for HuggingFace
export HUGGINGFACE_API_KEY='your_huggingface_token_here'
```

### 3. Verify Setup

```bash
python scripts/collect_data.py --validate-setup
```

## 🎯 Usage Examples

### Basic Synthetic Data Generation

Generate 50 synthetic conversations:

```bash
python scripts/collect_data.py --sources synthetic --max-items 50
```

### Specific Scenarios

Generate data for specific agentic scenarios:

```bash
python scripts/collect_data.py --sources synthetic --max-items 30 \
  --datasets tool_usage error_handling code_debugging
```

### High-Volume Generation

Generate large amounts of training data:

```bash
python scripts/collect_data.py --sources synthetic --max-items 500
```

## 📊 Scenario Types Available

### 1. **Tool Usage** 🔧
- API integration examples
- Function calling patterns
- Tool selection reasoning
- **Example**: "Help me analyze sales data from multiple CSV files"

### 2. **Multi-Step Reasoning** 🧠  
- Complex problem decomposition
- Logical reasoning chains
- Step-by-step solutions
- **Example**: "Plan a comprehensive marketing strategy for a new product launch"

### 3. **Error Handling** 🐛
- Debugging approaches
- Error diagnosis
- Solution implementation
- **Example**: "My API is returning 500 errors intermittently"

### 4. **Code Debugging** 💻
- Code analysis
- Bug identification
- Performance optimization
- **Example**: "This Python function is not returning the expected results"

### 5. **Data Analysis** 📈
- Data exploration
- Statistical analysis
- Insight generation
- **Example**: "Analyze customer churn patterns in our subscription data"

### 6. **Web Research** 🔍
- Information gathering
- Source evaluation
- Research synthesis
- **Example**: "Research competitors in the AI automation space"

### 7. **API Integration** 🔗
- API exploration
- Integration planning
- Implementation guidance
- **Example**: "Integrate Stripe payment processing into our application"

### 8. **Problem Solving** 🎯
- Analytical thinking
- Option evaluation
- Solution recommendation
- **Example**: "Our team is struggling with remote collaboration efficiency"

## ⚙️ Configuration Options

The system uses `configs/data_collection.yaml` for configuration:

```yaml
synthetic_generation:
  preferred_provider: 'groq'  # groq, together, huggingface
  default_scenarios:
    - 'tool_usage'
    - 'multi_step_reasoning'
    - 'error_handling'
    - 'code_debugging'
    - 'data_analysis'
    - 'api_integration'
  max_items_per_scenario: 100
  max_turns_per_conversation: 6
  temperature: 0.8
  require_tool_usage: true
  require_reasoning: true
```

## 📁 Output Format

Generated conversations are saved as JSONL files in `data/collected/synthetic/`:

```json
{
  "source": "synthetic",
  "item_id": "tool_usage_0_1",
  "content": {
    "conversation_type": "synthetic_tool_usage",
    "turns": [
      {
        "role": "user",
        "content": "Help me analyze sales data from multiple CSV files"
      },
      {
        "role": "assistant", 
        "content": "I'll help you analyze your sales data step by step...",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "data_processor",
              "description": "Process and analyze datasets",
              "arguments": {
                "data_file": "example_data_file",
                "operation": "example_operation"
              }
            }
          }
        ]
      }
    ],
    "agentic_patterns": ["tool_calling", "step_by_step_reasoning"],
    "domain": "tool_usage",
    "complexity": "medium",
    "available_tools": [...]
  },
  "quality_score": 0.85,
  "timestamp": "2025-05-29T23:45:00Z",
  "metadata": {
    "scenario_type": "tool_usage",
    "provider": "groq",
    "model": "llama3-8b-8192"
  }
}
```

## 🎛️ Advanced Usage

### Custom Configuration

Create a custom config file:

```python
from data.collection.synthetic_generator import SyntheticConfig, collect_synthetic_data
from data.collection.base_collector import CollectionConfig

# Custom synthetic config
synthetic_config = SyntheticConfig(
    provider="groq",
    api_key="your_api_key",
    model_name="llama3-8b-8192",
    scenario_types=["tool_usage", "error_handling"],
    max_turns_per_conversation=8,
    temperature=0.7,
    require_tool_usage=True
)

# Run collection
collection_config = CollectionConfig(output_dir="custom_output")
result = await collect_synthetic_data(collection_config, synthetic_config, max_items=100)
```

### Quality Filtering

The system automatically filters conversations based on quality metrics:

- **Tool usage presence** (20% weight)
- **Agentic patterns** (40% weight) 
- **Content length** (20% weight)
- **Conversation complexity** (10% weight)
- **Multi-turn structure** (10% weight)

## 🔍 Quality Assessment

Each generated conversation receives a quality score (0.0-1.0):

- **0.8-1.0**: High quality - Rich agentic patterns, tool usage, complex reasoning
- **0.5-0.8**: Medium quality - Some agentic patterns, moderate complexity
- **0.0-0.5**: Low quality - Basic conversations, filtered out by default

## 📈 Performance Tips

### 1. **Groq for Speed** ⚡
Groq offers the fastest inference (~275 tokens/second) with generous free limits.

### 2. **Batch Processing** 📦
Generate data in batches rather than one-by-one:

```bash
# Generate multiple smaller batches
for i in {1..5}; do
  python scripts/collect_data.py --sources synthetic --max-items 100
done
```

### 3. **Scenario Focus** 🎯
Focus on specific scenarios relevant to your use case:

```bash
python scripts/collect_data.py --sources synthetic --max-items 200 \
  --datasets tool_usage api_integration
```

## 🚨 Troubleshooting

### API Key Issues

```bash
# Verify your API key is set
echo $GROQ_API_KEY

# Test the setup
python scripts/collect_data.py --validate-setup
```

### Rate Limiting

If you hit rate limits:

1. **Reduce rate limiting** in config:
   ```yaml
   data_collection:
     rate_limit_per_second: 1.0  # Slower requests
   ```

2. **Use smaller batches**:
   ```bash
   python scripts/collect_data.py --sources synthetic --max-items 20
   ```

3. **Switch providers** if one is rate-limited

### Low Quality Output

If conversations are too simple:

1. **Increase temperature**:
   ```yaml
   synthetic_generation:
     temperature: 0.9  # More creative responses
   ```

2. **Use better models**:
   - Groq: `llama3-70b-8192` (if available)
   - Together: `meta-llama/Llama-3-70b-chat-hf`

## 📊 Expected Output

With a good API key, expect:

- **Generation Rate**: 10-50 conversations/minute (depending on provider)
- **Quality Distribution**: ~70% high-quality, ~25% medium, ~5% filtered out
- **Agentic Patterns**: 80%+ conversations include tool usage or reasoning
- **Average Length**: 300-800 tokens per conversation

## 🎉 Success Indicators

You'll know the system is working well when you see:

✅ **High-quality conversations** with clear reasoning chains  
✅ **Realistic tool usage** patterns and function calls  
✅ **Diverse scenarios** covering multiple agentic behaviors  
✅ **Consistent format** suitable for training agentic models  
✅ **Good throughput** (multiple conversations per minute)

## 🔄 Next Steps

After generating synthetic data:

1. **Combine with real data** from HuggingFace datasets
2. **Process and validate** using the data processing pipeline
3. **Train your agentic model** using the enhanced training pipeline
4. **Evaluate performance** on agentic benchmarks

---

**Ready to generate high-quality agentic training data? Get your free API key and start creating!** 🚀 