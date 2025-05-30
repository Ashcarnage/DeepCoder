# Agentic Data Sources Catalog

## 🎯 Overview
This document catalogs high-quality data sources for training agentic behavior in language models, focusing on tool use, reasoning chains, and problem-solving interactions.

## 🏆 **Primary Agentic Datasets (HuggingFace)**

### 1. **ToolBench Dataset**
- **Source**: `ShishirPatil/gorilla-openfunctions-v1`
- **Size**: ~100K function calling examples
- **Quality**: High - real API documentation and usage
- **Focus**: Function calling, API usage, tool selection
- **Format**: JSON with function definitions and call examples

### 2. **WebArena Dataset**  
- **Source**: `webarena/webarena`
- **Size**: ~800 web interaction tasks
- **Quality**: Very High - real web browsing tasks
- **Focus**: Web navigation, form filling, multi-step tasks
- **Format**: Structured traces of web interactions

### 3. **AgentBench Dataset**
- **Source**: `THUDM/AgentBench`
- **Size**: ~3K diverse agentic tasks
- **Quality**: High - curated benchmark tasks
- **Focus**: Multi-domain agent evaluation tasks
- **Format**: Task definitions with expected agent behaviors

### 4. **Tool Learning Dataset**
- **Source**: `qwenlm/qwen-agent-data`
- **Size**: ~50K tool interaction examples
- **Quality**: High - from Qwen team
- **Focus**: Tool selection, reasoning, execution
- **Format**: Conversation format with tool usage

### 5. **ReAct Dataset**
- **Source**: `chenxwh/ReAct`
- **Size**: ~10K reasoning and acting examples
- **Quality**: High - from research paper
- **Focus**: Reasoning chains with action execution
- **Format**: Thought-Action-Observation sequences

## 🌐 **Web Scraping Sources**

### 1. **GitHub Issues & Discussions**
- **Source**: GitHub API
- **Target Repos**: 
  - `microsoft/semantic-kernel`
  - `langchain-ai/langchain`
  - `openai/swarm`
  - `microsoft/autogen`
- **Quality**: Medium-High - real developer problems
- **Focus**: Tool integration, debugging, problem-solving
- **Rate Limit**: 5000 requests/hour (authenticated)

### 2. **Stack Overflow**
- **Source**: Stack Exchange API
- **Tags**: `["agents", "ai-tools", "function-calling", "api-integration"]`
- **Quality**: Medium - community Q&A
- **Focus**: Tool usage problems and solutions
- **Rate Limit**: 300 requests/second

### 3. **Reddit Programming Communities**
- **Source**: Reddit API
- **Subreddits**: 
  - `/r/MachineLearning`
  - `/r/ArtificialIntelligence`
  - `/r/LocalLLaMA`
  - `/r/ChatGPT`
- **Quality**: Medium - community discussions
- **Focus**: AI tool usage, agent experiences
- **Rate Limit**: 100 requests/minute

### 4. **AI Research Forums**
- **Source**: Web scraping
- **Targets**:
  - OpenAI Community Forum
  - Anthropic Discord (public channels)
  - HuggingFace Community
- **Quality**: High - expert discussions
- **Focus**: Advanced agentic patterns, best practices

## 📚 **Research Paper Sources**

### 1. **ArXiv Papers**
- **Source**: ArXiv API
- **Categories**: `cs.AI`, `cs.CL`, `cs.LG`
- **Keywords**: `["agent", "tool use", "function calling", "reasoning"]`
- **Quality**: Very High - academic research
- **Focus**: Novel agentic methodologies, case studies
- **Format**: Full text extraction from PDFs

### 2. **ACL Anthology**
- **Source**: ACL Anthology API
- **Quality**: Very High - peer-reviewed
- **Focus**: NLP agents, dialogue systems
- **Format**: Structured academic papers

### 3. **Google Scholar**
- **Source**: Unofficial API/scraping
- **Quality**: High - academic citations
- **Focus**: Comprehensive agent research
- **Rate Limit**: Careful scraping needed

## 🤖 **Synthetic Data Generation**

### 1. **GPT-4 Generated Conversations**
- **Method**: Prompt engineering for agentic scenarios
- **Quality**: High - controlled generation
- **Focus**: Consistent tool use patterns
- **Volume**: Unlimited (cost-limited)

### 2. **Claude-3 Agent Simulations**
- **Method**: Multi-turn agent role-playing
- **Quality**: High - sophisticated reasoning
- **Focus**: Complex problem-solving scenarios
- **Volume**: Good (API limits)

### 3. **Qwen2.5-Coder Tool Scenarios**
- **Method**: Code-focused agentic tasks
- **Quality**: Medium-High - coding specific
- **Focus**: Programming tool usage
- **Volume**: High (local generation)

## 🏢 **Commercial/API Sources**

### 1. **OpenAI ChatGPT Conversations**
- **Source**: ChatGPT exported conversations
- **Quality**: High - real user interactions
- **Focus**: Tool use, problem-solving
- **Legal**: Check ToS compliance

### 2. **Anthropic Claude Conversations**
- **Source**: Claude conversation exports
- **Quality**: Very High - sophisticated reasoning
- **Focus**: Complex analytical tasks
- **Legal**: Check ToS compliance

### 3. **GitHub Copilot Interactions**
- **Source**: VS Code extension logs (if available)
- **Quality**: High - real coding assistance
- **Focus**: Code generation, tool integration
- **Legal**: Privacy concerns

## 📊 **Data Quality Assessment**

### **High Quality Indicators**
- ✅ Clear tool selection reasoning
- ✅ Multi-step problem decomposition
- ✅ Error handling and recovery
- ✅ Contextual tool usage
- ✅ Complete interaction sequences

### **Medium Quality Indicators**
- ⚠️ Some tool usage examples
- ⚠️ Partial reasoning chains
- ⚠️ Community discussions
- ⚠️ Educational content

### **Low Quality Indicators**
- ❌ Single-turn interactions
- ❌ No tool usage
- ❌ Unclear reasoning
- ❌ Incomplete examples
- ❌ Spam/irrelevant content

## 🎯 **Collection Priorities**

### **Phase 1: Immediate (High ROI)**
1. ToolBench Dataset (HuggingFace)
2. WebArena Dataset (HuggingFace)
3. Qwen Agent Data (HuggingFace)
4. GPT-4 Synthetic Generation

### **Phase 2: Medium-term**
1. GitHub Issues Scraping
2. Stack Overflow API
3. ArXiv Paper Extraction
4. Claude Synthetic Generation

### **Phase 3: Long-term**
1. Reddit Communities
2. Research Forums
3. Commercial Conversation Data
4. Academic Paper Corpus

## 🔧 **Technical Implementation Notes**

### **Data Collection Framework Requirements**
- Rate limiting and respect for ToS
- Data deduplication across sources
- Quality scoring and filtering
- Consistent format conversion
- Progress tracking and resumability
- Error handling and retry logic

### **Storage Format**
```json
{
  "source": "toolbench",
  "conversation_id": "tb_001234",
  "quality_score": 0.85,
  "agentic_patterns": ["tool_selection", "reasoning_chain", "error_handling"],
  "turns": [
    {
      "role": "user",
      "content": "I need to check the weather in Tokyo",
      "timestamp": "2025-01-01T12:00:00Z"
    },
    {
      "role": "assistant", 
      "content": "I'll help you check the weather in Tokyo. Let me use the weather API.",
      "tool_calls": [
        {
          "tool": "weather_api",
          "function": "get_current_weather",
          "arguments": {"location": "Tokyo, Japan"}
        }
      ],
      "reasoning": "User wants current weather data for Tokyo, so I'll use the weather API to get accurate real-time information."
    }
  ]
}
```

## 📈 **Success Metrics**

### **Quantity Targets**
- **Total Conversations**: 100K+
- **High-Quality Examples**: 50K+
- **Tool Use Examples**: 75K+
- **Multi-turn Sequences**: 40K+

### **Quality Targets**
- **Average Quality Score**: >0.7
- **Tool Use Coverage**: 80% of examples
- **Reasoning Chain Coverage**: 60% of examples
- **Error Handling Examples**: 20% of examples

---

**Status**: 📋 Catalog Complete - Ready for Implementation  
**Next Step**: Implement data collection infrastructure  
**Estimated Collection Time**: 2-3 weeks for Phase 1 sources 