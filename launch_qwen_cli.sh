#!/bin/bash
# 🤖 Qwen Interactive CLI Launcher
# ================================

echo "🚀 Starting Qwen Interactive CLI..."
echo "📡 Connecting to SGLang server on localhost:30000"
echo "🎯 SAD-enhanced Qwen 30B model ready!"
echo ""

# Check if SGLang server is running
if curl -s http://localhost:30000/health > /dev/null 2>&1; then
    echo "✅ SGLang server is running"
    echo "🔥 Launching interactive CLI..."
    echo ""
    python interactive_qwen_cli.py
else
    echo "❌ SGLang server not detected on localhost:30000"
    echo "💡 Please ensure your SGLang server is running first"
    echo ""
    echo "Start it with:"
    echo "cd /workspace/persistent/models/qwen3-30b-a3b"
    echo "python -m sglang.launch_server --model-path . --host 0.0.0.0 --port 30000 ..."
    exit 1
fi 