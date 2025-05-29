#!/bin/bash

# SGLang Server Startup Script for Qwen3 MOE Model
# This script starts the SGLang server with proper configuration and health checks

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/configs/config.yaml"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/sglang_server.pid"
LOG_FILE="$LOG_DIR/sglang_server.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if server is running
is_server_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to wait for server health
wait_for_health() {
    local timeout=${1:-300}  # Default 5 minutes
    local start_time=$(date +%s)
    
    log "Waiting for SGLang server to become healthy..."
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            log "ERROR: Server health check timeout after ${timeout} seconds"
            return 1
        fi
        
        if curl -s -f "http://127.0.0.1:30000/health" > /dev/null 2>&1; then
            log "SGLang server is healthy!"
            return 0
        fi
        
        log "Server not ready yet... (${elapsed}s elapsed)"
        sleep 5
    done
}

# Function to start server
start_server() {
    log "Starting SGLang server for Qwen3 MOE model..."
    
    # Check if server is already running
    if is_server_running; then
        local pid=$(cat "$PID_FILE")
        log "SGLang server is already running (PID: $pid)"
        return 0
    fi
    
    # Verify model exists
    local model_path="/workspace/persistent/models/qwen3-30b-a3b"
    if [ ! -d "$model_path" ]; then
        log "ERROR: Model path not found: $model_path"
        log "Please ensure the Qwen3 model is downloaded to the workspace"
        return 1
    fi
    
    # Check available GPU memory
    if command -v nvidia-smi &> /dev/null; then
        log "GPU Status:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
            log "  GPU: $line"
        done
    fi
    
    # Start SGLang server
    log "Launching SGLang server..."
    
    # Use Python module from the SGLang manager
    cd "$PROJECT_ROOT"
    
    nohup python -m src.sglang_manager start \
        --config "$CONFIG_FILE" \
        --timeout 600 \
        >> "$LOG_FILE" 2>&1 &
    
    local server_pid=$!
    echo $server_pid > "$PID_FILE"
    
    log "SGLang server started with PID: $server_pid"
    
    # Wait for server to become healthy
    if wait_for_health 600; then
        log "SGLang server startup completed successfully"
        
        # Display server info
        log "Server configuration:"
        log "  Host: 127.0.0.1"
        log "  Port: 30000"
        log "  Model: Qwen3-30B-A3B"
        log "  Model Path: $model_path"
        log "  API Endpoint: http://127.0.0.1:30000/v1"
        
        return 0
    else
        log "ERROR: SGLang server failed to become healthy"
        stop_server
        return 1
    fi
}

# Function to stop server
stop_server() {
    log "Stopping SGLang server..."
    
    if ! is_server_running; then
        log "SGLang server is not running"
        return 0
    fi
    
    local pid=$(cat "$PID_FILE")
    log "Stopping SGLang server (PID: $pid)..."
    
    # Try graceful shutdown first
    if kill -TERM "$pid" 2>/dev/null; then
        # Wait for graceful shutdown
        local count=0
        while [ $count -lt 30 ] && ps -p "$pid" > /dev/null 2>&1; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if ps -p "$pid" > /dev/null 2>&1; then
            log "Force killing SGLang server..."
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    
    rm -f "$PID_FILE"
    log "SGLang server stopped"
}

# Function to restart server
restart_server() {
    log "Restarting SGLang server..."
    stop_server
    sleep 5
    start_server
}

# Function to check server status
check_status() {
    if is_server_running; then
        local pid=$(cat "$PID_FILE")
        log "SGLang server is running (PID: $pid)"
        
        # Check health endpoint
        if curl -s -f "http://127.0.0.1:30000/health" > /dev/null 2>&1; then
            log "Server health check: HEALTHY"
        else
            log "Server health check: UNHEALTHY"
        fi
        
        # Get server info
        local server_info=$(curl -s "http://127.0.0.1:30000/v1/models" 2>/dev/null || echo "Unable to fetch server info")
        log "Server info: $server_info"
        
        return 0
    else
        log "SGLang server is not running"
        return 1
    fi
}

# Function to display help
show_help() {
    cat << EOF
SGLang Server Management Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    start       Start the SGLang server
    stop        Stop the SGLang server  
    restart     Restart the SGLang server
    status      Check server status
    logs        Show server logs
    help        Show this help message

Options:
    --config FILE   Path to config file (default: configs/config.yaml)
    --timeout SEC   Startup timeout in seconds (default: 600)

Examples:
    $0 start                    # Start server with default config
    $0 stop                     # Stop server
    $0 restart                  # Restart server
    $0 status                   # Check if server is running
    $0 logs                     # Show recent logs

EOF
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        log "No log file found at $LOG_FILE"
    fi
}

# Main script logic
case "${1:-help}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log "ERROR: Unknown command: $1"
        show_help
        exit 1
        ;;
esac 