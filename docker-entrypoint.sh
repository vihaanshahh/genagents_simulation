#!/bin/bash
# Docker entrypoint script for GenAgents simulation

set -e

# Function to print colored output
print_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# Check if AWS_BEARER_TOKEN_BEDROCK is set
if [ -z "$AWS_BEARER_TOKEN_BEDROCK" ]; then
    print_error "AWS_BEARER_TOKEN_BEDROCK environment variable is not set"
    print_info "Please set it using: docker run -e AWS_BEARER_TOKEN_BEDROCK=your_token ..."
    exit 1
fi

print_success "Environment configured"
print_info "LLM Version: ${LLM_VERS:-gpt-oss-120b}"
print_info "Debug Mode: ${DEBUG:-False}"

# Execute the command
print_info "Starting GenAgents simulation..."
exec "$@"
