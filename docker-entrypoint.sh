#!/bin/bash
set -e

# Default to sleepbci-serve if no command provided
if [ $# -eq 0 ]; then
    echo "No command provided, starting API server..."
    exec sleepbci-serve
fi

# If command starts with sleepbci-, execute it directly
if [[ "$1" == sleepbci-* ]]; then
    echo "Executing command: $@"
    exec "$@"
fi

# Otherwise, treat as shell command
echo "Executing shell command: $@"
exec "$@"
