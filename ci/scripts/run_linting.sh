#!/usr/bin/env bash

set -e

# Source message handler functions
source ci/scripts/handle_messages.sh

# Function to get all Python files
get_python_files() {
    find . -type f -name "*.py" -not -path "./.sandbox/*"
}

# Run Flake8 linter
function run_flake8 {
    info "Running Flake8 linter"
    files=$(get_python_files)
    if [ -z "$files" ]; then
        info "No Python files to lint"
        return
    fi
    for file in $files; do
        if [ -f "$file" ]; then
            echo "Linting file: $file"
            if ! flake8 "$file"; then
                error "Flake8 did not pass for file: $file"
                exit 2
            fi
        else
            echo "[WARNING] File not found: $file"
        fi
    done
    info "Flake8 passed!"
}

# Run pylint on all non-test Python files
function run_pylint {
    info "Performing pylinting skipping tests"
    msg_format='{C}:{line:3d},{column:2d}: {msg} ({msg_id}: {symbol})'
    files=$(get_python_files)
    if [ -z "$files" ]; then
        info "No Python files to lint"
        return
    fi
    for file in $files; do
        if [ -f "$file" ]; then
            echo "Pylinting file: $file"
            if ! pylint "$file" --msg-template="$msg_format" --rcfile=./.pylintrc; then
                error "Pylint did not pass for file: $file"
                exit 2
            fi
        else
            echo "[WARNING] File not found: $file"
        fi
    done
    info "Pylint passed on the code"
}

# Execute the linting functions
run_flake8
run_pylint
