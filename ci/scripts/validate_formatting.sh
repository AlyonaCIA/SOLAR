#!/usr/bin/env bash

set -e

# Source message handler functions
source ci/scripts/handle_messages.sh

# Function to get all Python files
get_python_files() {
    find . -type f -name "*.py" -not -path "./.sandbox/*"
}

# Function to check formatting with autopep8
check_autopep8() {
    info "Checking formatting with autopep8"
    files=$(get_python_files)
    if [ -z "$files" ]; then
        info "No Python files to check"
        return
    fi
    echo "Processing files: $files"
    if autopep8 --diff $files --exit-code; then
        info "Autopep8 passed!"
    else
        error "Autopep8 did not pass"
        exit 2
    fi
}

# Function to check doc formats with docformatter
check_docformatter() {
    info "Checking doc formats"
    files=$(get_python_files)
    if [ -z "$files" ]; then
        info "No Python files to check"
        return
    fi
    echo "Processing files: $files"
    if docformatter --wrap-summaries 88 --wrap-descriptions 88 --check $files; then
        info "Docformatter passed!"
    else
        error "Docformatter did not pass"
        exit 2
    fi
}

# Function to check for unused imports and variables with autoflake
check_autoflake() {
    info "Checking for unused imports and variables with autoflake"
    files=$(get_python_files)
    if [ -z "$files" ]; then
        info "No Python files to check"
        return
    fi
    echo "Processing files: $files"
    if autoflake --check --remove-all-unused-imports --ignore-init-module-imports $files; then
        info "Autoflake passed!"
    else
        error "Autoflake did not pass"
        exit 2
    fi
}

# Main script
check_autopep8
check_docformatter
check_autoflake
