#!/usr/bin/env bash

set -e

# Source message handler functions
source ci/scripts/handle_messages.sh

# Function to get all Python files
get_python_files() {
    find . -type f -name "*.py" -not -path "./.sandbox/*"
}

# Main script
info "Sorting imports with isort"
python_files=$(get_python_files)
if [ -z "$python_files" ]; then
    info "No Python files to format"
    exit 0
fi
isort $python_files

info "Performing autopep8 on files"
autopep8 --in-place $python_files

info "Performing selective aggressive autopep8"
autopep8 --global-config ci/.aggressive.pep8 --aggressive --in-place $python_files

info "Performing docformatter"
docformatter --wrap-summaries 88 --wrap-descriptions 88 --in-place $python_files

info "Removing unused imports and variables"
autoflake --in-place --ignore-init-module-imports --remove-all-unused-imports --remove-unused-variables $python_files

info "Formatting completed"
