#!/usr/bin/env bash

set -e

# Source message handler functions
source ci/scripts/handle_messages.sh

# Function to get all Python files
get_python_files() {
    find . -type f -name "*.py" -not -path "./.sandbox/*"
}

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        error "$1 could not be found. Please install $1 to proceed."
        exit 1
    fi
}

# Main script

# Check necessary commands
check_command isort
check_command autopep8
check_command docformatter
check_command autoflake

info "Sorting imports with isort"
isort $(get_python_files)

info "Performing autopep8 on files"
autopep8 --in-place $(get_python_files)

# Run autopep8 with some aggressive parameters enabled
info "Performing selective aggressive autopep8"
autopep8 --global-config ${CI_SCRIPTS_PATH}/.aggressive.pep8 \
         --aggressive \
         --in-place $(get_python_files)

info "Performing docformatter"
docformatter --wrap-summaries 88 \
             --wrap-descriptions 88 \
             --in-place $(get_python_files)

info "Removing unused imports and variables"
autoflake --in-place \
          --ignore-init-module-imports \
          --remove-all-unused-imports \
          --remove-unused-variables $(get_python_files)

info "Formatting completed"
