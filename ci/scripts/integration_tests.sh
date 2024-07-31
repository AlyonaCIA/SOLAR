#!/usr/bin/env bash

set -e

# Source message handler functions
source ci/scripts/handle_messages.sh

# Function to handle errors
error_exit() {
    error "$1"
    exit 1
}

# Function to run pytest with integration tests
run_pytest() {
    info "Running integration tests with pytest"
    if [ -d "test/integration_test" ] && [ "$(ls -A test/integration_test)" ]; then
        pytest --junitxml=junit/test-results-integration.xml \
               --disable-warnings \
               --tb=short \
               --durations=10 \
               test/integration_test/ || error_exit "Integration tests failed"
    else
        error "No integration tests found"
        exit 1
    fi
}

# Function to generate coverage report
generate_coverage_report() {
    info "Generating coverage report"
    coverage html
    coverage report || error_exit "Coverage report generation failed"
}

# Main script
info "Starting integration tests"

# Create directory for junit reports if it doesn't exist
mkdir -p junit

run_pytest
generate_coverage_report

info "Integration tests and coverage report completed successfully"
