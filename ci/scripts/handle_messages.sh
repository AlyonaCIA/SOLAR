#!/usr/bin/env bash

# Check if tput is available
if ! command -v tput &> /dev/null; then
    echo "[ERROR] tput not found, colors and styles won't be applied."
    exit 1
fi

# Set the terminal type to support colors
export TERM=xterm-256color

# Define ANSI escape codes for colors and styles using tput
bold=$(tput bold)
red=$(tput setaf 1)
green=$(tput setaf 2)
yellow=$(tput setaf 3)
reset=$(tput sgr0)

# Function to print messages with specified color
# Arguments:
#   $1: Color code
#   $2: Message type (INFO, WARNING, ERROR)
#   $3: The actual message to be printed
print_message() {
    local color=$1
    local type=$2
    local message=$3
    echo -e "${bold}${color}[${type}]${reset} ${message}"
}

# Function to print info message
# Arguments:
#   $1: The info message to be printed
info() {
    print_message "$green" "INFO" "$1"
}

# Function to print warning message
# Arguments:
#   $1: The warning message to be printed
warning() {
    print_message "$yellow" "WARNING" "$1"
}

# Function to print error message
# Arguments:
#   $1: The error message to be printed
# Outputs the message to stderr
error() {
    print_message "$red" "ERROR" "$1" >&2
}

# Function to handle errors
error_exit() {
    error "$1"
    exit 1
}
