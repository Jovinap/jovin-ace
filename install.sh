#!/usr/bin/env bash
# Installation script for Jovin Ace CLI

set -e

# Styling colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}🚀 Starting Jovin Ace CLI installation...${NC}"

# Check requirements
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

if ! command -v git &>/dev/null; then
    echo -e "${RED}Error: Git is required but not installed.${NC}"
    exit 1
fi

INSTALL_DIR="$HOME/.local/share/jovin-ace"
BIN_DIR="$HOME/.local/bin"

# Cleanup previous installation if exists
if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}Existing installation found at $INSTALL_DIR. Updating...${NC}"
    rm -rf "$INSTALL_DIR"
fi

# Clone repository
echo -e "${CYAN}Cloning Jovin Ace repository...${NC}"
git clone --depth 1 https://github.com/jovinap/jovin-ace.git "$INSTALL_DIR"

# Install dependencies
echo -e "${CYAN}Installing Python requirements...${NC}"
if [ -f "$INSTALL_DIR/requirements.txt" ]; then
    # Attempt installing dependencies (using --break-system-packages for system-wide restrictions if necessary)
    python3 -m pip install -r "$INSTALL_DIR/requirements.txt" --break-system-packages || python3 -m pip install -r "$INSTALL_DIR/requirements.txt" || true
fi

# Ensure BIN_DIR exists and link executable
mkdir -p "$BIN_DIR"
chmod +x "$INSTALL_DIR/jovin-ace"
ln -sf "$INSTALL_DIR/jovin-ace" "$BIN_DIR/jovin-ace"

echo -e "\n${GREEN}✓ Jovin Ace CLI successfully installed!${NC}"
echo -e "${CYAN}Executable linked to: ${YELLOW}$BIN_DIR/jovin-ace${NC}"
echo -e "\nMake sure ${YELLOW}$BIN_DIR${NC} is in your system PATH variable."
echo -e "You can test it by running: ${GREEN}jovin-ace status${NC}"
