#!/bin/bash
# Edge Impulse Training Automation Script
# This script automates what's possible via CLI

set -e

echo "=========================================="
echo "Edge Impulse Training Automation"
echo "=========================================="

# Check if Edge Impulse CLI is installed
if ! command -v edge-impulse &> /dev/null; then
    echo "Installing Edge Impulse CLI..."
    npm install -g edge-impulse-cli
fi

# Check if logged in
if ! edge-impulse whoami &> /dev/null; then
    echo "Please login to Edge Impulse..."
    edge-impulse login
fi

echo ""
echo "✓ Edge Impulse CLI ready"
echo ""

# Show available commands
echo "Available CLI Commands:"
echo "======================"
echo ""
echo "1. List projects:"
echo "   edge-impulse projects list"
echo ""
echo "2. Check data status:"
echo "   edge-impulse data --help"
echo ""
echo "3. Download trained model:"
echo "   edge-impulse download"
echo ""
echo "4. Run model tests:"
echo "   edge-impulse test"
echo ""
echo "5. Deploy model:"
echo "   edge-impulse deploy"
echo ""
echo "=========================================="
echo "IMPORTANT: Training requires web interface"
echo "=========================================="
echo ""
echo "Steps that MUST be done in web interface:"
echo "1. Design Impulse (Image → Transfer Learning → Classification)"
echo "2. Generate Features"
echo "3. Configure Training Settings"
echo "4. Start Training"
echo ""
echo "After training, you can use CLI to:"
echo "- Download model: edge-impulse download"
echo "- Test model: edge-impulse test"
echo "- Deploy model: edge-impulse deploy"
echo ""
echo "Web Interface: https://studio.edgeimpulse.com"
echo ""

