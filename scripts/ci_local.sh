#!/bin/bash
# Local CI/CD testing script
# Run this script to test CI/CD workflows locally

set -e

echo "=========================================="
echo "Running Local CI/CD Tests"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if we're in the project root
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
if pip install -q -r requirements.txt flake8 black isort bandit safety; then
    print_status "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Step 2: Run linting
echo ""
echo "Step 2: Running linting checks..."
echo "  - Flake8..."
if flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics; then
    print_status "Flake8 critical checks passed"
else
    print_warning "Flake8 found some issues (non-critical)"
fi

echo "  - Black (formatting check)..."
if black --check .; then
    print_status "Black formatting check passed"
else
    print_warning "Code formatting issues found. Run 'black .' to fix."
fi

echo "  - isort (import sorting)..."
if isort --check-only .; then
    print_status "Import sorting check passed"
else
    print_warning "Import sorting issues found. Run 'isort .' to fix."
fi

# Step 3: Security scanning
echo ""
echo "Step 3: Running security scans..."
echo "  - Bandit..."
if bandit -r . -ll; then
    print_status "Bandit security scan passed"
else
    print_warning "Bandit found some security issues"
fi

echo "  - Safety (dependency check)..."
if safety check; then
    print_status "Safety dependency check passed"
else
    print_warning "Safety found some dependency vulnerabilities"
fi

# Step 4: Run tests
echo ""
echo "Step 4: Running tests..."
if pytest tests/ -v --tb=short; then
    print_status "All tests passed"
else
    print_error "Some tests failed"
    exit 1
fi

# Step 5: Run tests with coverage
echo ""
echo "Step 5: Running tests with coverage..."
if pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html; then
    print_status "Coverage report generated (see htmlcov/index.html)"
else
    print_error "Coverage test failed"
    exit 1
fi

# Step 6: Docker build test (optional)
echo ""
read -p "Do you want to test Docker builds? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Step 6: Testing Docker builds..."
    echo "  - Building API image..."
    if docker build -f Dockerfile.api -t soc-anomaly-api:test .; then
        print_status "API Docker image built successfully"
    else
        print_error "API Docker build failed"
        exit 1
    fi
    
    echo "  - Building Dashboard image..."
    if docker build -f Dockerfile.dashboard -t soc-anomaly-dashboard:test .; then
        print_status "Dashboard Docker image built successfully"
    else
        print_error "Dashboard Docker build failed"
        exit 1
    fi
fi

echo ""
echo "=========================================="
print_status "Local CI/CD tests completed!"
echo "=========================================="
