@echo off
REM Local CI/CD testing script for Windows
REM Run this script to test CI/CD workflows locally

echo ==========================================
echo Running Local CI/CD Tests
echo ==========================================
echo.

REM Check if we're in the project root
if not exist "requirements.txt" (
    echo [ERROR] Please run this script from the project root directory
    exit /b 1
)

REM Step 1: Install dependencies
echo Step 1: Installing dependencies...
pip install -q -r requirements.txt flake8 black isort bandit safety
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Step 2: Run linting
echo Step 2: Running linting checks...
echo   - Flake8...
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
if errorlevel 1 (
    echo [WARNING] Flake8 found some issues
) else (
    echo [OK] Flake8 critical checks passed
)

echo   - Black (formatting check)...
black --check .
if errorlevel 1 (
    echo [WARNING] Code formatting issues found. Run 'black .' to fix.
) else (
    echo [OK] Black formatting check passed
)

echo   - isort (import sorting)...
isort --check-only .
if errorlevel 1 (
    echo [WARNING] Import sorting issues found. Run 'isort .' to fix.
) else (
    echo [OK] Import sorting check passed
)
echo.

REM Step 3: Security scanning
echo Step 3: Running security scans...
echo   - Bandit...
bandit -r . -ll
if errorlevel 1 (
    echo [WARNING] Bandit found some security issues
) else (
    echo [OK] Bandit security scan passed
)

echo   - Safety (dependency check)...
safety check
if errorlevel 1 (
    echo [WARNING] Safety found some dependency vulnerabilities
) else (
    echo [OK] Safety dependency check passed
)
echo.

REM Step 4: Run tests
echo Step 4: Running tests...
pytest tests/ -v --tb=short
if errorlevel 1 (
    echo [ERROR] Some tests failed
    exit /b 1
)
echo [OK] All tests passed
echo.

REM Step 5: Run tests with coverage
echo Step 5: Running tests with coverage...
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
if errorlevel 1 (
    echo [ERROR] Coverage test failed
    exit /b 1
)
echo [OK] Coverage report generated (see htmlcov\index.html)
echo.

REM Step 6: Docker build test (optional)
set /p docker_test="Do you want to test Docker builds? (y/n): "
if /i "%docker_test%"=="y" (
    echo Step 6: Testing Docker builds...
    echo   - Building API image...
    docker build -f Dockerfile.api -t soc-anomaly-api:test .
    if errorlevel 1 (
        echo [ERROR] API Docker build failed
        exit /b 1
    )
    echo [OK] API Docker image built successfully
    
    echo   - Building Dashboard image...
    docker build -f Dockerfile.dashboard -t soc-anomaly-dashboard:test .
    if errorlevel 1 (
        echo [ERROR] Dashboard Docker build failed
        exit /b 1
    )
    echo [OK] Dashboard Docker image built successfully
)

echo.
echo ==========================================
echo [OK] Local CI/CD tests completed!
echo ==========================================
