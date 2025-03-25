#!/bin/bash

# Run Shape Verification Tests Script
#
# This script runs the small integration tests that verify tensor shapes 
# and check for NaN values in the training pipeline.
#
# Features:
# - Runs individual tests separately to isolate failures
# - Provides clear output with test status
# - Logs detailed information to shape_test_debug.log
# - Can be used for quick validation before running larger tests

set -e

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Running Shape Verification Tests${NC}"
echo "Detailed logs will be written to shape_test_debug.log"
echo "------------------------------------------------------"

# Function to run a specific test and report
run_test() {
    TEST_NAME=$1
    echo -e "Running ${YELLOW}$TEST_NAME${NC}..."
    if python -m pytest tests/test_small_integration.py::$TEST_NAME -v; then
        echo -e "${GREEN}✅ $TEST_NAME passed${NC}"
        return 0
    else
        echo -e "${RED}❌ $TEST_NAME failed${NC}"
        return 1
    fi
}

# Run each test individually to pinpoint issues
TEST_RESULTS=()

# Single agent test
if run_test test_single_agent_shapes; then
    TEST_RESULTS+=("single_agent: PASS")
else
    TEST_RESULTS+=("single_agent: FAIL")
fi

# Multi-agent test
if run_test test_multi_agent_shapes; then
    TEST_RESULTS+=("multi_agent: PASS")
else
    TEST_RESULTS+=("multi_agent: FAIL")
fi

# Meta-agent test (often the most complex)
if run_test test_meta_agent_ensemble; then
    TEST_RESULTS+=("meta_agent: PASS")
else
    TEST_RESULTS+=("meta_agent: FAIL")
fi

# Pipeline tests
if run_test test_train_pipeline_minimal; then
    TEST_RESULTS+=("train_pipeline: PASS")
else
    TEST_RESULTS+=("train_pipeline: FAIL")
fi

if run_test test_multi_agent_train_pipeline_minimal; then
    TEST_RESULTS+=("multi_agent_train_pipeline: PASS")
else
    TEST_RESULTS+=("multi_agent_train_pipeline: FAIL")
fi

# Print summary
echo "------------------------------------------------------"
echo -e "${YELLOW}Test Results Summary:${NC}"
for result in "${TEST_RESULTS[@]}"; do
    if [[ $result == *"PASS"* ]]; then
        echo -e "${GREEN}$result${NC}"
    else
        echo -e "${RED}$result${NC}"
    fi
done

# Show log file location
echo "------------------------------------------------------"
echo "Check shape_test_debug.log for detailed information on tensor shapes and errors"

# Count failures
FAILURES=$(echo "${TEST_RESULTS[@]}" | tr ' ' '\n' | grep -c "FAIL")
if [ $FAILURES -gt 0 ]; then
    echo -e "${RED}$FAILURES test(s) failed. Fix shape issues before proceeding with validate_training.py${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed! Safe to proceed with validate_training.py${NC}"
    exit 0
fi 