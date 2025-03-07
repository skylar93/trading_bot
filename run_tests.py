#!/usr/bin/env python
"""
Test Runner for Multi-Agent Trading System

This script runs all the tests for the multi-agent trading system and generates a detailed report.
It can be used to verify that the system is working correctly and that recent changes haven't
broken any functionality.

Usage:
    python run_tests.py [options]

Options:
    --verbose, -v     Show detailed test output
    --coverage        Generate a coverage report
    --single=FILE     Run a single test file (e.g., test_multi_agent_env.py)
    --help, -h        Show this help message
"""

import sys
import os
import subprocess
import argparse
import time

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run tests for multi-agent trading system")
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed test output')
    parser.add_argument('--coverage', action='store_true', help='Generate a coverage report')
    parser.add_argument('--single', type=str, help='Run a single test file')
    return parser.parse_args()

def main():
    """Main function to run tests"""
    args = parse_args()
    
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Build the command
    cmd = ['pytest']
    
    # Add verbose flag if requested
    if args.verbose:
        cmd.append('-v')
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(['--cov=envs', '--cov=agents', '--cov-report=term', '--cov-report=html'])
    
    # Add specific test file if requested
    if args.single:
        test_file = args.single
        if not test_file.startswith('tests/'):
            test_file = f'tests/{test_file}'
        cmd.append(test_file)
    else:
        cmd.append('tests/')
    
    # Print the command we're about to run
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the tests and measure time
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Tests completed in {elapsed_time:.2f} seconds with exit code: {result.returncode}")
    
    if args.coverage:
        print("\nCoverage report generated in htmlcov/index.html")
    
    # Return the exit code from pytest
    return result.returncode

if __name__ == '__main__':
    sys.exit(main()) 