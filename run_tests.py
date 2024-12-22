"""Test Runner Script"""

import pytest
import os
import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_tests(
    test_path: str = "tests",
    markers: str = None,
    verbose: bool = True,
    parallel: bool = True,
    coverage: bool = True,
):
    """Run tests with specified configuration

    Args:
        test_path: Path to test files/directory
        markers: Test markers to run (e.g., "not slow")
        verbose: Whether to show detailed output
        parallel: Whether to run tests in parallel
        coverage: Whether to collect coverage data
    """
    # Build pytest arguments
    args = [test_path]

    if verbose:
        args.append("-v")

    if parallel:
        args.append("-n=auto")

    if markers:
        args.append("-m=" + markers)

    if coverage:
        args.extend(["--cov=training", "--cov-report=html"])

    # Run tests
    logger.info(f"Running tests with args: {' '.join(args)}")
    return pytest.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading bot tests")

    parser.add_argument(
        "--path", default="tests", help="Path to test files/directory"
    )

    parser.add_argument(
        "--markers", help="Test markers to run (e.g., 'not slow')"
    )

    parser.add_argument(
        "--no-verbose", action="store_true", help="Disable verbose output"
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel test execution",
    )

    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage collection",
    )

    args = parser.parse_args()

    exit_code = run_tests(
        test_path=args.path,
        markers=args.markers,
        verbose=not args.no_verbose,
        parallel=not args.no_parallel,
        coverage=not args.no_coverage,
    )

    sys.exit(exit_code)
