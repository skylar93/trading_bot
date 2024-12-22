import ccxt
import pandas as pd


def test_basic_setup():
    """Test if basic imports and setup are working"""
    print("\n=== Testing Basic Setup ===")

    # Test pandas import and basic operation
    df = pd.DataFrame({"A": [1, 2, 3]})
    print("Pandas DataFrame creation: ✅")

    # Test ccxt import and basic exchange list
    exchanges = ccxt.exchanges
    print(f"Available exchanges: {len(exchanges)}")
    print("CCXT import test: ✅")

    print("\nAll basic tests passed! ✅")


if __name__ == "__main__":
    test_basic_setup()
