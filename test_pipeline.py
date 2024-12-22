from data.utils.data_loader import DataLoader
from data.utils.qlib_processor import QlibProcessor
from data.utils.feature_generator import FeatureGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # 1. Create data loader
        logger.info("Initializing DataLoader...")
        loader = DataLoader(
            exchange_id="binance", symbol="BTC/USDT", timeframe="1h"
        )

        # 2. Fetch some test data
        logger.info("Fetching test data...")
        df = loader.fetch_data("2023-01-01", "2023-01-02")
        logger.info(f"Fetched data shape: {df.shape}")
        logger.info("\nSample data:")
        logger.info(df.head())

        # 3. Process with Qlib
        logger.info("\nProcessing data with Qlib...")
        processor = QlibProcessor(data_dir="data/qlib_data")
        processor.process_data(df, "BTC/USDT")

        # 4. Generate features
        logger.info("\nGenerating technical features...")
        feature_gen = FeatureGenerator()
        df_features = feature_gen.generate_features(df)

        logger.info(f"\nGenerated features shape: {df_features.shape}")
        logger.info("\nFeature columns:")
        logger.info(df_features.columns.tolist())

        # 5. Display some statistics
        logger.info("\nFeature statistics:")
        logger.info(df_features.describe())

        return df_features

    except Exception as e:
        logger.error(f"Error in pipeline test: {str(e)}")
        raise


if __name__ == "__main__":
    df = main()
