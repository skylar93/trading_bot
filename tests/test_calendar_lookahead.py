"""C4: verify calendar features have no future lookahead by default."""
import pandas as pd
import numpy as np
import pytest


class TestCalendarLookahead:
    def test_default_lookahead_is_zero(self):
        from training.data.calendar_features import CalendarConfig
        cfg = CalendarConfig()
        assert cfg.event_lookahead_days == 0, (
            f"Default lookahead should be 0, got {cfg.event_lookahead_days}"
        )

    def test_no_future_signal_with_zero_lookahead(self):
        from training.data.calendar_features import CalendarFeatureEngine, CalendarConfig
        cfg = CalendarConfig(event_lookahead_days=0)
        engine = CalendarFeatureEngine(cfg)
        # Create index around a known FOMC date
        # Use a date range that includes a future FOMC meeting
        idx = pd.date_range("2024-01-28", periods=5, freq="D", tz="UTC")
        # FOMC meeting on 2024-01-31
        flags = engine._event_flag(idx)
        # Days before the meeting (Jan 28, 29, 30) should have NO positive signal
        for i, d in enumerate(idx):
            if d < pd.Timestamp("2024-01-31", tz="UTC"):
                assert flags.iloc[i] <= 0, (
                    f"Date {d} should have no positive look-ahead signal, got {flags.iloc[i]}"
                )
