"""Tests for ticker extraction."""

import pytest
import pandas as pd

from src.data.ticker_extractor import TickerExtractor


class TestTickerExtractor:
    """Tests for TickerExtractor class."""

    def test_init(self):
        """Test initialization."""
        extractor = TickerExtractor()
        assert len(extractor.FALSE_POSITIVES) > 0

    def test_extract_dollar_tickers(self):
        """Test extracting $TICKER format."""
        extractor = TickerExtractor()

        text = "I'm buying $AAPL and $TSLA today!"
        tickers = extractor.extract_dollar_tickers(text)

        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_extract_dollar_tickers_filters_false_positives(self):
        """Test that false positives are filtered."""
        extractor = TickerExtractor()

        text = "Just did my $DD on the $CEO announcement"
        tickers = extractor.extract_dollar_tickers(text)

        assert "DD" not in tickers
        assert "CEO" not in tickers

    def test_extract_uppercase_tickers(self):
        """Test extracting uppercase tickers."""
        extractor = TickerExtractor()
        extractor.add_tickers(["AAPL", "TSLA", "GME"])

        text = "AAPL is going up but THE market is down"
        tickers = extractor.extract_uppercase_tickers(text, require_validation=True)

        assert "AAPL" in tickers
        assert "THE" not in tickers

    def test_is_valid_ticker(self):
        """Test ticker validation."""
        extractor = TickerExtractor()

        # False positives should be invalid
        assert not extractor.is_valid_ticker("CEO")
        assert not extractor.is_valid_ticker("DD")
        assert not extractor.is_valid_ticker("YOLO")

        # Valid tickers (when no list provided, basic validation)
        assert extractor.is_valid_ticker("AAPL")
        assert extractor.is_valid_ticker("MSFT")

    def test_is_valid_ticker_with_list(self):
        """Test validation against a list."""
        extractor = TickerExtractor()
        extractor.add_tickers(["AAPL", "TSLA"])

        assert extractor.is_valid_ticker("AAPL")
        assert not extractor.is_valid_ticker("XXXX")  # Not in list

    def test_extract_tickers_combined(self):
        """Test combined extraction."""
        extractor = TickerExtractor()
        extractor.add_tickers(["AAPL", "TSLA", "GME"])

        text = "I'm buying $AAPL and looking at TSLA, also GME"
        tickers = extractor.extract_tickers(text, include_uppercase=True)

        assert "AAPL" in tickers  # From $AAPL
        assert "TSLA" in tickers  # From uppercase
        assert "GME" in tickers   # From uppercase

    def test_extract_from_post(self):
        """Test extracting from post title and body."""
        extractor = TickerExtractor()

        title = "Why $AAPL is the best stock"
        selftext = "Here's my analysis on $TSLA too..."

        tickers = extractor.extract_from_post(title, selftext)

        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_count_tickers(self):
        """Test counting ticker mentions."""
        extractor = TickerExtractor()

        df = pd.DataFrame({
            "title": ["$AAPL is great", "$AAPL again", "$TSLA today"],
            "selftext": ["", "", ""],
        })

        counts = extractor.count_tickers(df)

        assert len(counts) == 2
        # AAPL should have 2 mentions
        aapl_count = counts[counts["ticker"] == "AAPL"]["count"].iloc[0]
        assert aapl_count == 2

    def test_add_ticker_column(self):
        """Test adding tickers column to DataFrame."""
        extractor = TickerExtractor()

        df = pd.DataFrame({
            "title": ["Buying $AAPL", "Selling $TSLA"],
            "selftext": ["", ""],
        })

        result = extractor.add_ticker_column(df)

        assert "tickers" in result.columns
        assert "AAPL" in result.iloc[0]["tickers"]
        assert "TSLA" in result.iloc[1]["tickers"]

    def test_empty_text(self):
        """Test extracting from empty text."""
        extractor = TickerExtractor()

        tickers = extractor.extract_tickers("")
        assert tickers == []

        tickers = extractor.extract_tickers(None)
        assert tickers == []

    def test_case_sensitivity(self):
        """Test case handling."""
        extractor = TickerExtractor()

        # $ticker should be converted to uppercase
        text = "$aapl and $Tsla"
        tickers = extractor.extract_dollar_tickers(text)

        assert "AAPL" in tickers
        assert "TSLA" in tickers
