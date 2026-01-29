"""Extract stock tickers from Reddit text."""

import re
from pathlib import Path

import pandas as pd


class TickerExtractor:
    """Extract and validate stock tickers from text."""

    # Common false positives to filter out
    FALSE_POSITIVES = {
        # Single letters that are rarely tickers in context
        "A", "I", "U", "R", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M",
        "N", "O", "P", "Q", "S", "T", "V", "W", "X", "Y", "Z",
        # Common acronyms
        "CEO", "CFO", "COO", "CTO", "IPO", "FDA", "SEC", "ETF", "GDP", "CPI",
        "FOMC", "FDIC", "IMF", "FED", "NYSE", "NASDAQ", "DOJ", "EPA", "CDC",
        "WHO", "USA", "UK", "EU", "UN", "NATO", "FBI", "CIA", "NSA", "IRS",
        # Reddit/trading slang
        "DD", "YOLO", "FOMO", "ATH", "ATL", "EOD", "EOM", "EOY", "EOW",
        "ITM", "OTM", "ATM", "IV", "DTE", "OP", "IMO", "IMHO", "TBH",
        "WSB", "HODL", "FD", "LEAPS", "PUTS", "CALLS", "RH", "TDA", "IBKR",
        "PM", "AM", "PT", "EST", "PST", "CST", "UTC", "TA", "FA",
        "TL", "DR", "TLDR", "PSA", "FYI", "BTW", "WTF", "LOL", "LMAO",
        "EPS", "PE", "PB", "ROI", "ROE", "ROA", "DCF", "NPV", "IRR",
        "EBITDA", "EBIT", "FCF", "YOY", "QOQ", "MOM", "TTM", "FY", "Q1", "Q2", "Q3", "Q4",
        # Common words that look like tickers
        "ALL", "ARE", "BIG", "CAN", "DAY", "FOR", "HAS", "HIM", "HIS", "HOW",
        "ITS", "MAY", "NEW", "NOW", "OLD", "ONE", "OUR", "OUT", "OWN", "SAY",
        "SHE", "THE", "TOO", "TWO", "WAY", "WHO", "WHY", "YES", "YET", "YOU",
        "BEST", "CALL", "CASH", "CORP", "EARN", "EVER", "GAIN", "GOOD", "GOLD",
        "GROW", "HIGH", "HOLD", "HOPE", "LIFE", "LONG", "LOSS", "LOVE", "MAKE",
        "MORE", "MUCH", "MUST", "NEXT", "OPEN", "PLAY", "REAL", "RISK", "SAFE",
        "SAME", "SAVE", "SELF", "SELL", "SOME", "STAY", "STOP", "SURE", "TAKE",
        "TEAM", "TELL", "THAN", "THAT", "THEM", "THEN", "THIS", "TIME", "TURN",
        "VERY", "WANT", "WEEK", "WELL", "WHAT", "WHEN", "WILL", "WITH", "WORK",
        "YEAR", "YOUR", "ZERO", "MOON", "YELL", "PUMP", "DUMP", "BEAR", "BULL",
        "BTFD", "FCEL", "ROPE", "GAIN", "PORN", "LOSS",
        # Crypto terms
        "BTC", "ETH", "LTC", "XRP", "DOGE", "SHIB", "NFT", "DEFI", "WEB3",
    }

    # Pattern for $TICKER format
    DOLLAR_TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")

    # Pattern for potential tickers (uppercase 1-5 letters)
    UPPERCASE_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")

    def __init__(self, ticker_list_path: str | None = None):
        """Initialize the ticker extractor.

        Args:
            ticker_list_path: Path to CSV file with valid tickers.
        """
        self.valid_tickers: set[str] = set()

        if ticker_list_path:
            self.load_ticker_list(ticker_list_path)

    def load_ticker_list(self, path: str) -> None:
        """Load valid tickers from a CSV file.

        Args:
            path: Path to CSV file with 'Symbol' or 'ticker' column.
        """
        df = pd.read_csv(path)

        # Try common column names
        for col in ["Symbol", "symbol", "Ticker", "ticker", "SYMBOL", "TICKER"]:
            if col in df.columns:
                self.valid_tickers = set(df[col].str.upper().dropna())
                break

    def add_tickers(self, tickers: list[str]) -> None:
        """Add tickers to the valid set.

        Args:
            tickers: List of ticker symbols to add.
        """
        self.valid_tickers.update(t.upper() for t in tickers)

    def is_valid_ticker(self, ticker: str) -> bool:
        """Check if a ticker is valid.

        Args:
            ticker: Ticker symbol to validate.

        Returns:
            True if valid.
        """
        ticker_upper = ticker.upper()

        # Filter out false positives
        if ticker_upper in self.FALSE_POSITIVES:
            return False

        # If we have a ticker list, validate against it
        if self.valid_tickers:
            return ticker_upper in self.valid_tickers

        # Basic validation: 1-5 uppercase letters
        return bool(re.match(r"^[A-Z]{1,5}$", ticker_upper))

    def extract_dollar_tickers(self, text: str) -> list[str]:
        """Extract tickers in $TICKER format.

        Args:
            text: Text to extract from.

        Returns:
            List of extracted tickers.
        """
        matches = self.DOLLAR_TICKER_PATTERN.findall(text.upper())
        return [m for m in matches if self.is_valid_ticker(m)]

    def extract_uppercase_tickers(
        self,
        text: str,
        require_validation: bool = True,
    ) -> list[str]:
        """Extract potential tickers from uppercase words.

        Args:
            text: Text to extract from.
            require_validation: Require ticker to be in valid list.

        Returns:
            List of extracted tickers.
        """
        matches = self.UPPERCASE_PATTERN.findall(text)
        results = []

        for match in matches:
            if match in self.FALSE_POSITIVES:
                continue

            if require_validation and self.valid_tickers:
                if match in self.valid_tickers:
                    results.append(match)
            else:
                results.append(match)

        return results

    def extract_tickers(
        self,
        text: str,
        include_uppercase: bool = False,
        require_validation: bool = True,
    ) -> list[str]:
        """Extract all tickers from text.

        Args:
            text: Text to extract from.
            include_uppercase: Also extract uppercase words as potential tickers.
            require_validation: Require validation for uppercase tickers.

        Returns:
            List of unique extracted tickers.
        """
        if not text:
            return []

        tickers = set()

        # Always extract $TICKER format
        tickers.update(self.extract_dollar_tickers(text))

        # Optionally extract uppercase
        if include_uppercase:
            tickers.update(self.extract_uppercase_tickers(text, require_validation))

        return list(tickers)

    def extract_from_post(
        self,
        title: str,
        selftext: str = "",
        include_uppercase: bool = False,
    ) -> list[str]:
        """Extract tickers from a Reddit post.

        Args:
            title: Post title.
            selftext: Post body text.
            include_uppercase: Include uppercase word extraction.

        Returns:
            List of unique tickers found.
        """
        combined_text = f"{title} {selftext}"
        return self.extract_tickers(combined_text, include_uppercase)

    def count_tickers(
        self,
        df: pd.DataFrame,
        text_columns: list[str] = ["title", "selftext"],
        include_uppercase: bool = False,
    ) -> pd.DataFrame:
        """Count ticker mentions in a DataFrame.

        Args:
            df: DataFrame with text columns.
            text_columns: Columns to extract tickers from.
            include_uppercase: Include uppercase extraction.

        Returns:
            DataFrame with ticker counts.
        """
        ticker_counts: dict[str, int] = {}

        for _, row in df.iterrows():
            combined_text = " ".join(str(row.get(col, "")) for col in text_columns)
            tickers = self.extract_tickers(combined_text, include_uppercase)

            for ticker in tickers:
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

        result = pd.DataFrame([
            {"ticker": ticker, "count": count}
            for ticker, count in ticker_counts.items()
        ])

        if not result.empty:
            result = result.sort_values("count", ascending=False)

        return result

    def add_ticker_column(
        self,
        df: pd.DataFrame,
        text_columns: list[str] = ["title", "selftext"],
        include_uppercase: bool = False,
        column_name: str = "tickers",
    ) -> pd.DataFrame:
        """Add a column with extracted tickers to DataFrame.

        Args:
            df: Input DataFrame.
            text_columns: Columns to extract from.
            include_uppercase: Include uppercase extraction.
            column_name: Name for the new column.

        Returns:
            DataFrame with new tickers column.
        """
        df = df.copy()

        def extract_row_tickers(row):
            combined_text = " ".join(str(row.get(col, "")) for col in text_columns)
            return self.extract_tickers(combined_text, include_uppercase)

        df[column_name] = df.apply(extract_row_tickers, axis=1)
        return df
