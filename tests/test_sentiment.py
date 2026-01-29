"""Tests for sentiment scoring."""

import pytest
import pandas as pd

from src.data.sentiment_scorer import SentimentScorer, calculate_comment_velocity


class TestSentimentScorer:
    """Tests for SentimentScorer class."""

    def test_init_vader(self):
        """Test VADER initialization."""
        scorer = SentimentScorer(model="vader")
        assert scorer.model_name == "vader"
        assert scorer.vader is not None

    def test_score_vader_positive(self):
        """Test VADER scores positive text correctly."""
        scorer = SentimentScorer(model="vader")
        result = scorer.score_vader("This is amazing! Great product, love it!")

        assert "compound" in result
        assert result["compound"] > 0.5
        assert result["pos"] > result["neg"]

    def test_score_vader_negative(self):
        """Test VADER scores negative text correctly."""
        scorer = SentimentScorer(model="vader")
        result = scorer.score_vader("This is terrible. Worst experience ever. Hate it.")

        assert result["compound"] < -0.5
        assert result["neg"] > result["pos"]

    def test_score_vader_neutral(self):
        """Test VADER scores neutral text correctly."""
        scorer = SentimentScorer(model="vader")
        result = scorer.score_vader("The stock traded at $150 today.")

        assert -0.3 < result["compound"] < 0.3
        assert result["neu"] > 0.5

    def test_score_text_empty(self):
        """Test scoring empty text returns neutral."""
        scorer = SentimentScorer(model="vader")
        result = scorer.score_text("")

        assert result["compound"] == 0.0
        assert result["neu"] == 1.0

    def test_score_text_none_like(self):
        """Test scoring whitespace-only text."""
        scorer = SentimentScorer(model="vader")
        result = scorer.score_text("   ")

        assert result["compound"] == 0.0

    def test_score_post_title_only(self):
        """Test scoring a post with title only."""
        scorer = SentimentScorer(model="vader")
        result = scorer.score_post(
            title="This stock is amazing and I love it!",
            selftext="",
        )

        assert "compound" in result
        assert "title_sentiment" in result
        assert result["compound"] > 0

    def test_score_post_with_selftext(self):
        """Test scoring a post with title and selftext."""
        scorer = SentimentScorer(model="vader")
        result = scorer.score_post(
            title="AAPL is going to the moon!",
            selftext="I'm so bullish on this stock. Great earnings ahead.",
        )

        assert result["compound"] > 0
        assert result["selftext_sentiment"] > 0

    def test_score_post_with_comments(self):
        """Test scoring a post with comments."""
        scorer = SentimentScorer(model="vader")
        result = scorer.score_post(
            title="What do you think about TSLA?",
            selftext="",
            comments=["Love it!", "Great company", "Best stock ever"],
        )

        assert result["num_comments_scored"] == 3
        assert result["comment_sentiment"] > 0

    def test_score_post_weighting(self):
        """Test that weighting affects final score."""
        scorer = SentimentScorer(model="vader")

        # Positive title, negative selftext
        result = scorer.score_post(
            title="Amazing stock!",
            selftext="Actually this is terrible and will crash.",
            title_weight=1.0,
            selftext_weight=0.1,  # Low weight for selftext
        )

        # Should be closer to title sentiment
        assert result["compound"] > 0

    def test_score_dataframe(self):
        """Test scoring a DataFrame."""
        scorer = SentimentScorer(model="vader")

        df = pd.DataFrame({
            "title": ["Great news!", "Terrible earnings", "Stock is flat"],
            "selftext": ["", "", ""],
        })

        result = scorer.score_dataframe(df)

        assert "sentiment" in result.columns
        assert len(result) == 3
        assert result.iloc[0]["sentiment"] > 0  # Positive
        assert result.iloc[1]["sentiment"] < 0  # Negative

    def test_calculate_daily_sentiment_mean(self):
        """Test daily sentiment aggregation with mean."""
        scorer = SentimentScorer(model="vader")

        df = pd.DataFrame({
            "created_utc": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
            "sentiment": [0.5, 0.3, -0.2],
            "score": [10, 5, 20],
        })

        result = scorer.calculate_daily_sentiment(df, agg_method="mean")

        assert len(result) == 2
        assert result[result["date"] == pd.Timestamp("2023-01-01").date()]["sentiment"].iloc[0] == 0.4

    def test_calculate_daily_sentiment_weighted(self):
        """Test daily sentiment aggregation with weighting."""
        scorer = SentimentScorer(model="vader")

        df = pd.DataFrame({
            "created_utc": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "sentiment": [0.5, -0.5],
            "score": [100, 1],  # First post has much higher score
            "num_comments": [50, 5],
        })

        result = scorer.calculate_daily_sentiment(df, agg_method="weighted")

        # Weighted towards the high-score positive post
        assert result.iloc[0]["sentiment"] > 0


class TestCommentVelocity:
    """Tests for comment velocity calculation."""

    def test_calculate_comment_velocity(self):
        """Test comment velocity calculation."""
        df = pd.DataFrame({
            "created_utc": pd.to_datetime([
                "2023-01-01", "2023-01-02", "2023-01-03",
                "2023-01-04", "2023-01-05", "2023-01-06",
                "2023-01-07", "2023-01-08",
            ]),
            "num_comments": [100, 120, 110, 115, 108, 112, 118, 500],  # Spike on last day
        })

        result = calculate_comment_velocity(df, window=7)

        assert "comment_velocity" in result.columns
        # Last day should have high velocity due to spike
        assert result.iloc[-1]["comment_velocity"] > 2.0

    def test_calculate_comment_velocity_empty(self):
        """Test velocity calculation with empty data."""
        df = pd.DataFrame(columns=["created_utc", "num_comments"])
        result = calculate_comment_velocity(df)

        assert len(result) == 0
