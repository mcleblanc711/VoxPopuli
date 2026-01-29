"""Sentiment scoring using VADER and FinBERT."""

import logging
from typing import Literal

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class SentimentScorer:
    """Score sentiment of text using VADER or FinBERT."""

    def __init__(
        self,
        model: Literal["vader", "finbert"] = "vader",
        device: str = "cpu",
    ):
        """Initialize the sentiment scorer.

        Args:
            model: Model to use ('vader' or 'finbert').
            device: Device for FinBERT ('cpu' or 'cuda').
        """
        self.model_name = model
        self.device = device
        self.vader = None
        self.finbert = None
        self.tokenizer = None

        if model == "vader":
            self._init_vader()
        elif model == "finbert":
            self._init_finbert()

    def _init_vader(self) -> None:
        """Initialize VADER sentiment analyzer."""
        self.vader = SentimentIntensityAnalyzer()
        logger.info("Initialized VADER sentiment analyzer")

    def _init_finbert(self) -> None:
        """Initialize FinBERT model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.finbert.to(self.device)
            self.finbert.eval()
            logger.info(f"Initialized FinBERT on {self.device}")

        except ImportError:
            logger.error("transformers/torch not installed. Falling back to VADER.")
            self.model_name = "vader"
            self._init_vader()

    def score_vader(self, text: str) -> dict:
        """Score text using VADER.

        Args:
            text: Text to score.

        Returns:
            Dict with neg, neu, pos, compound scores.
        """
        if not self.vader:
            self._init_vader()

        return self.vader.polarity_scores(text)

    def score_finbert(self, text: str) -> dict:
        """Score text using FinBERT.

        Args:
            text: Text to score.

        Returns:
            Dict with neg, neu, pos, compound scores.
        """
        if not self.finbert:
            self._init_finbert()

        if self.model_name == "vader":
            # Fallback if FinBERT failed to load
            return self.score_vader(text)

        import torch
        import torch.nn.functional as F

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.finbert(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        probs = probs.cpu().numpy()[0]

        # FinBERT outputs: [positive, negative, neutral]
        pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])

        # Calculate compound score (-1 to 1)
        compound = pos - neg

        return {
            "neg": neg,
            "neu": neu,
            "pos": pos,
            "compound": compound,
        }

    def score_text(self, text: str) -> dict:
        """Score text using the configured model.

        Args:
            text: Text to score.

        Returns:
            Dict with sentiment scores.
        """
        if not text or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        if self.model_name == "finbert":
            return self.score_finbert(text)
        return self.score_vader(text)

    def score_post(
        self,
        title: str,
        selftext: str = "",
        comments: list[str] | None = None,
        title_weight: float = 1.0,
        selftext_weight: float = 0.5,
        comments_weight: float = 0.3,
        top_n_comments: int = 10,
    ) -> dict:
        """Score a Reddit post with optional comments.

        Args:
            title: Post title.
            selftext: Post body.
            comments: List of comment bodies.
            title_weight: Weight for title sentiment.
            selftext_weight: Weight for selftext sentiment.
            comments_weight: Weight for comments sentiment.
            top_n_comments: Number of comments to include.

        Returns:
            Dict with weighted sentiment scores.
        """
        # Score individual components
        title_score = self.score_text(title)
        selftext_score = self.score_text(selftext) if selftext else None

        # Calculate comment sentiment if provided
        comment_scores = []
        if comments:
            for comment in comments[:top_n_comments]:
                score = self.score_text(comment)
                comment_scores.append(score)

        # Calculate weighted average
        total_weight = title_weight
        weighted_compound = title_score["compound"] * title_weight

        if selftext_score and selftext:
            total_weight += selftext_weight
            weighted_compound += selftext_score["compound"] * selftext_weight

        if comment_scores:
            avg_comment_compound = sum(s["compound"] for s in comment_scores) / len(
                comment_scores
            )
            total_weight += comments_weight
            weighted_compound += avg_comment_compound * comments_weight

        final_compound = weighted_compound / total_weight if total_weight > 0 else 0

        return {
            "compound": final_compound,
            "title_sentiment": title_score["compound"],
            "selftext_sentiment": selftext_score["compound"] if selftext_score else None,
            "comment_sentiment": (
                sum(s["compound"] for s in comment_scores) / len(comment_scores)
                if comment_scores
                else None
            ),
            "num_comments_scored": len(comment_scores),
        }

    def score_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        selftext_col: str = "selftext",
        output_col: str = "sentiment",
    ) -> pd.DataFrame:
        """Add sentiment scores to a DataFrame.

        Args:
            df: Input DataFrame.
            title_col: Column containing titles.
            selftext_col: Column containing selftext.
            output_col: Name for output column.

        Returns:
            DataFrame with sentiment column.
        """
        df = df.copy()

        def score_row(row):
            title = str(row.get(title_col, ""))
            selftext = str(row.get(selftext_col, ""))
            return self.score_post(title, selftext)["compound"]

        df[output_col] = df.apply(score_row, axis=1)
        return df

    def calculate_daily_sentiment(
        self,
        df: pd.DataFrame,
        date_col: str = "created_utc",
        sentiment_col: str = "sentiment",
        score_col: str = "score",
        agg_method: Literal["mean", "weighted"] = "weighted",
    ) -> pd.DataFrame:
        """Aggregate sentiment by day.

        Args:
            df: DataFrame with sentiment and date columns.
            date_col: Date column name.
            sentiment_col: Sentiment column name.
            score_col: Score column for weighting.
            agg_method: 'mean' or 'weighted' by score.

        Returns:
            DataFrame with daily aggregated sentiment.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df[date_col]).dt.date

        if agg_method == "weighted":

            def weighted_sentiment(group):
                weights = group[score_col].clip(lower=1)  # Min weight of 1
                return (group[sentiment_col] * weights).sum() / weights.sum()

            result = (
                df.groupby("date")
                .apply(weighted_sentiment)
                .reset_index(name="sentiment")
            )
        else:
            result = df.groupby("date")[sentiment_col].mean().reset_index()
            result.columns = ["date", "sentiment"]

        # Add post count and comment velocity
        counts = df.groupby("date").agg({
            sentiment_col: "count",
            "num_comments": "sum" if "num_comments" in df.columns else "count",
        }).reset_index()
        counts.columns = ["date", "post_count", "comment_count"]

        result = result.merge(counts, on="date")

        return result


def calculate_comment_velocity(
    df: pd.DataFrame,
    date_col: str = "created_utc",
    comments_col: str = "num_comments",
    window: int = 7,
) -> pd.DataFrame:
    """Calculate comment velocity (rate of change).

    Args:
        df: DataFrame with date and comment columns.
        date_col: Date column name.
        comments_col: Comments count column.
        window: Rolling window size in days.

    Returns:
        DataFrame with velocity column.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col]).dt.date

    daily = df.groupby("date")[comments_col].sum().reset_index()
    daily.columns = ["date", "comments"]
    daily = daily.sort_values("date")

    # Calculate rolling average
    daily["rolling_avg"] = daily["comments"].rolling(window, min_periods=1).mean()

    # Velocity = current / rolling average
    daily["comment_velocity"] = daily["comments"] / daily["rolling_avg"]

    return daily
