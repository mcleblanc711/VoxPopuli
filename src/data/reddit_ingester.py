"""Reddit data ingestion from Arctic Shift Pushshift dumps."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
import zstandard as zstd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RedditIngester:
    """Download and parse Reddit data from Arctic Shift dumps."""

    ARCTIC_SHIFT_BASE = "https://files.pushshift.io/reddit"
    SUBMISSIONS_PATH = "submissions"
    COMMENTS_PATH = "comments"

    def __init__(self, data_dir: str = "data/reddit"):
        """Initialize the ingester.

        Args:
            data_dir: Directory to store downloaded and processed data.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)

    def get_dump_url(self, year: int, month: int, data_type: str = "submissions") -> str:
        """Get the URL for a specific month's dump.

        Args:
            year: Year (e.g., 2023)
            month: Month (1-12)
            data_type: 'submissions' or 'comments'

        Returns:
            URL to the zst-compressed dump file.
        """
        path = self.SUBMISSIONS_PATH if data_type == "submissions" else self.COMMENTS_PATH
        filename = f"RS_{year}-{month:02d}.zst" if data_type == "submissions" else f"RC_{year}-{month:02d}.zst"
        return f"{self.ARCTIC_SHIFT_BASE}/{path}/{filename}"

    def download_dump(
        self,
        year: int,
        month: int,
        data_type: str = "submissions",
        chunk_size: int = 8192,
    ) -> Path:
        """Download a monthly dump file.

        Args:
            year: Year
            month: Month (1-12)
            data_type: 'submissions' or 'comments'
            chunk_size: Download chunk size in bytes

        Returns:
            Path to the downloaded file.
        """
        url = self.get_dump_url(year, month, data_type)
        filename = f"{data_type}_{year}_{month:02d}.zst"
        output_path = self.raw_dir / filename

        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return output_path

        logger.info(f"Downloading {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return output_path

    def parse_zst(
        self,
        file_path: Path,
        subreddits: list[str] | None = None,
        min_score: int = 1,
    ) -> Iterator[dict]:
        """Parse a zst-compressed NDJSON file.

        Args:
            file_path: Path to the .zst file
            subreddits: List of subreddits to filter (None = all)
            min_score: Minimum score threshold

        Yields:
            Parsed JSON objects matching the criteria.
        """
        subreddits_lower = {s.lower() for s in subreddits} if subreddits else None

        dctx = zstd.ZstdDecompressor()

        with open(file_path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                text_stream = reader.read().decode("utf-8", errors="ignore")

                for line in text_stream.split("\n"):
                    if not line.strip():
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Filter by subreddit
                    subreddit = obj.get("subreddit", "").lower()
                    if subreddits_lower and subreddit not in subreddits_lower:
                        continue

                    # Filter by score
                    if obj.get("score", 0) < min_score:
                        continue

                    yield obj

    def parse_zst_streaming(
        self,
        file_path: Path,
        subreddits: list[str] | None = None,
        min_score: int = 1,
        buffer_size: int = 65536,
    ) -> Iterator[dict]:
        """Parse a zst file using streaming decompression (memory efficient).

        Args:
            file_path: Path to the .zst file
            subreddits: List of subreddits to filter
            min_score: Minimum score threshold
            buffer_size: Read buffer size

        Yields:
            Parsed JSON objects.
        """
        subreddits_lower = {s.lower() for s in subreddits} if subreddits else None
        dctx = zstd.ZstdDecompressor()

        with open(file_path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                buffer = ""

                while True:
                    chunk = reader.read(buffer_size)
                    if not chunk:
                        break

                    buffer += chunk.decode("utf-8", errors="ignore")
                    lines = buffer.split("\n")
                    buffer = lines[-1]  # Keep incomplete line

                    for line in lines[:-1]:
                        if not line.strip():
                            continue

                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        subreddit = obj.get("subreddit", "").lower()
                        if subreddits_lower and subreddit not in subreddits_lower:
                            continue

                        if obj.get("score", 0) < min_score:
                            continue

                        yield obj

    def extract_submission_fields(self, obj: dict) -> dict:
        """Extract relevant fields from a submission object.

        Args:
            obj: Raw submission JSON object

        Returns:
            Dict with extracted fields.
        """
        created_utc = obj.get("created_utc", 0)
        if isinstance(created_utc, str):
            created_utc = int(float(created_utc))

        return {
            "post_id": obj.get("id", ""),
            "title": obj.get("title", ""),
            "selftext": obj.get("selftext", ""),
            "score": obj.get("score", 0),
            "num_comments": obj.get("num_comments", 0),
            "created_utc": datetime.utcfromtimestamp(created_utc),
            "subreddit": obj.get("subreddit", ""),
            "author": obj.get("author", "[deleted]"),
            "upvote_ratio": obj.get("upvote_ratio", 0.0),
            "permalink": obj.get("permalink", ""),
        }

    def extract_comment_fields(self, obj: dict) -> dict:
        """Extract relevant fields from a comment object.

        Args:
            obj: Raw comment JSON object

        Returns:
            Dict with extracted fields.
        """
        created_utc = obj.get("created_utc", 0)
        if isinstance(created_utc, str):
            created_utc = int(float(created_utc))

        return {
            "comment_id": obj.get("id", ""),
            "post_id": obj.get("link_id", "").replace("t3_", ""),
            "body": obj.get("body", ""),
            "score": obj.get("score", 0),
            "created_utc": datetime.utcfromtimestamp(created_utc),
            "subreddit": obj.get("subreddit", ""),
            "author": obj.get("author", "[deleted]"),
            "parent_id": obj.get("parent_id", ""),
        }

    def to_parquet(
        self,
        file_path: Path,
        subreddits: list[str],
        output_name: str | None = None,
        data_type: str = "submissions",
        min_score: int = 1,
    ) -> Path:
        """Convert a zst dump to parquet format.

        Args:
            file_path: Path to the .zst file
            subreddits: List of subreddits to extract
            output_name: Output filename (without extension)
            data_type: 'submissions' or 'comments'
            min_score: Minimum score threshold

        Returns:
            Path to the created parquet file.
        """
        if output_name is None:
            output_name = file_path.stem

        output_path = self.data_dir / f"{output_name}.parquet"

        logger.info(f"Converting {file_path} to parquet")

        records = []
        extractor = (
            self.extract_submission_fields
            if data_type == "submissions"
            else self.extract_comment_fields
        )

        for obj in tqdm(
            self.parse_zst_streaming(file_path, subreddits, min_score),
            desc="Processing",
        ):
            records.append(extractor(obj))

        if not records:
            logger.warning(f"No records found for subreddits: {subreddits}")
            return output_path

        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)

        logger.info(f"Saved {len(df)} records to {output_path}")
        return output_path

    def ingest_range(
        self,
        subreddits: list[str],
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        data_type: str = "submissions",
        min_score: int = 1,
    ) -> list[Path]:
        """Download and convert a range of monthly dumps.

        Args:
            subreddits: List of subreddits to extract
            start_year: Start year
            start_month: Start month
            end_year: End year
            end_month: End month
            data_type: 'submissions' or 'comments'
            min_score: Minimum score threshold

        Returns:
            List of created parquet file paths.
        """
        output_files = []
        current_year = start_year
        current_month = start_month

        while (current_year, current_month) <= (end_year, end_month):
            try:
                # Download
                zst_path = self.download_dump(current_year, current_month, data_type)

                # Convert to parquet
                output_name = f"{data_type}_{current_year}_{current_month:02d}"
                parquet_path = self.to_parquet(
                    zst_path, subreddits, output_name, data_type, min_score
                )
                output_files.append(parquet_path)

            except requests.HTTPError as e:
                logger.warning(f"Failed to download {current_year}-{current_month:02d}: {e}")

            # Next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        return output_files

    def load_parquet(self, pattern: str = "*.parquet") -> pd.DataFrame:
        """Load all parquet files matching pattern.

        Args:
            pattern: Glob pattern for files to load

        Returns:
            Combined DataFrame from all matching files.
        """
        files = list(self.data_dir.glob(pattern))
        if not files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
