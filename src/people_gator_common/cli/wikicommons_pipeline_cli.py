from __future__ import annotations

import argparse
import logging

from people_gator_common.cli.helpers import add_rate_limiter_args, configure_logging, rate_limiter_from_args
from people_gator_common.pipelines.wikicommons import run_wikicommons_images_download_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for Wikicommons image download pipeline.")
    parser.add_argument("--input-csv", type=str, default="output/people_all_P18.csv", help="CSV file containing image URLs.")
    parser.add_argument("--url-column", type=str, default="value", help="Column name in --input-csv containing image URLs.")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--hide-progress", action="store_true", help="Disable tqdm progress bar output.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    add_rate_limiter_args(parser, default_interval=5.0, default_retries=5, default_backoff=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    rate_limiter = rate_limiter_from_args(args)
    stats = run_wikicommons_images_download_pipeline(
        input_csv=str(args.input_csv),
        output_dir=str(args.output_dir),
        rate_limiter=rate_limiter,
        url_column=args.url_column,
        show_progress=not bool(args.hide_progress),
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Wikicommons pipeline finished. total=%d queued=%d skipped_existing=%d downloaded=%d failed=%d",
        stats.total, stats.queued, stats.skipped_existing, stats.downloaded, stats.failed,
        extra={"total": stats.total, "queued": stats.queued, "skipped_existing": stats.skipped_existing, "downloaded": stats.downloaded, "failed": stats.failed},
    )
    return 1 if stats.failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
