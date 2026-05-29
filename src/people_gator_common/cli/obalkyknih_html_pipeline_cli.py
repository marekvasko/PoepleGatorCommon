from __future__ import annotations

import argparse
import logging

from people_gator_common.cli.helpers import add_rate_limiter_args, configure_logging, rate_limiter_from_args
from people_gator_common.pipelines.obalkyknih import run_obalkyknih_html_download_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for ObalkyKnih HTML download pipeline.")
    parser.add_argument("-i", "--input-csv", type=str, default="nlcr_aut/unique_control_numbers.csv", help="CSV file containing NLCR AUT IDs.")
    parser.add_argument("-o", "--output-dir", type=str, default="html", help="Directory to save downloaded HTML files.")
    parser.add_argument("--auth-id-column", type=str, default="entity", help="Column name containing NLCR AUT IDs.")
    parser.add_argument("--hide-progress", action="store_true", help="Disable tqdm progress bar output.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    add_rate_limiter_args(parser, default_interval=0.5, default_retries=3, default_backoff=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    stats = run_obalkyknih_html_download_pipeline(
        input_csv=str(args.input_csv),
        output_dir=str(args.output_dir),
        rate_limiter=rate_limiter_from_args(args),
        auth_id_column=args.auth_id_column,
        show_progress=not bool(args.hide_progress),
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Obalkyknih pipeline finished. total=%d queued=%d skipped_existing=%d downloaded=%d failed=%d",
        stats.total, stats.queued, stats.skipped_existing, stats.downloaded, stats.failed,
        extra={"total": stats.total, "queued": stats.queued, "skipped_existing": stats.skipped_existing, "downloaded": stats.downloaded, "failed": stats.failed},
    )
    return 1 if stats.failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
