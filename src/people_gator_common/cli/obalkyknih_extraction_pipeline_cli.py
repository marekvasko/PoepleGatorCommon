from __future__ import annotations

import argparse

from people_gator_common.cli.helpers import configure_logging
from people_gator_common.pipelines.obalkyknih import run_obalkyknih_html_parse_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for ObalkyKnih HTML extraction pipeline.")
    parser.add_argument("-i", "--input-dir", type=str, default="html/obalkyknih.cz", help="Directory containing downloaded HTML files.")
    parser.add_argument("-o", "--output-csv", type=str, default="obalkyknih/previewimage.csv", help="CSV file to save extracted preview image URLs.")
    parser.add_argument("-s", "--output-skipped-csv", type=str, default="obalkyknih/non_existing_previewimage.csv", help="CSV file to save skipped entries.")
    parser.add_argument("-f", "--force", action="store_true", help="Force reprocessing of all entries.")
    parser.add_argument("-w", "--num-workers", type=int, default=16, help="Number of worker processes.")
    parser.add_argument("--hide-progress", action="store_true", help="Disable tqdm progress bar output.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    run_obalkyknih_html_parse_pipeline(
        html_dir=args.input_dir,
        output_csv=args.output_csv,
        output_skipped_csv=args.output_skipped_csv,
        show_progress=not args.hide_progress,
        diff_only=not args.force,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
