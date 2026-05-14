from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from people_gator_common.pipelines.nlcr_aut import run_nlcr_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for NLCR AUT MARC21 pipeline.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--personal-authorities-link",
        type=str,
        default="https://aleph.nkp.cz/data/aut_ja.xml.gz",
        help="URL to NLCR personal authorities MARC XML (gz).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_nlcr_pipeline(output_dir=Path(args.output_dir), personal_authorities_link=args.personal_authorities_link)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
