from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

from people_gator_common.filesystem import open_file, glob
from people_gator_common.utils import unique_values_from_csv as unique_values_from_csv  # re-export
from os.path import join


def create_people_of_all(*, output_dir: str, input_csvs_glob: str, output_file: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__ + "." + create_people_of_all.__name__)
    input_files = glob(join(output_dir, input_csvs_glob))
    logger.debug("Found %d input files for glob pattern %s in directory %s", len(input_files), input_csvs_glob, output_dir)
    if not input_files:
        logger.warning("No input files found for glob pattern %s in directory %s", input_csvs_glob, output_dir)
        return pd.DataFrame()
    dataframes: list[pd.DataFrame] = []
    for input_file in input_files:
        with open_file(input_file, "r") as f:
            df = pd.read_csv(f)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True).drop_duplicates().sort_values(by="entity").reset_index(drop=True)
    logger.debug("Combined DataFrame has %d rows", len(combined_df))
    with open_file(output_file, "w") as f:
        combined_df.to_csv(f, index=False)
    logger.info("Saved combined DataFrame to %s", output_file)
    return combined_df



