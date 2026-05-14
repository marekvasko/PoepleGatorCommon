from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

from people_gator_common.filesytem import ensure_parent_dir, open_file, glob
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


def unique_values_from_csv(
    source_csv: str,
    target_csv: str,
    values_column: str = "value",
    new_column_name: str = "entity",
) -> pd.DataFrame:
    logger = logging.getLogger(__name__ + "." + unique_values_from_csv.__name__)
    with open_file(source_csv, "r") as f:
        df = pd.read_csv(f)
    logger.debug("Read %d rows from source CSV %s", len(df), source_csv)
    unique_values_df = df[[values_column]].drop_duplicates().sort_values(by=values_column).reset_index(drop=True)
    unique_values_df.columns = [new_column_name]
    logger.debug("Extracted %d unique values from column %s in source CSV %s", len(unique_values_df), values_column, source_csv)

    ensure_parent_dir(target_csv)
    with open_file(target_csv, "w", encoding="utf-8") as f:
        unique_values_df.to_csv(f, index=False)
    logger.info("Saved %d unique values to %s", len(unique_values_df), target_csv)

    return unique_values_df
