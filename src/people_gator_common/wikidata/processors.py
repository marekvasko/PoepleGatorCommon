from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd


def create_people_of_all(*, output_dir: Path, input_csvs_glob: str, output_file: Path) -> pd.DataFrame:
    logger = logging.getLogger(__name__ + "." + create_people_of_all.__name__)
    input_files = list(output_dir.glob(input_csvs_glob))
    logger.debug("Found %d input files for glob pattern %s in directory %s", len(input_files), input_csvs_glob, output_dir)
    if not input_files:
        logger.warning("No input files found for glob pattern %s in directory %s", input_csvs_glob, output_dir)
        return pd.DataFrame()
    dataframes: list[pd.DataFrame] = []
    for input_file in input_files:
        df = pd.read_csv(input_file)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True).drop_duplicates().sort_values(by="entity").reset_index(drop=True)
    logger.debug("Combined DataFrame has %d rows", len(combined_df))
    combined_df.to_csv(output_file, index=False)
    logger.info("Saved combined DataFrame to %s", output_file)
    return combined_df


def unique_values_from_csv(
    source_csv: Path,
    target_csv: Path,
    values_column: str = "value",
    new_column_name: str = "entity",
) -> pd.DataFrame:
    logger = logging.getLogger(__name__ + "." + unique_values_from_csv.__name__)
    df = pd.read_csv(source_csv)
    logger.debug("Read %d rows from source CSV %s", len(df), source_csv)
    unique_values_df = df[[values_column]].drop_duplicates().sort_values(by=values_column).reset_index(drop=True)
    unique_values_df.columns = [new_column_name]
    logger.debug("Extracted %d unique values from column %s in source CSV %s", len(unique_values_df), values_column, source_csv)

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    unique_values_df.to_csv(target_csv, index=False)
    logger.info("Saved %d unique values to %s", len(unique_values_df), target_csv)

    return unique_values_df
