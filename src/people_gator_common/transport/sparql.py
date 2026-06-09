from __future__ import annotations
from collections.abc import Callable, Iterable
from itertools import islice
from json import JSONDecodeError
import logging
from typing import TypeAlias, cast
import pandas as pd
from SPARQLWrapper import JSON, SPARQLWrapper
from people_gator_common.filesystem import open_file
from people_gator_common.rate_limiter import RateLimiter, RateLimiterInstance, RateLimiterTooManyRequestsError
from people_gator_common.errors import (
    is_retryable_error
)

SparqlValue: TypeAlias = dict[str, str]
SparqlBinding: TypeAlias = dict[str, SparqlValue]
SparqlHead: TypeAlias = dict[str, list[str]]
SparqlResults: TypeAlias = dict[str, list[SparqlBinding]]
SparqlJsonResult: TypeAlias = dict[str, SparqlHead | SparqlResults]


def chunked(iterable: Iterable[str], chunk_size: int) -> Iterable[list[str]]:
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk

def parse_sparql_results(json_result: SparqlJsonResult) -> pd.DataFrame:
    headers = cast(SparqlHead, json_result["head"])["vars"]
    results = cast(SparqlResults, json_result["results"])["bindings"]

    parsed_rows: list[dict[str, str | None]] = []
    for result in results:
        row: dict[str, str | None] = {}
        for header in headers:
            row[header] = result.get(header, {}).get("value")
        parsed_rows.append(row)

    return pd.DataFrame(parsed_rows).drop_duplicates().reset_index(drop=True)


# Error detection helpers moved to wikidata.errors

def single_query_attempt(                         
    client: SPARQLWrapper,
    query: str,
    rate_limiter_instance: RateLimiterInstance,
):
    logger = logging.getLogger(__name__ + "." + single_query_attempt.__name__)
    client.setQuery(query)
    client.setReturnFormat(JSON)
    max_query_log_len = 128
    trunc_query = " ".join(line.strip() for line in query.splitlines() if line.strip())
    trunc_end = "..."
    logger.info(
        "Attempting query: %s",
        trunc_query[: (max_query_log_len - len(trunc_end))] + (trunc_end if len(trunc_query) > max_query_log_len else ""),
    )
    rate_limiter_instance.request()
    json_result = cast(SparqlJsonResult, client.query().convert())
    logger.info("Successful query")
    return json_result

def execute_sparql_query(
    *,
    client: SPARQLWrapper,
    query: str,
    rate_limiter: RateLimiter,
) -> SparqlJsonResult:
    logger = logging.getLogger(__name__ + "." + execute_sparql_query.__name__)
    with rate_limiter as rl:
        while True:
            try:
                return single_query_attempt(client, query, rl)
            except JSONDecodeError:
                logger.warning("JSON decode error")
                continue
            except RateLimiterTooManyRequestsError as error:
                logger.warning("Too many requests error: %s", error)
                raise
            except Exception as error:
                logger.warning("Error: %s", error)
                if is_retryable_error(error):
                    logger.info("Retrying after error")
                    continue
                raise


def rate_limited_query_all(
    query_builder: Callable[[int, int], str],
    partial_results_dir: str | None,
    *,
    client: SPARQLWrapper,
    limit: int,
    initial_offset: int,
    rate_limiter: RateLimiter,
) -> pd.DataFrame:
    dataframes: list[pd.DataFrame] = []
    offset = initial_offset
    logger = logging.getLogger(__name__ + "." + rate_limited_query_all.__name__)

    while True:
        query = query_builder(limit, offset)
        try:
            json_result = execute_sparql_query(
                client=client,
                query=query,
                rate_limiter=rate_limiter,
            )
        except Exception:
            logger.exception("Failed query at offset %s", offset)
            raise

        df = parse_sparql_results(json_result)
        if df.empty:
            break
        if partial_results_dir is not None:
            batch_file = f"{partial_results_dir}/results_offset_{offset}.csv"
            with open_file(batch_file, "w", encoding="utf-8") as f:
                df.to_csv(f, index=False)
            logger.info("Saved partial results for offset %d to %s", offset, batch_file)
        dataframes.append(df)
        offset += limit

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    return pd.DataFrame()


def rate_limited_query_all_cursor(
    query_builder: Callable[[str, int], str],
    partial_results_dir: str | None,
    *,
    client: SPARQLWrapper,
    limit: int,
    rate_limiter: RateLimiter,
) -> pd.DataFrame:
    dataframes: list[pd.DataFrame] = []
    after_id = "http://www.wikidata.org/entity/Q0"
    logger = logging.getLogger(__name__ + "." + rate_limited_query_all_cursor.__name__)

    while True:
        query = query_builder(after_id, limit)

        try:
            json_result = execute_sparql_query(
                client=client,
                query=query,
                rate_limiter=rate_limiter,
            )
        except Exception:
            logger.exception("Failed query after %s", after_id)
            raise

        df = parse_sparql_results(json_result)
        if df.empty:
            break
        if partial_results_dir is not None:
            partial_file = f"{partial_results_dir}/results_after_{after_id.split('/')[-1]}.csv"
            with open_file(partial_file, "w", encoding="utf-8") as f:
                df.to_csv(f, index=False)
            logger.info("Saved partial results to %s", partial_file)
        dataframes.append(df)
        after_id = df["entity"].iloc[-1]

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    return pd.DataFrame()


def rate_limited_query_entities(
    entities: list[str],
    query_builder: Callable[[list[str]], str],
    partial_results_dir: str | None,
    *,
    client: SPARQLWrapper,
    batch_size: int,
    rate_limiter: RateLimiter,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    logger = logging.getLogger(__name__ + "." + rate_limited_query_entities.__name__)

    for batch_index, batch in enumerate(chunked(entities, batch_size), start=1):
        query = query_builder(batch)
        json_result = execute_sparql_query(
            client=client,
            query=query,
            rate_limiter=rate_limiter,
        )
        df = parse_sparql_results(json_result)
        if df.empty:
            continue
        if partial_results_dir is not None:
            batch_file = f"{partial_results_dir}/results_batch_{batch_index}.csv"
            with open_file(batch_file, "w", encoding="utf-8") as f:
                df.to_csv(f, index=False)
            logger.info("Saved partial results for batch %d to %s", batch_index, batch_file)
        rows.append(df)
        logger.info("Processed property batch %d with %d entities", batch_index, len(batch))

    if rows:
        return pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return pd.DataFrame()