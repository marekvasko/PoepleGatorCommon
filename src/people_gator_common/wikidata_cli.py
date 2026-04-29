from __future__ import annotations

import argparse
from datetime import datetime
from json import JSONDecodeError
import logging
from pathlib import Path
import random
import sys
import time
from typing import TypeAlias, cast
from urllib.error import HTTPError, URLError

import pandas as pd
from SPARQLWrapper import JSON, SPARQLWrapper

DEFAULT_ENDPOINT_URL = "https://query.wikidata.org/sparql"
DEFAULT_LANGUAGES = "cs,sk,de,en,[AUTO_LANGUAGE],mul"

SparqlValue: TypeAlias = dict[str, str]
SparqlBinding: TypeAlias = dict[str, SparqlValue]
SparqlHead: TypeAlias = dict[str, list[str]]
SparqlResults: TypeAlias = dict[str, list[SparqlBinding]]
SparqlJsonResult: TypeAlias = dict[str, SparqlHead | SparqlResults]
RateLimitState: TypeAlias = dict[str, float]


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


def build_historical_european_countries_query(
    *,
    limit: int,
    offset: int,
    languages: str,
) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?country ?countryLabel WHERE {{
  ?country wdt:P31 wd:Q3024240 ;
           wdt:P30 wd:Q46 .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"{languages}\". }}
}}
LIMIT {limit}
OFFSET {offset}
""".strip()

def build_european_country_query(
    *,
    limit: int,
    offset: int,
    languages: str,
) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?country ?countryLabel WHERE {{
  ?country wdt:P31 wd:Q6256 ;
           wdt:P30 wd:Q46 .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"{languages}\". }}
}}
LIMIT {limit}
OFFSET {offset}
""".strip()

def build_people_of_country_query_count(*, country_id: str) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT (COUNT(*) AS ?count) WHERE {{
    ?person wdt:P31 wd:Q5 ;
            wdt:P27 wd:{country_id} .
}}
""".strip()

def build_people_of_country_query(*, country_id: str, limit: int, offset: int, languages: str) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?person ?personLabel ?birthDate ?deathDate ?image ?nlcr_aut_id ?google_knowledge_graph_id WHERE {{
  ?person wdt:P31 wd:Q5 ;
          wdt:P27 wd:{country_id} .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"{languages}\". }}
  OPTIONAL {{ ?person wdt:P569 ?birthDate . }}
  OPTIONAL {{ ?person wdt:P570 ?deathDate . }}
  OPTIONAL {{ ?person wdt:P18 ?image . }}
  OPTIONAL {{ ?person wdt:P106 ?occupation . }}
  OPTIONAL {{ ?person wdt:P691 ?nlcr_aut_id . }}
  OPTIONAL {{ ?person wdt:P2671 ?google_knowledge_graph_id . }}
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def build_languages_spoken_in_europe_query(*, client: SPARQLWrapper) -> str:
    return """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT DISTINCT ?language ?languageLabel WHERE {
  ?language wdt:P31 wd:Q34770 ;
            wdt:P17 ?country .
  ?country wdt:P30 wd:Q46 .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?languageLabel
""".strip()

def append_query_dump(
    *,
    query_log_file: Path,
    query_name: str,
    query: str,
    offset: int,
    country_id: str | None = None,
) -> None:
    lines = [f"# {datetime.now().isoformat(timespec='seconds')} {query_name}"]
    if country_id is not None:
        lines.append(f"# country_id={country_id}")
    lines.append(f"# offset={offset}")
    lines.append(query)
    lines.append("")

    query_log_file.parent.mkdir(parents=True, exist_ok=True)
    with query_log_file.open("a", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines))
        file_handle.write("\n")


def configure_failure_logger(failure_log_file: Path) -> logging.Logger:
    logger = logging.getLogger("people_gator.wikidata")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    failure_log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(failure_log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def append_entities_to_csv(df: pd.DataFrame, output_file: Path) -> int:
    if df.empty:
        return 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_file.exists() and output_file.stat().st_size > 0
    df.to_csv(output_file, mode="a", header=not file_exists, index=False)
    return len(df)


def is_too_many_requests_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "too many requests" in message


def is_connection_reset_error(error: Exception) -> bool:
    if isinstance(error, ConnectionResetError):
        return True

    if isinstance(error, URLError):
        reason = error.reason
        if isinstance(reason, ConnectionResetError):
            return True
        errno = getattr(reason, "errno", None)
        if errno == 104:
            return True

    message = str(error).lower()
    return "connection reset by peer" in message or "errno 104" in message


def is_retryable_http_error(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        return error.code in {500, 502, 503, 504}

    message = str(error).lower()
    return (
        "http error 500" in message
        or "http error 502" in message
        or "http error 503" in message
        or "http error 504" in message
    )


def execute_sparql_query(
    *,
    client: SPARQLWrapper,
    logger: logging.Logger,
    query: str,
    query_name: str,
    min_request_interval: float,
    max_retries: int,
    retry_backoff_seconds: float,
    rate_limit_state: RateLimitState,
) -> SparqlJsonResult:
    last_request_at = rate_limit_state.get("last_request_at", 0.0)
    elapsed = time.monotonic() - last_request_at
    wait_seconds = min_request_interval - elapsed
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    total_attempts = max_retries + 1
    attempt = 1
    while True:
        try:
            client.setQuery(query)
            client.setReturnFormat(JSON)
            json_result = cast(SparqlJsonResult, client.query().convert())
            rate_limit_state["last_request_at"] = time.monotonic()
            return json_result
        except JSONDecodeError as error:
            rate_limit_state["last_request_at"] = time.monotonic()
            if attempt < total_attempts:
                backoff = retry_backoff_seconds * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, retry_backoff_seconds)
                sleep_seconds = backoff + jitter
                logger.warning(
                    "Failed to decode JSON during %s; retrying in %.2fs (attempt %s/%s)",
                    query_name,
                    sleep_seconds,
                    attempt + 1,
                    total_attempts,
                )
                time.sleep(sleep_seconds)
                attempt += 1
                continue
            logger.exception("JSON decode failed for %s", query_name)
            raise
        except Exception as error:
            rate_limit_state["last_request_at"] = time.monotonic()
            is_retryable_error = (
                is_too_many_requests_error(error)
                or is_connection_reset_error(error)
                or is_retryable_http_error(error)
            )
            if is_retryable_error and attempt < total_attempts:
                backoff = retry_backoff_seconds * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, retry_backoff_seconds)
                sleep_seconds = backoff + jitter
                logger.warning(
                    "Transient error during %s (%s); retrying in %.2fs (attempt %s/%s)",
                    query_name,
                    error.__class__.__name__,
                    sleep_seconds,
                    attempt + 1,
                    total_attempts,
                )
                time.sleep(sleep_seconds)
                attempt += 1
                continue
            raise


def query_all_historical_european_countries(
    *,
    client: SPARQLWrapper,
    logger: logging.Logger,
    query_log_file: Path,
    limit: int,
    initial_offset: int,
    languages: str,
    countries_file: Path,
    min_request_interval: float,
    max_retries: int,
    retry_backoff_seconds: float,
    rate_limit_state: RateLimitState,
) -> list[tuple[str, str, str | None]]:
    countries: list[tuple[str, str, str | None]] = []
    offset = initial_offset

    while True:
        query = build_historical_european_countries_query(
            limit=limit,
            offset=offset,
            languages=languages,
        )
        append_query_dump(
            query_log_file=query_log_file,
            query_name="historical european countries",
            query=query,
            offset=offset,
        )

        try:
            json_result = execute_sparql_query(
                client=client,
                logger=logger,
                query=query,
                query_name="historical european countries",
                min_request_interval=min_request_interval,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                rate_limit_state=rate_limit_state,
            )
        except Exception:
            logger.exception(
                "Failed historical European countries query at offset %s",
                offset,
            )
            raise

        df = parse_sparql_results(json_result)
        append_entities_to_csv(df, countries_file)
        for country in df.itertuples(index=False):
            country_uri = getattr(country, "country", None)
            country_label = getattr(country, "countryLabel", None)
            if country_uri:
                countries.append(
                    (
                        extract_entity_id(country_uri),
                        country_uri,
                        cast(str | None, country_label),
                    )
                )

        if len(df) < limit:
            break
        offset += limit

    return countries


def query_all_current_european_countries(
    *,
    client: SPARQLWrapper,
    logger: logging.Logger,
    query_log_file: Path,
    limit: int,
    initial_offset: int,
    languages: str,
    countries_file: Path,
    min_request_interval: float,
    max_retries: int,
    retry_backoff_seconds: float,
    rate_limit_state: RateLimitState,
) -> list[tuple[str, str, str | None]]:
    countries: list[tuple[str, str, str | None]] = []
    offset = initial_offset

    while True:
        query = build_european_country_query(
            limit=limit,
            offset=offset,
            languages=languages,
        )
        append_query_dump(
            query_log_file=query_log_file,
            query_name="current european countries",
            query=query,
            offset=offset,
        )

        try:
            json_result = execute_sparql_query(
                client=client,
                logger=logger,
                query=query,
                query_name="current european countries",
                min_request_interval=min_request_interval,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                rate_limit_state=rate_limit_state,
            )
        except Exception:
            logger.exception(
                "Failed current European countries query at offset %s",
                offset,
            )
            raise

        df = parse_sparql_results(json_result)
        append_entities_to_csv(df, countries_file)
        for country in df.itertuples(index=False):
            country_uri = getattr(country, "country", None)
            country_label = getattr(country, "countryLabel", None)
            if country_uri:
                countries.append(
                    (
                        extract_entity_id(country_uri),
                        country_uri,
                        cast(str | None, country_label),
                    )
                )

        if len(df) < limit:
            break
        offset += limit

    return countries


def query_all_people_of_country(
    *,
    client: SPARQLWrapper,
    logger: logging.Logger,
    query_log_file: Path,
    country_id: str,
    limit: int,
    initial_offset: int,
    languages: str,
    people_file: Path,
    country_uri: str | None = None,
    country_label: str | None = None,
    min_request_interval: float,
    max_retries: int,
    retry_backoff_seconds: float,
    rate_limit_state: RateLimitState,
) -> int:
    total_rows = 0
    offset = initial_offset

    while True:
        query = build_people_of_country_query(
            country_id=country_id,
            limit=limit,
            offset=offset,
            languages=languages,
        )
        append_query_dump(
            query_log_file=query_log_file,
            query_name="people of country",
            query=query,
            offset=offset,
            country_id=country_id,
        )

        try:
            json_result = execute_sparql_query(
                client=client,
                logger=logger,
                query=query,
                query_name=f"people of country {country_id}",
                min_request_interval=min_request_interval,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                rate_limit_state=rate_limit_state,
            )
        except Exception:
            logger.exception(
                "Failed people query for country_id=%s at offset %s",
                country_id,
                offset,
            )
            raise

        df = parse_sparql_results(json_result)
        if country_uri is not None:
            df["country"] = country_uri
        if country_label is not None:
            df["countryLabel"] = country_label
        total_rows += append_entities_to_csv(df, people_file)

        if len(df) < limit:
            break
        offset += limit

    return total_rows


def extract_entity_id(entity_uri: str) -> str:
    return entity_uri.rsplit("/", maxsplit=1)[-1]


def query_all_people_for_historical_european_countries(
    *,
    client: SPARQLWrapper,
    logger: logging.Logger,
    query_log_file: Path,
    limit: int,
    initial_offset: int,
    languages: str,
    countries_file: Path,
    people_file: Path,
    min_request_interval: float,
    max_retries: int,
    retry_backoff_seconds: float,
    rate_limit_state: RateLimitState,
) -> tuple[int, int]:
    historical_countries = query_all_historical_european_countries(
        client=client,
        logger=logger,
        query_log_file=query_log_file,
        limit=limit,
        initial_offset=initial_offset,
        languages=languages,
        countries_file=countries_file,
        min_request_interval=min_request_interval,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        rate_limit_state=rate_limit_state,
    )

    current_countries = query_all_current_european_countries(
        client=client,
        logger=logger,
        query_log_file=query_log_file,
        limit=limit,
        initial_offset=initial_offset,
        languages=languages,
        countries_file=countries_file,
        min_request_interval=min_request_interval,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        rate_limit_state=rate_limit_state,
    )

    countries_by_id: dict[str, tuple[str, str | None]] = {}
    for country_id, country_uri, country_label in historical_countries + current_countries:
        countries_by_id[country_id] = (country_uri, country_label)

    total_people_rows = 0
    for country_id, (country_uri, country_label) in countries_by_id.items():
        print(f"Querying people for country {country_id} ({country_label})")
        total_people_rows += query_all_people_of_country(
            client=client,
            logger=logger,
            query_log_file=query_log_file,
            country_id=country_id,
            limit=limit,
            initial_offset=initial_offset,
            languages=languages,
            people_file=people_file,
            country_uri=country_uri,
            country_label=country_label,
            min_request_interval=min_request_interval,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            rate_limit_state=rate_limit_state,
        )

    return len(countries_by_id), total_people_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Wikidata SPARQL exports with query and failure logging.",
    )

    common_parent = argparse.ArgumentParser(add_help=False)
    runtime_group = common_parent.add_argument_group("Runtime")
    runtime_group.add_argument(
        "-l",
        "--limit",
        type=int,
        default=10_000,
        help="Page size for SPARQL LIMIT.",
    )
    runtime_group.add_argument(
        "-o",
        "--initial-offset",
        type=int,
        default=0,
        help="Start offset for pagination.",
    )
    runtime_group.add_argument(
        "-u",
        "--endpoint-url",
        default=DEFAULT_ENDPOINT_URL,
        help="SPARQL endpoint URL.",
    )
    runtime_group.add_argument(
        "-a",
        "--agent-email",
        default="ivasko@fit.vutbr.cz",
        help="Contact email used in the User-Agent.",
    )
    runtime_group.add_argument(
        "-r",
        "--min-request-interval",
        type=float,
        default=0.25,
        help="Minimum delay in seconds between SPARQL requests.",
    )
    runtime_group.add_argument(
        "-R",
        "--max-retries",
        type=int,
        default=6,
        help="Number of retries after a 429 response.",
    )
    runtime_group.add_argument(
        "-b",
        "--retry-backoff-seconds",
        type=float,
        default=1.5,
        help="Base backoff in seconds for retrying 429 responses.",
    )

    output_group = common_parent.add_argument_group("Output and logging")
    output_group.add_argument(
        "-q",
        "--query-log-file",
        type=Path,
        default=Path("data/wikidata_queries.sparql"),
        help="File where executed query texts are appended.",
    )
    output_group.add_argument(
        "-f",
        "--failure-log-file",
        type=Path,
        default=Path("data/wikidata_query_failures.log"),
        help="File where failed query executions are logged.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    countries_parser = subparsers.add_parser(
        "countries",
        parents=[common_parent],
        help="Fetch all historical European countries.",
    )
    countries_query_group = countries_parser.add_argument_group("Countries query")
    countries_query_group.add_argument(
        "-g",
        "--languages",
        default=DEFAULT_LANGUAGES,
        help="Preferred label languages.",
    )
    countries_query_group.add_argument(
        "-C",
        "--countries-file",
        type=Path,
        required=True,
        help="CSV file where country entities are appended.",
    )

    people_parser = subparsers.add_parser(
        "people",
        parents=[common_parent],
        help="Fetch all people for a single country.",
    )
    people_query_group = people_parser.add_argument_group("People query")
    people_query_group.add_argument(
        "-c",
        "--country-id",
        required=True,
        help="Wikidata country id, e.g. Q214.",
    )
    people_query_group.add_argument(
        "-g",
        "--languages",
        default=DEFAULT_LANGUAGES,
        help="Preferred label languages.",
    )
    people_query_group.add_argument(
        "-P",
        "--people-file",
        type=Path,
        required=True,
        help="CSV file where person entities are appended.",
    )

    both_parser = subparsers.add_parser(
        "both",
        parents=[common_parent],
        help="Fetch countries first, then fetch people for each country.",
    )
    both_query_group = both_parser.add_argument_group("Combined query")
    both_query_group.add_argument(
        "-g",
        "--languages",
        default=DEFAULT_LANGUAGES,
        help="Preferred label languages for country discovery.",
    )
    both_query_group.add_argument(
        "-C",
        "--countries-file",
        type=Path,
        required=True,
        help="CSV file where discovered country entities are appended.",
    )
    both_query_group.add_argument(
        "-P",
        "--people-file",
        type=Path,
        required=True,
        help="CSV file where person entities are appended.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    user_agent = (
        "PeopleGator-WikidataIngest/0.1 "
        f"Python/{sys.version_info[0]}.{sys.version_info[1]} "
        f"({args.agent_email})"
    )
    client = SPARQLWrapper(args.endpoint_url, agent=user_agent)
    logger = configure_failure_logger(args.failure_log_file)
    rate_limit_state: RateLimitState = {"last_request_at": 0.0}

    if args.command == "countries":
        historical_countries = query_all_historical_european_countries(
            client=client,
            logger=logger,
            query_log_file=args.query_log_file,
            limit=args.limit,
            initial_offset=args.initial_offset,
            languages=args.languages,
            countries_file=args.countries_file,
            min_request_interval=args.min_request_interval,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            rate_limit_state=rate_limit_state,
        )
        current_countries = query_all_current_european_countries(
            client=client,
            logger=logger,
            query_log_file=args.query_log_file,
            limit=args.limit,
            initial_offset=args.initial_offset,
            languages=args.languages,
            countries_file=args.countries_file,
            min_request_interval=args.min_request_interval,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            rate_limit_state=rate_limit_state,
        )
        print(
            f"Appended {len(historical_countries) + len(current_countries)} countries "
            f"to {args.countries_file}"
        )
    elif args.command == "people":
        people_rows = query_all_people_of_country(
            client=client,
            logger=logger,
            query_log_file=args.query_log_file,
            country_id=args.country_id,
            limit=args.limit,
            initial_offset=args.initial_offset,
            languages=args.languages,
            people_file=args.people_file,
            min_request_interval=args.min_request_interval,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            rate_limit_state=rate_limit_state,
        )
        print(f"Appended {people_rows} people rows to {args.people_file}")
    else:
        country_count, people_rows = query_all_people_for_historical_european_countries(
            client=client,
            logger=logger,
            query_log_file=args.query_log_file,
            limit=args.limit,
            initial_offset=args.initial_offset,
            languages=args.languages,
            countries_file=args.countries_file,
            people_file=args.people_file,
            min_request_interval=args.min_request_interval,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            rate_limit_state=rate_limit_state,
        )
        print(f"Appended {country_count} countries to {args.countries_file}")
        print(f"Appended {people_rows} people rows to {args.people_file}")

    print(f"Query log: {args.query_log_file}")
    print(f"Failure log: {args.failure_log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
