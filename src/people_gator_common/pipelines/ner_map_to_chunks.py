"""Pipeline for mapping NER entities to their character spans in source documents.

Given disambiguated entity clusters (output of the NER post-processing pipeline)
and the original document texts, this pipeline uses fuzzy matching to locate
each entity / name variant in the source text, computes confidence scores,
consolidates overlapping spans, and writes enriched cluster JSON files with
span annotations.
"""

from __future__ import annotations

import json
import logging
import os
from typing import cast

from fuzzysearch import find_near_matches

from people_gator_common.filesystem import exists, makedirs, open_file, glob as fs_glob

logger = logging.getLogger(__name__)


def _build_id_to_text(documents_dir: str) -> dict[str, str]:
    """Load document texts from ``*.jsonl`` files and return an id→text mapping."""
    id_to_text: dict[str, str] = {}
    document_files = sorted(fs_glob(os.path.join(documents_dir, "*.jsonl")))
    if not document_files:
        # Try glob without filesystem abstraction (local filesystem only)
        from glob import glob as std_glob
        document_files = sorted(std_glob(os.path.join(documents_dir, "*.jsonl")))
    for document_file in document_files:
        with open(document_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = document.get("id")
                doc_text = document.get("text")
                if doc_id and doc_text:
                    id_to_text[str(doc_id)] = str(doc_text)
    logger.info("Built id→text mapping for %d documents from %s", len(id_to_text), documents_dir)
    return id_to_text


def _consolidate_spans(
    spans: list[dict[str, int | float]],
    text: str | None = None,
) -> list[dict[str, int | float | str]]:
    """Merge overlapping or adjacent spans, keeping the highest score.

    When *text* is provided, the consolidated spans are enriched with a
    ``Text`` field containing the substring from the source document.
    """
    if not spans:
        return []

    spans.sort(key=lambda x: int(x["Start"]))

    consolidated: list[dict[str, int | float]] = [spans[0]]
    for current in spans[1:]:
        last = consolidated[-1]
        if int(current["Start"]) <= int(last["End"]):
            last["End"] = max(int(last["End"]), int(current["End"]))
            last["Score"] = max(float(last["Score"]), float(current["Score"]))
        else:
            consolidated.append(current)
    consolidated_retyped = cast(list[dict[str, int | float | str]], consolidated)
    if text is not None:
        for span in consolidated_retyped:
            span["Text"] = text[int(span["Start"]):int(span["End"])]
    return consolidated_retyped


def _find_spans_for_variants(
    variants: list[str],
    text: str,
    *,
    max_l_dist: int = 5,
    score_threshold: float = 80.0,
) -> list[dict[str, int | float]]:
    """Fuzzy-search each variant in *text* and return scored spans."""
    spans: list[dict[str, int | float]] = []
    for variant in variants:
        matches = find_near_matches(variant, text, max_l_dist=max_l_dist)
        for m in matches:
            match_len = max(len(variant), m.end - m.start)
            score = 100.0 * (1.0 - m.dist / match_len)
            if score > score_threshold:
                spans.append({
                    "Start": m.start,
                    "End": m.end,
                    "Score": score,
                })
    return spans


def run_ner_map_to_chunks_pipeline(
    *,
    documents_dir: str,
    linked_clusters_dir: str,
    output_dir: str,
    max_l_dist: int = 5,
    score_threshold: float = 80.0,
) -> None:
    """Enrich linked NER clusters with character-span annotations.

    Parameters
    ----------
    documents_dir
        Directory containing ``*.jsonl`` files with document ``id`` and ``text``.
    linked_clusters_dir
        Directory containing ``*.json`` files with disambiguated entity clusters.
    output_dir
        Directory where enriched cluster JSON files are written.
    max_l_dist
        Maximum Levenshtein distance for fuzzy substring matching.
    score_threshold
        Minimum span confidence score (0-100) to keep a match.
    """
    makedirs(output_dir, exist_ok=True)

    logger.info("Building document id→text mapping from %s", documents_dir)
    id_to_text = _build_id_to_text(documents_dir)

    linked_clusters_jsons = sorted(fs_glob(os.path.join(linked_clusters_dir, "*.json")))
    if not linked_clusters_jsons:
        from glob import glob as std_glob
        linked_clusters_jsons = sorted(std_glob(os.path.join(linked_clusters_dir, "*.json")))

    logger.info("Processing %d linked-cluster files from %s", len(linked_clusters_jsons), linked_clusters_dir)

    for cluster_json in linked_clusters_jsons:
        with open_file(cluster_json) as f:
            disambiguated_entities = json.load(f)

        for cluster in disambiguated_entities.get("Clusters", []):
            for entity in cluster.get("Entities", []):
                doc_id = entity.get("id")
                text = id_to_text.get(doc_id)
                if text is None:
                    logger.warning("Document %s not found in id_to_text mapping", doc_id)
                    continue

                # Gather all name variants (Variants list + primary Name)
                variants: list[str] = list(entity.get("Variants", []))
                primary_name = entity.get("Name")
                if primary_name:
                    variants.append(str(primary_name))

                spans = _find_spans_for_variants(
                    variants, text,
                    max_l_dist=max_l_dist,
                    score_threshold=score_threshold,
                )
                entity["Spans"] = _consolidate_spans(spans, text)

        # Determine output path by mirroring the directory structure
        rel_path = os.path.relpath(cluster_json, linked_clusters_dir)
        output_path = os.path.join(output_dir, rel_path)
        makedirs(os.path.dirname(output_path), exist_ok=True)
        with open_file(output_path, "w") as f:
            json.dump(disambiguated_entities, f, indent=2, ensure_ascii=False)

        logger.debug("Processed %s → %s", cluster_json, output_path)

    logger.info("Done – wrote enriched clusters to %s", output_dir)
