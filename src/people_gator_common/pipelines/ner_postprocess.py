"""NER post-processing pipeline: normalize → soft-link → LLM-link → output.

Orchestrates the full workflow for resolving named entities extracted from
historical OCR documents into unified real-world referents with common names,
merged descriptions, and source-entity mappings.
"""

from __future__ import annotations

import json
import logging
import os
import re
from itertools import combinations
from multiprocessing import Pool, cpu_count
from typing import Any, cast

import numpy as np
import pandas as pd
from openai import OpenAI
from rapidfuzz import fuzz
from rapidfuzz.process import extractOne
from recordlinkage.preprocessing import clean
from sklearn.cluster import DBSCAN

from people_gator_common.filesystem import exists, makedirs, open_file, glob as fs_glob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: Ingest
# ---------------------------------------------------------------------------


def ingest_ner_jsonl(
    input_dir: str,
    output_path: str,
    *,
    force: bool = False,
) -> str:
    """Read per-document NER JSONL files and produce a consolidated DataFrame.

    Each source file is expected to contain one JSON object per line with at
    least ``ner`` (list of entity dicts) and ``id`` fields.

    Parameters
    ----------
    input_dir
        Directory containing ``*.jsonl`` files (one per document).
    output_path
        Path to write the consolidated ``random_documents_ner.jsonl``.
    force
        Re-ingest even if *output_path* already exists.

    Returns
    -------
    The *output_path* written.
    """
    if exists(output_path) and not force:
        logger.info("Ingested NER file %s already exists, skipping", output_path)
        return output_path

    logger.info("Ingesting NER JSONL files from %s", input_dir)
    ners: list[dict[str, Any]] = []
    from glob import glob as fs_glob

    list_of_files = sorted(fs_glob(os.path.join(input_dir, "*.jsonl")))
    for file_path in list_of_files:
        document_id = os.path.basename(file_path).split(".")[0]
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("JSON decode error in %s: %s", file_path, exc)
                    continue
                local_ners: list[Any] = data.get("ner", [])
                if not isinstance(local_ners, list):
                    local_ners = []
                doc_line_id = str(data.get("id", ""))
                for ner in local_ners:
                    ners.append({
                        "document_id": document_id,
                        "id": doc_line_id,
                        "entity": str(ner.get("entity", "")),
                        "type": str(ner.get("type", "")),
                        "all_name_variants": [str(v) for v in ner.get("all_name_variants", [])],
                        "information": str(ner.get("information", "")),
                    })

    ner_df = pd.DataFrame(ners)
    makedirs(os.path.dirname(output_path), exist_ok=True)
    with open_file(output_path, "w") as f:
        ner_df.to_json(f, orient="records", lines=True)
    logger.info("Ingested %d NER records to %s", len(ner_df), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Step 2: Normalize
# ---------------------------------------------------------------------------


def _normalize(s: str | pd.Series) -> str | pd.Series:
    """Strip accents using unicode normalization.

    Accepts either a single string or a pandas Series of strings.
    """
    return clean(s, strip_accents="unicode")  # type: ignore[no-any-return]


def normalize_ner(
    input_path: str,
    output_path: str,
    *,
    force: bool = False,
) -> str:
    """Normalize entity names and variants (strip accents).

    Parameters
    ----------
    input_path
        Path to the consolidated NER JSONL.
    output_path
        Path to write the normalized JSONL.

    Returns
    -------
    The *output_path* written.
    """
    if exists(output_path) and not force:
        logger.info("Normalized NER file %s already exists, skipping", output_path)
        return output_path

    logger.info("Normalizing NER entities from %s", input_path)
    with open_file(input_path, "r") as f:
        ner_df = pd.read_json(f, orient="records", lines=True)

    normalized_ner: list[dict[str, Any]] = []
    normalized_entities = _normalize(ner_df["entity"])
    if isinstance(normalized_entities, pd.Series):
        norm_iter = normalized_entities.tolist()
    else:
        norm_iter = [normalized_entities] * len(ner_df)
    for i, row, normalized_entity in zip(ner_df.index, ner_df.itertuples(), norm_iter):
        raw_variants = cast(list[str], row.all_name_variants)
        normalized_variants = _normalize(pd.Series(raw_variants)).tolist()  # type: ignore[union-attr]
        normalized_ner.append({
            "document_id": row.document_id,
            "id": row.id,
            "entity": normalized_entity,
            "all_name_variants": normalized_variants,
            "type": row.type,
            "information": row.information,
        })

    normalized_ner_df = pd.DataFrame(normalized_ner)
    makedirs(os.path.dirname(output_path), exist_ok=True)
    with open_file(output_path, "w") as f:
        normalized_ner_df.to_json(f, orient="records", lines=True)
    logger.info("Normalized %d records to %s", len(normalized_ner_df), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Step 3: Soft-link (fuzzy clustering)
# ---------------------------------------------------------------------------


def _best_fuzz_score(
    name_a: str,
    variants_a: list[str],
    name_b: str,
    variants_b: list[str],
) -> float:
    """Return the best fuzzy match score between two entities (0–100)."""
    name_match: float = fuzz.partial_token_sort_ratio(name_a, name_b)
    scores: list[float] = []
    all_name_a = [name_a] + variants_a
    all_name_b = [name_b] + variants_b
    for nm_a in all_name_a:
        match = extractOne(nm_a, all_name_b, scorer=fuzz.partial_token_sort_ratio)
        scores.append(match[1])  # type: ignore[index]
    best = (name_match + max(scores)) / 2
    return best


def _score_group(
    args: tuple[str, str, list[int], list[str], list[list[str]]],
) -> tuple[str, str, list[int], np.ndarray]:
    """Compute the pairwise fuzzy-score matrix for a single group."""
    doc_id, ner_type, indices, entities, variants = args
    n = len(indices)
    score_matrix = np.zeros((n, n))
    for local_a, local_b in combinations(range(n), 2):
        score = _best_fuzz_score(
            entities[local_a], variants[local_a],
            entities[local_b], variants[local_b],
        )
        score_matrix[local_a, local_b] = score
        score_matrix[local_b, local_a] = score
    return (doc_id, ner_type, indices, score_matrix)


def soft_link_ner(
    input_path: str,
    output_dir: str,
    *,
    threshold: float = 92.0,
    n_jobs: int = max(1, cpu_count() - 1),
    force: bool = False,
) -> str:
    """Fuzzy-cluster NER entities within (document, type) groups using DBSCAN.

    Writes per-group JSONL files to *output_dir*.

    Parameters
    ----------
    input_path
        Path to the normalized NER JSONL.
    output_dir
        Directory for per-group cluster JSONL files.
    threshold
        DBSCAN eps is derived as ``100 - threshold``.  Higher → stricter
        matching (entities must be more similar to cluster together).
    n_jobs
        Number of parallel workers.
    force
        Re-run even if output directory already contains files.

    Returns
    -------
    The *output_dir* written to.
    """
    makedirs(output_dir, exist_ok=True)

    # Quick check: if we already have outputs and force is off, skip
    existing_outputs = [p for p in os.listdir(output_dir) if p.endswith(".jsonl")] if os.path.isdir(output_dir) else []
    if existing_outputs and not force:
        logger.info("Soft-link output dir %s has %d files, skipping", output_dir, len(existing_outputs))
        return output_dir

    logger.info("Soft-linking NER entities from %s (threshold=%.1f)", input_path, threshold)
    with open_file(input_path, "r") as f:
        normalized_ner_df = pd.read_json(f, orient="records", lines=True)

    # Also load original (non-normalized) for output
    # Derive original path from normalized path convention
    orig_path = input_path.replace("_normalized", "")
    if exists(orig_path):
        with open_file(orig_path, "r") as f:
            ner_df = pd.read_json(f, orient="records", lines=True)
    else:
        logger.warning("Original NER file %s not found; using normalized data as originals", orig_path)
        ner_df = normalized_ner_df

    groups = normalized_ner_df.groupby(["document_id", "type"])
    total_groups = len(groups)
    logger.info("Linking within %d (document, type) groups · %d workers", total_groups, n_jobs)

    # Prepare group inputs
    score_inputs: list[tuple[str, str, list[int], list[str], list[list[str]]]] = []
    for (doc_id, ner_type), group in groups:
        indices = group.index.tolist()
        entities = [str(normalized_ner_df.loc[i, "entity"]) for i in indices]
        variants = [[str(s) for s in cast(list[str], normalized_ner_df.at[i, "all_name_variants"])] for i in indices]
        score_inputs.append((str(doc_id), str(ner_type), indices, entities, variants))

    # Phase 1 – parallel scoring
    with Pool(processes=n_jobs) as pool:
        scored_groups = list(pool.imap_unordered(_score_group, score_inputs))

    # Cluster worker globals
    _ORIG_LOOKUP: dict[int, dict[str, Any]] = {
        i: {
            "entity": str(ner_df.loc[i, "entity"]),
            "variants": [str(s) for s in cast(list[str], ner_df.at[i, "all_name_variants"])],
            "information": ner_df.at[i, "information"],
            "document_id": str(ner_df.loc[i, "document_id"]),
            "id": str(ner_df.loc[i, "id"]),
        }
        for i in ner_df.index
    }

    # Phase 2 – clustering + writing
    eps = 100.0 - threshold
    for doc_id, ner_type, indices, score_matrix in scored_groups:
        distance_matrix = 100.0 - score_matrix
        # A zero score on the diagonal means "not compared", not a distance
        # of 100. DBSCAN needs each entity to be its own neighbor so that
        # unmatched entities receive distinct cluster labels instead of -1.
        np.fill_diagonal(distance_matrix, 0.0)
        clustering = DBSCAN(eps=eps, min_samples=1, metric="precomputed")
        clustering.fit(distance_matrix)

        remapped: dict[int, list[int]] = {}
        for local_idx, cluster_id in enumerate(clustering.labels_):
            global_idx = indices[local_idx]
            remapped.setdefault(int(cluster_id), []).append(global_idx)

        document_cluster_path = os.path.join(output_dir, f"{doc_id}__{ner_type}.jsonl")
        with open_file(document_cluster_path, "w") as f:
            for member_indices in remapped.values():
                cluster_record = [
                    {
                        "entity": str(_ORIG_LOOKUP[i]["entity"]),
                        "variants": [str(s) for s in _ORIG_LOOKUP[i]["variants"]],
                        "information": str(_ORIG_LOOKUP[i]["information"]),
                        "document_id": str(_ORIG_LOOKUP[i]["document_id"]),
                        "id": str(_ORIG_LOOKUP[i]["id"]),
                    }
                    for i in member_indices
                ]
                f.write(json.dumps(cluster_record) + "\n")

    logger.info("Soft-linking complete, wrote clusters to %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Step 4: LLM-link (disambiguation)
# ---------------------------------------------------------------------------

DISAMBIGUATION_PROMPT_TEMPLATE = """\
You are an expert in entity resolution for historical documents (OCR-derived text, possibly noisy).

# Task
You will be given a set of entities extracted from a single document. All entities are of the same type, but the type can be anything: persons, places, organizations, dates, etc.
Each entity has a numeric Id, a Name (with possible name variants), and a Description.
All of these entities were tentatively grouped together as referring to the same real-world referent. This proposal comes from fuzzy string matching and MAY BE WRONG — similar-looking names can belong to different referents.

Your task is to verify or split this group: decide which entities truly refer to the SAME real-world referent and output the corrected clusters. The result may be one cluster (the proposal was right), several clusters, or all singletons.

# Decision rules
Treat two entities as the same referent when:
- They are spelling, OCR, or transcription variants of the same name (e.g. swapped/missing diacritics, l/1, rn/m confusions).
- They are inflected forms of the same name (grammatical declension, e.g. Czech cases: "Karel Čapek" / "Karla Čapka").
- One is an abbreviation, initialism, or shortened form of the other ("J. Novák" / "Josef Novák") and the descriptions do not contradict each other.
- One is missing a middle name or patronymic that the other has, but otherwise they match ("Jan Novák" / "Jan Karel Novák").
- They differ only by titles or honorifics ("dr. Novák" / "Novák").
- The ordering of the name components is swapped ("Novák Jan" / "Jan Novák").
- The description proposal might confuse some facts, such as gender. Try to unify even when majority of other facts and context match.
- For dates: they denote the same point or span in time, possibly at different precision or in different formats ("14. 6. 1914" / "14. června 1914", "1914-06-14" / "14. júna 1914").

Split entities into separate clusters ONLY when there is clear evidence they are different referents:
- The descriptions attribute incompatible facts (different occupations, locations, life dates, roles in the text) though expect some innacuracies in the descriptions.
- The names clearly identify different people (e.g. different first names with a shared surname: "Jan Novák" vs "Petr Novák").
- Dates clearly denote different points in time ("1914" vs "1941", "12. 6. 1914" vs "14. 6. 1914").

When the evidence is ambiguous or merely missing (vague descriptions, compatible but non-overlapping facts), KEEP THE ENTITIES TOGETHER. Splitting the same referent is the more costly error: the clusters will later be matched against a database, and a wrongly split referent produces duplicate matches. Only split when the entities cannot plausibly be the same.

# Output requirements
- Every input EntityId must appear in exactly one UniqueEntity. Do not invent, drop, or duplicate Ids.
- Number UniqueEntityIds sequentially starting from 1.
- CommonDescription: a few sentences, written in the same language as the input descriptions, merging the non-contradictory information from all member entities. For a singleton UniqueEntity, restate its description concisely. Do not refer to the entity as a "person" unless it actually is one.
- ShortDescription: basic reduced information useful for entity identification, e.g. for people: occupation, gender, citizenship, interesting facts, etc.
- Output ONLY a single JSON object. No explanations, no text before or after it.

# Example output
```json
{{
  "UniqueEntities": [
    {{
      "EntityIds": [1, 2, 3],
      "ShortDescription": "Basic reduced information useful for entity identification, e.g. for people: occupation, gender, citizenship, interesting facts, etc.",
      "CommonDescription": "Merged summary of the cluster members' descriptions containing all information from source entities.",
      "CommonName": "Most complete name variant all parts, no titles."
    }},
    {{
      "EntityIds": [4],
      "ShortDescription": "Basic reduced information useful for entity identification, e.g. for people: occupation, gender, citizenship, interesting facts, etc.",
      "CommonDescription": "Concise restatement of entity 4's description.",
      "CommonName": "Most complete name."
    }}
  ]
}}
```

# Input Entities
```json
{entities}
```
"""


def _original_to_entity_json(originals: list[dict[str, Any]]) -> str:
    """Format a list of entity dicts into JSON for the LLM prompt."""
    entities: list[dict[str, Any]] = []
    for i, entity in enumerate(originals, start=1):
        entities.append({
            "EntityId": i,
            "Name": entity["entity"],
            "Variants": entity["variants"],
            "Description": entity["information"],
        })
    return json.dumps({"Entities": entities}, ensure_ascii=False, indent=2)


def _parse_llm_json_response(content: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response, with fallback."""
    content = content.strip()
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
    try:
        return json.loads(content)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s", exc)
        return {"UniqueEntities": []}


def _call_llm_disambiguate(
    client: OpenAI,
    model: str,
    entity_json: str,
) -> tuple[str, int, int]:
    """Call the LLM for disambiguation and return (response_text, prompt_tokens, completion_tokens)."""
    user_prompt = DISAMBIGUATION_PROMPT_TEMPLATE.format(entities=entity_json)
    chat_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
    )
    response_text = chat_response.choices[0].message.content or ""
    prompt_tokens = chat_response.usage.prompt_tokens if chat_response.usage is not None else 0
    completion_tokens = chat_response.usage.completion_tokens if chat_response.usage is not None else 0
    return response_text, prompt_tokens, completion_tokens


def llm_link_ner(
    input_dir: str,
    output_dir: str,
    *,
    api_key: str,
    model: str = "deepseek/deepseek-v4-pro",
    base_url: str = "https://openrouter.ai/api/v1",
    entity_type: str = "being",
    skip_singletons: bool = True,
    force: bool = False,
) -> str:
    """Disambiguate soft-linked clusters via LLM, producing final merged entities.

    For each (document, type) group, each cluster is sent to the LLM for
    verification/splitting.  Single-entity clusters can optionally be skipped
    (they are trivially resolved).

    Parameters
    ----------
    input_dir
        Directory with soft-link JSONL files (from :func:`soft_link_ner`).
    output_dir
        Directory for per-document LLM-linked JSON outputs.
    api_key
        OpenRouter API key.
    model
        Model identifier for the LLM.
    base_url
        OpenRouter-compatible base URL.
    entity_type
        Only process clusters of this NER type.
    skip_singletons
        Skip clusters with only one entity (they need no disambiguation).
    force
        Re-process even if output already exists.

    Returns
    -------
    The *output_dir* written to.
    """
    makedirs(output_dir, exist_ok=True)
    client = OpenAI(base_url=base_url, api_key=api_key)

    total_prompt_tokens = 0
    total_completion_tokens = 0

    cluster_files = sorted(fs_glob(os.path.join(input_dir, f"*__{entity_type}.jsonl")))
    logger.info("Found %d cluster files for type %s", len(cluster_files), entity_type)

    for cluster_file in cluster_files:
        doc_id = os.path.basename(cluster_file).split("__")[0]
        output_path = os.path.join(output_dir, f"{doc_id}__{entity_type}.json")

        if exists(output_path) and not force:
            logger.info("Output for document %s already exists, skipping", doc_id)
            continue

        # Load clusters from the soft-link file
        clusters: list[list[dict[str, Any]]] = []
        with open_file(cluster_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                clusters.append(json.loads(line))

        responses: list[str] = []
        parsed_originals: list[dict[str, Any]] = []

        for cluster in clusters:
            # Optimization: skip singletons
            if skip_singletons and len(cluster) == 1:
                entity_json = _original_to_entity_json(cluster)
                # Synthesize a trivial "cluster" response for the singleton
                singleton_data = {
                    "UniqueEntities": [{
                        "EntityIds": [1],
                        "ShortDescription": cluster[0]["information"],
                        "CommonDescription": cluster[0]["information"],
                        "CommonName": cluster[0]["entity"],
                    }]
                }
                responses.append(json.dumps(singleton_data, ensure_ascii=False))
                parsed_originals.append(json.loads(entity_json))
                continue

            entity_json = _original_to_entity_json(cluster)
            parsed_originals.append(json.loads(entity_json))

            response_text, pt, ct = _call_llm_disambiguate(client, model, entity_json)
            responses.append(response_text)
            total_prompt_tokens += pt
            total_completion_tokens += ct
            logger.debug(
                "Document %s: prompt=%d, completion=%d tokens (running total: %d/%d)",
                doc_id, pt, ct, total_prompt_tokens, total_completion_tokens,
            )

        # Parse and merge responses into final structure
        # Build a per-cluster lookup: EntityId (1-based) → original source data
        cluster_idx = 0
        cleaned_clusters: list[dict[str, Any]] = []
        for response_str, parsed_original in zip(responses, parsed_originals):
            parsed_response = _parse_llm_json_response(response_str)
            entity_map: dict[int, dict[str, Any]] = {}
            if parsed_original and "Entities" in parsed_original:
                entity_map = {e["EntityId"]: e for e in parsed_original["Entities"]}

            # Map EntityId → source record from the original cluster members
            source_cluster = clusters[cluster_idx]
            source_lookup: dict[int, dict[str, str]] = {}
            for local_idx, src in enumerate(source_cluster, start=1):
                source_lookup[local_idx] = {
                    "document_id": str(src.get("document_id", "")),
                    "id": str(src.get("id", "")),
                }
            cluster_idx += 1

            unique_entities = parsed_response.get("UniqueEntities", [])
            if not unique_entities:
                # Fallback: treat entire cluster as one entity
                unique_entities = [{
                    "EntityIds": list(entity_map.keys()) if entity_map else [],
                    "CommonName": "",
                    "CommonDescription": "",
                    "ShortDescription": "",
                }]

            for ue in unique_entities:
                entity_ids: list[int] = ue.get("EntityIds", [])
                member_entities: list[dict[str, Any]] = []
                for eid in entity_ids:
                    if eid not in entity_map:
                        continue
                    e = entity_map[eid]
                    src = source_lookup.get(eid, {})
                    member_entities.append({
                        "Name": e.get("Name", ""),
                        "Variants": e.get("Variants", []),
                        "Description": e.get("Description", ""),
                        "document_id": src.get("document_id", ""),
                        "id": src.get("id", ""),
                    })

                cleaned_cluster: dict[str, Any] = {
                    "CommonName": ue.get("CommonName", ""),
                    "CommonDescription": ue.get("CommonDescription", ""),
                    "ShortDescription": ue.get("ShortDescription", ""),
                    "Entities": member_entities,
                }
                cleaned_clusters.append(cleaned_cluster)

        output_data = {"Clusters": cleaned_clusters}
        output_json = json.dumps(output_data, ensure_ascii=False, indent=2)

        with open_file(output_path, "w") as f:
            f.write(output_json)

        logger.info(
            "Wrote %d merged clusters for document %s to %s",
            len(cleaned_clusters), doc_id, output_path,
        )

    logger.info(
        "LLM-linking complete. Total tokens: prompts=%d, completions=%d",
        total_prompt_tokens, total_completion_tokens,
    )
    return output_dir


# ---------------------------------------------------------------------------
# Glob helper (local filesystem, used internally)
# ---------------------------------------------------------------------------

def glob(path: str) -> list[str]:
    """Local-filesystem glob for file discovery during pipeline steps."""
    from glob import glob as _glob
    return sorted(_glob(path))


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_ner_postprocess_pipeline(
    input_dir: str,
    output_dir: str,
    *,
    api_key: str,
    model: str = "deepseek/deepseek-v4-pro",
    base_url: str = "https://openrouter.ai/api/v1",
    entity_type: str = "being",
    fuzzy_threshold: float = 92.0,
    skip_singletons: bool = True,
    n_jobs: int = max(1, cpu_count() - 1),
    force: bool = False,
) -> None:
    """Run the full NER post-processing pipeline.

    Phases:
        1. Ingest – read per-document NER JSONL into a consolidated DataFrame
        2. Normalize – strip accents from entity names and variants
        3. Soft-link – fuzzy-cluster entities within (document, type) groups
        4. LLM-link – disambiguate clusters via LLM, produce final merged entities

    Parameters
    ----------
    input_dir
        Directory containing per-document ``*.jsonl`` NER extraction files.
    output_dir
        Root directory for all pipeline outputs.
    api_key
        OpenRouter API key (typically from ``NER_POSTPROCESS_API_KEY`` env var).
    model
        Model identifier for the LLM disambiguation step.
    base_url
        OpenRouter-compatible base URL.
    entity_type
        NER entity type to process (e.g. ``"being"``, ``"location"``).
    fuzzy_threshold
        Threshold for fuzzy clustering (0–100).  Higher → stricter.
    skip_singletons
        Skip LLM calls for single-entity clusters.
    n_jobs
        Parallel workers for the fuzzy-scoring phase.
    force
        Re-run all steps even if intermediate outputs exist.
    """
    logger.info("Starting NER post-processing pipeline")
    logger.info("Input: %s | Output: %s | Entity type: %s", input_dir, output_dir, entity_type)

    makedirs(output_dir, exist_ok=True)

    # ── Phase 1: Ingest ────────────────────────────────────────────────
    ingested_path = os.path.join(output_dir, "ner_consolidated.jsonl")
    ingest_ner_jsonl(input_dir, ingested_path, force=force)

    # ── Phase 2: Normalize ─────────────────────────────────────────────
    normalized_path = os.path.join(output_dir, "ner_consolidated_normalized.jsonl")
    normalize_ner(ingested_path, normalized_path, force=force)

    # ── Phase 3: Soft-link ─────────────────────────────────────────────
    soft_linked_dir = os.path.join(output_dir, "softlinked_ner_clusters")
    soft_link_ner(
        normalized_path,
        soft_linked_dir,
        threshold=fuzzy_threshold,
        n_jobs=n_jobs,
        force=force,
    )

    # ── Phase 4: LLM-link ──────────────────────────────────────────────
    llm_linked_dir = os.path.join(output_dir, "llmlinked_ner_clusters")
    llm_link_ner(
        soft_linked_dir,
        llm_linked_dir,
        api_key=api_key,
        model=model,
        base_url=base_url,
        entity_type=entity_type,
        skip_singletons=skip_singletons,
        force=force,
    )

    logger.info("NER post-processing pipeline complete. Results in %s", llm_linked_dir)
