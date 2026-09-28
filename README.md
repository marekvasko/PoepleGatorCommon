## people-gator-common

Shared utilities, models, and clients used across the People Gator monorepo.

## Authority Import CLI

The `people-gator-import-authorities` CLI imports authority records (Wikidata/Wikimedia and NLCR) into Weaviate.

Example:

```bash
uv run people-gator-import-authorities \
	--wikidata-dir wikidata \
	--wikidata-prefix people_of_all \
	--nlcr-dir nlcr_aut \
	--weaviate-http-host localhost \
	--weaviate-http-port 8080
```

Input directories can be:

- Storage-relative paths resolved by `people_gator_common.filesystem` (local/S3/LakeFS based on `PEOPLEGATOR__STORAGE__...` settings).
- Explicit remote URIs like `s3://bucket/prefix`.

## Historical face import

Upload raw detections from a tree containing `*.people_gator.jsonl` files as
unassigned face records. Each face is linked to its typed page source; crop image
bytes and embeddings are not uploaded.

```bash
PEOPLE_GATOR_API_USERNAME=test PEOPLE_GATOR_API_PASSWORD=... \
  people-gator-import-faces /path/to/people_gator__data
```

Attach an existing `embeddings.npy` artifact (and its adjacent manifest and
metadata files) without decoding, detecting, aligning, or transforming images:

```bash
PEOPLE_GATOR_API_USERNAME=test PEOPLE_GATOR_API_PASSWORD=... \
  people-gator-backfill-face-embeddings /path/to/embeddings.npy
```

## LLM NER CLI

The `people-gator-llm-ner` command extracts named entities from PeopleGator
page-layout JSON using the storage backend configured by
`PEOPLEGATOR__STORAGE__...`. The input prefix must contain
`<library>/<document-id>/<page-id>.json`; output is written below the
output prefix as `<library>/<document-id>/<page-id>.json`.

The command lists both prefixes once, skips outputs that already exist, and
processes only missing pages. Set `OPENROUTER_API_KEY` (or place it in `.env`),
then run. The canonical prefixes are the defaults:

```bash
uv run people-gator-llm-ner
```

They can also be overridden explicitly:

```bash
uv run people-gator-llm-ner \
    --input-prefix page-layouts \
    --output-prefix page-named-entities
```

Use `--max-pages 1` for a one-page smoke test. See
`uv run people-gator-llm-ner --help` for model, chunking, reasoning, and
overwrite options.

## NER entity-linking decisions

The `people-gator-ner-entity-linking` command reads the per-page output of
`people-gator-ner-clusters`. For every resolved source cluster it searches the
People Gator API with repeated `query.text` parameters, fetches full details for
each returned entity, and asks the configured OpenRouter model whether to merge
with one candidate or create a person. Short and long descriptions are rendered
in Czech (`cs`), including when database descriptions are in another language.

By default the decision is applied: existing entities are patched, new entities
are posted, the page source is resolved through `/api/v1/pages/lookup`, and all
extractive text mentions are uploaded. The per-page JSON is a durable audit
journal containing the source evidence, candidates, LLM request/response,
decision, original and resulting entity states, and every created mention. It is
checkpointed after each API mutation. The `modifications.jsonl` file under the
output prefix is an append-only record of each applied operation with its
before and after states.

When page lookup returns 404, the pipeline registers the missing page through
`POST /api/v1/pages` and uses the returned `source_id` for its mentions. A
concurrent-registration 409 is resolved by repeating the lookup.

If a live run is interrupted, rerun the same command with the same output
prefix. Incomplete apply journals are resumed automatically; recorded entities
and mentions are not posted twice. Do not use `--force` for recovery.

The overall ETA is calculated from document-pages that will actually run;
completed outputs and pages beyond `--max-pages` are excluded. Use
`--parallel-entities 4` to keep up to four entity searches and LLM decisions
in flight across page files (the default). Results are committed in input order. API access
is serialized. In-flight decisions that include an edited candidate are
cancelled and restarted; every candidate list changed by an earlier mutation
is sent back to the model before it is applied. The
`--parallel-pages` and `--parallel-requests` options remain aliases.
Use `--progress-interval 5` for updates every five seconds (default: 30 seconds).
The CLI suppresses `httpx` and `httpcore` request logs, leaving pipeline
progress and decision logs visible.

```bash
PEOPLE_GATOR_API_BEARER_TOKEN=... OPENROUTER_API_KEY=... \
  uv run people-gator-ner-entity-linking
```

Instead of an existing bearer token, set `PEOPLE_GATOR_API_USERNAME` and
`PEOPLE_GATOR_API_PASSWORD` to use the OAuth2 password flow. Apply mode requests
`browse`, `annotate`, and `ingest` scopes.

Use `--decision-only` to retain the old non-mutating behavior. To reverse the
newest journaled page (mentions first, then entities in reverse order), use:

```bash
uv run people-gator-ner-entity-linking --rollback-latest
```

Rollback requires `admin` in addition to browse/annotate permissions. It checks
that an entity still matches the journaled post-change state before restoring or
deleting it; `--force-rollback` overrides that conflict check.
