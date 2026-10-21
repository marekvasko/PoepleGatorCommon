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

## NER clustering scope

`people-gator-ner-clusters` defaults to `--scope page`: it groups and
disambiguates entities within each page independently. To resolve people using
the extracted evidence from all available pages of each document, run:

```bash
uv run people-gator-ner-clusters --scope document
```

Document scope groups inputs by both library and document ID. It runs the same
fuzzy name clustering and LLM disambiguation across the combined NER chunks.
The model compares the extracted names, aliases and entity information across
pages; it does not read the full document text. Different people with similar
names can still be split by the disambiguation model. Pairwise fuzzy scoring
grows quadratically with the number of extracted entity records, and large
name groups produce larger LLM requests.

Every scope unit produces **exactly one self-contained JSON dump**:

| Scope | Default output key |
| --- | --- |
| `page` | `page-named-entities-clusters/<library>/<document-id>/<page-id>.json` |
| `document` | `document-named-entities-clusters/<library>/<document-id>.json` |

Override the prefix with `--output-prefix`. Document scope writes all resolved
clusters and their members to the document file, including entities from every
input page. It does not generate per-page projections. Multiple documents
produce one dump each; this is not a single aggregate file for the entire run.

Both scopes use `NerClustersDump` with `schema_version: 1` and a flat `clusters`
array. Each cluster contains:

- `cluster_id`: a deterministic UUID derived from the scope and sorted member
  record IDs, independent of the cluster name, generated descriptions and LLM
  output order. It is stable when membership is unchanged. A split, merge or
  added/removed member changes it; it is not a permanent database person ID.
- `name`, `short_description`, `long_description`: the resolved identity and
  combined descriptive evidence.
- `members`: every original extracted entity record, including its names,
  information and mentions under `entity`. Each member has a deterministic
  `record_id`, `page_id`, `chunk_id` and zero-based `chunk_index` / `entity_index`
  locating it in the source page NER JSON. Identical OCR chunks on different
  pages retain distinct record IDs.
- `disambiguation_index` and `entity_indices`: references into the saved LLM
  audit data, not persistent identity IDs.

The dump also contains `library`, `document_id`, `scope`, an optional `page_id`
for page scope, and `inputs` listing every source page and storage path.
`disambiguations` retains prompts, reasoning and raw responses once per candidate
group. Each source entity record must occur in exactly one resolved cluster.
Even a scope unit without any extracted entities receives a dump with an empty
`clusters` array and its input page references.

Use `--scope document --max-documents 1` for a smoke test. The limit counts
documents needing processing and always includes all their available pages;
`--max-pages` is restricted to page scope. Run NER extraction for all intended
pages first: document scope can only use page outputs that already exist.

Dumps record an `input_fingerprint` for the inputs/settings and a `run_id` for
that generation. Unchanged, valid dumps are skipped, preserving review edits.
Changed NER inputs or clustering settings, missing dumps or interrupted writes
cause the scope unit to be regenerated. `--force` also regenerates complete
dumps. Legacy page outputs are upgraded to the new format on a clustering rerun.
Older per-page document projections are no longer read or generated; existing
files are left untouched.

Default INFO logging reports the storage listing, loaded input paths and counts,
current page/document, candidate groups and disambiguation model, skip/rebuild
decisions, and saved dump path. Use `--log-level WARNING` for quieter runs.

The entity-linking CLI accepts both new dump scopes, plus legacy page outputs.
Select the same `--scope page|document` used for clustering. It retains cluster
IDs and source member provenance in its link journals.

## NER entity-linking decisions

The `people-gator-ner-entity-linking` command reads the output of
`people-gator-ner-clusters`. Page scope remains the default. For every resolved
source cluster it searches the
People Gator API with repeated `query.text` parameters, fetches full details for
each returned entity, and asks the configured OpenRouter model whether to merge
with one candidate or create a person. Short and long descriptions are rendered
in Czech (`cs`), including when database descriptions are in another language.

To link entire document clusters, run:

```bash
uv run people-gator-ner-entity-linking --scope document
```

Document scope defaults to input prefix `document-named-entities-clusters` and
output prefix `document-named-entities-links`. It reads
`<library>/<document-id>.json` and writes one link journal at the same relative
key. Override either prefix explicitly when needed. Use `--max-documents 1` to
process one whole document; `--max-pages` is restricted to page scope.

Each document cluster receives one shared linking decision. The journal's
`links` list records `cluster_id`, `selected_entity_id`, the combined source
cluster, all `members` with their exact page/chunk provenance, and mutation audit
records. Every member mention uses that selected entity and its original page's
source ID. The journal caches those lookups in `page_source_ids`; equal mention
text on different pages remains distinct evidence. The summary counts each
cluster once and reports document counts alongside the number of covered pages.

By default the decision is applied: existing entities are patched, new entities
are posted, member page sources are resolved through `/api/v1/pages/lookup`, and
all extractive text mentions are uploaded. Page scope retains its existing
per-page journal paths. Each page/document JSON is a durable audit journal
containing the source evidence, candidates, LLM request/response,
decision, original and resulting entity states, and every created mention. It is
checkpointed after each API mutation. The `modifications.jsonl` file under the
output prefix is a cumulative record of each applied operation with its
before and after states. On S3, each checkpoint rewrites this file to avoid
preserving invalid object metadata during append.
If an older S3 journal already has an oversized `aws-chunked` header, startup
reads and validates its records with a fallback S3 client, then rewrites the
same content with clean metadata before resuming.

When page lookup returns 404, the pipeline registers the missing page through
`POST /api/v1/pages` and uses the returned `source_id` for its mentions. A
concurrent-registration 409 is resolved by repeating the lookup.

If a live run is interrupted, rerun the same command with the same output
prefix. Rerun the same command with the same input/output prefixes and mode to resume
after an interruption. Completed pages/documents are skipped, saved linking
decisions are reused, and apply mode continues only the remaining entity/mention
mutations. Resume also works in decision-only mode. Document checkpoints retain
page lookups and each completed member mention, so recovery does not repeat
completed page uploads or entity creation. The logs report resumed checkpoints
and skipped inputs. Work still in flight that had not reached a checkpoint may
repeat. Do not use `--force` for recovery.

Document journals retain a digest of the actual clustered evidence, ignoring
clustering run IDs and LLM audit metadata. If that evidence changes after linking
starts, the command refuses to reuse the journal: use a new output prefix, or
roll back the existing mutations and rerun with `--force`. Existing journals
from another mode are skipped. For a fresh run after `--decision-only`, use a
new output prefix or `--force`; a decision-only journal has no mutations to undo.

The overall ETA counts selected pages in page scope and selected documents in
document scope; completed outputs and inputs beyond the configured limit are
excluded. Use
`--parallel-entities 4` to keep up to four entity searches and LLM decisions
in flight across page/document files (the default). Results are committed in input order. API access
is serialized. In-flight decisions that include an edited candidate are
cancelled and restarted; every candidate list changed by an earlier mutation
is sent back to the model before it is applied. The
`--parallel-pages` and `--parallel-requests` options remain aliases.
Use `--progress-interval 5` for updates every five seconds (default: 30 seconds).
INFO logs identify loaded dump paths and page/cluster/record counts, each cluster
being resolved, candidate searches and detail fetches, the decision and selected
entity, page-source lookups, mention uploads and saved journal paths. The CLI
suppresses `httpx` and `httpcore` request logs. Use `--log-level WARNING` for
quieter output.

```bash
PEOPLE_GATOR_API_BEARER_TOKEN=... OPENROUTER_API_KEY=... \
  uv run people-gator-ner-entity-linking
```

Instead of an existing bearer token, set `PEOPLE_GATOR_API_USERNAME` and
`PEOPLE_GATOR_API_PASSWORD` to use the OAuth2 password flow. Apply mode requests
`browse`, `annotate`, and `ingest` scopes.
The linking pipeline and rollback retain the login refresh token and automatically
refresh on HTTP 401, retrying the rejected request once. When supplying an existing
bearer token, also supply its paired refresh token with `--refresh-token` or
`PEOPLE_GATOR_API_REFRESH_TOKEN` to enable automatic refresh.

Use `--decision-only` to retain the old non-mutating behavior. To reverse the
newest journaled page/document (mentions first, then entities in reverse order), use:

```bash
uv run people-gator-ner-entity-linking --rollback-latest
# For document-scoped journals:
uv run people-gator-ner-entity-linking --scope document --rollback-latest
```

Rollback requires `admin` in addition to browse/annotate permissions. It checks
that an entity still matches the journaled post-change state before restoring or
deleting it; `--force-rollback` overrides that conflict check.
