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
