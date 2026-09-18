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

## LLM NER CLI

The `people-gator-llm-ner` command extracts named entities from PeopleGator
page-layout JSON with an OpenRouter model. Input is expected at
`<library>/<document-id>.layouts/<page-id>.json`; output is written as
`<library>/<document-id>/<page-id>.json`. Existing outputs are skipped so an
interrupted corpus run can be resumed.

Set `OPENROUTER_API_KEY` (or place it in `.env`), then run:

```bash
uv run people-gator-llm-ner \
    --input-dir /path/to/people_gator__data \
    --output-dir peoplegator__data__ners
```

Use `--max-pages 1` for a one-page smoke test. See
`uv run people-gator-llm-ner --help` for model, chunking, reasoning, and
overwrite options.
