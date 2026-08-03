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
