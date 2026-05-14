from __future__ import annotations

from people_gator_common.pipelines.wikicommons import run_wikicommons_images_download_pipeline
from people_gator_common.pipelines.wikidata import run_wikidata_pipeline

__all__ = [
    "run_wikicommons_images_download_pipeline",
    "run_wikidata_pipeline",
]
