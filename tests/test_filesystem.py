from types import SimpleNamespace

import s3fs

import people_gator_common.filesystem as storage
from people_gator_common.settings import S3Settings, StorageSettings


def test_s3_uses_only_required_request_checksums(monkeypatch) -> None:
    options = {}

    def create_s3_filesystem(**kwargs):
        options.update(kwargs)
        return object()

    monkeypatch.setattr(storage, "fs", None)
    monkeypatch.setattr(storage, "prefix", None)
    monkeypatch.setattr(
        storage,
        "PeopleGatorSettings",
        lambda: SimpleNamespace(
            storage=StorageSettings(s3=S3Settings(enabled=True, bucket="test-bucket"))
        ),
    )
    monkeypatch.setattr(s3fs, "S3FileSystem", create_s3_filesystem)

    storage.get_fs()

    assert options["config_kwargs"]["request_checksum_calculation"] == "when_required"
