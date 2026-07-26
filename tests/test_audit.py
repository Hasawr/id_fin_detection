import threading

import pytest

from demos.audit_dashboard import resolve_saved_path
from shared.audit import AuditStore


def test_audit_store_uses_wal_and_prunes_payload_quota(tmp_path) -> None:
    store = AuditStore(
        tmp_path / "audit.db",
        payload_dir=tmp_path / "audit_payloads",
    )
    payload_directory = store.allocate_payload_dir()
    (payload_directory / "id.png").write_bytes(b"payload")
    relative = payload_directory.relative_to(store.payload_dir.parent)
    store.record(
        method="POST",
        path="/v1/id-fin",
        service="id-fin",
        status_code=200,
        latency_ms=1,
        payload_dir=str(relative),
    )

    with store._connect() as connection:
        journal_mode = connection.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]
    result = store.prune(retention_days=30, max_payload_bytes=0)

    assert str(journal_mode).lower() == "wal"
    assert result["payload_directories"] == 1
    assert not payload_directory.exists()
    assert store.recent_events(hours=None)[0].payload_dir is None


def test_audit_store_handles_concurrent_sqlite_writes(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")

    def write_event(index: int) -> None:
        store.record(
            method="POST",
            path="/v1/id-fin",
            service="id-fin",
            status_code=200,
            latency_ms=float(index),
        )

    threads = [
        threading.Thread(target=write_event, args=(index,))
        for index in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.summary(hours=None)["total"] == 10


def test_audit_store_prunes_expired_events_and_payloads(tmp_path) -> None:
    store = AuditStore(
        tmp_path / "audit.db",
        payload_dir=tmp_path / "audit_payloads",
    )
    payload_directory = store.allocate_payload_dir()
    (payload_directory / "id.png").write_bytes(b"payload")
    relative = payload_directory.relative_to(store.payload_dir.parent)
    event_id = store.record(
        method="POST",
        path="/v1/id-fin",
        service="id-fin",
        status_code=200,
        latency_ms=1,
        payload_dir=str(relative),
    )
    with store._connect() as connection:
        connection.execute(
            """
            UPDATE api_audit_events
            SET created_at = '2000-01-01T00:00:00+00:00'
            WHERE id = ?
            """,
            (event_id,),
        )
        connection.commit()

    result = store.prune(retention_days=30, max_payload_bytes=1024)

    assert result == {"events": 1, "payload_directories": 1}
    assert store.summary(hours=None)["total"] == 0
    assert not payload_directory.exists()


def test_dashboard_blocks_paths_outside_payload_root(tmp_path) -> None:
    store = AuditStore(
        tmp_path / "audit.db",
        payload_dir=tmp_path / "audit_payloads",
    )
    with pytest.raises(ValueError, match="outside"):
        resolve_saved_path(str(tmp_path / "secret.txt"), store)
    with pytest.raises(ValueError, match="outside"):
        resolve_saved_path("../secret.txt", store)


@pytest.mark.parametrize(
    ("latencies", "expected_median"),
    [
        ([10.0, 20.0, 30.0], 20.0),          # odd count
        ([10.0, 20.0, 30.0, 40.0], 25.0),    # even count averages the middle
        ([5.0], 5.0),                        # single row
    ],
)
def test_summary_median_latency(tmp_path, latencies, expected_median) -> None:
    """Median, not mean, describes a typical call.

    A few large batch requests pull the mean far above what one call costs,
    so the dashboard leads with the median.
    """
    store = AuditStore(tmp_path / "audit.db")
    for latency in latencies:
        store.record(
            method="POST",
            path="/v1/id-fin",
            service="id-fin",
            status_code=200,
            latency_ms=latency,
        )

    summary = store.summary(hours=None)

    assert summary["median_latency_ms"] == pytest.approx(expected_median)
    assert summary["avg_latency_ms"] == pytest.approx(
        sum(latencies) / len(latencies)
    )


def test_summary_median_latency_is_empty_safe(tmp_path) -> None:
    summary = AuditStore(tmp_path / "audit.db").summary(hours=None)
    assert summary["median_latency_ms"] == 0.0
