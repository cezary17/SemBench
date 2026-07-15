"""Optional client for recording logical-query energy boundaries."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Optional

ENV_MEASUREMENT_URL = "SEMBENCH_MEASUREMENT_URL"
ENV_MEASUREMENT_RUN_ID = "SEMBENCH_RUN_ID"
DEFAULT_TIMEOUT_SECONDS = 60.0


class QueryMeasurementError(RuntimeError):
    """Raised when a configured query boundary cannot be recorded."""


class QueryMeasurementClient:
    """Send synchronous query lifecycle events to the measurement process."""

    def __init__(
        self,
        base_url: str,
        run_id: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if not base_url:
            raise ValueError("measurement base URL cannot be empty")
        if not run_id:
            raise ValueError("measurement run ID cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("measurement timeout must be positive")

        self.base_url = base_url.rstrip("/")
        self.run_id = run_id
        self.timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_env(cls) -> Optional["QueryMeasurementClient"]:
        base_url = os.environ.get(ENV_MEASUREMENT_URL)
        run_id = os.environ.get(ENV_MEASUREMENT_RUN_ID)
        if base_url is None and run_id is None:
            return None
        if not base_url or not run_id:
            raise QueryMeasurementError(
                f"{ENV_MEASUREMENT_URL} and {ENV_MEASUREMENT_RUN_ID} "
                "must be configured together"
            )
        return cls(base_url, run_id)

    def start_query(
        self,
        *,
        system: str,
        use_case: str,
        query_id: object,
        query_ordinal: int,
    ) -> "QueryMeasurementBoundary":
        query_execution_id = str(uuid.uuid4())
        self._post(
            "/queries/start",
            {
                "query_execution_id": query_execution_id,
                "run_id": self.run_id,
                "system": system,
                "use_case": use_case,
                "query_id": str(query_id),
                "query_ordinal": query_ordinal,
            },
        )
        return QueryMeasurementBoundary(
            client=self,
            query_execution_id=query_execution_id,
            work_started_at_unix_ns=time.time_ns(),
            work_started_at_monotonic_ns=time.monotonic_ns(),
        )

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise QueryMeasurementError(
                f"measurement endpoint returned HTTP {exc.code}: {body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise QueryMeasurementError(
                f"measurement endpoint request failed: {exc}"
            ) from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise QueryMeasurementError(
                f"measurement endpoint returned invalid JSON: {body}"
            ) from exc
        if not isinstance(result, dict):
            raise QueryMeasurementError(
                "measurement endpoint response must be a JSON object"
            )
        return result


@dataclass
class QueryMeasurementBoundary:
    """A started logical query awaiting its end boundary."""

    client: QueryMeasurementClient
    query_execution_id: str
    work_started_at_unix_ns: int
    work_started_at_monotonic_ns: int
    work_ended_at_unix_ns: Optional[int] = None
    work_ended_at_monotonic_ns: Optional[int] = None
    finished: bool = False

    def mark_work_ended(self) -> None:
        if self.work_ended_at_monotonic_ns is not None:
            return
        self.work_ended_at_unix_ns = time.time_ns()
        self.work_ended_at_monotonic_ns = time.monotonic_ns()

    def finish(
        self,
        *,
        status: str,
        error_type: Optional[str] = None,
    ) -> None:
        if self.finished:
            raise QueryMeasurementError(
                f"query boundary {self.query_execution_id} was already finished"
            )
        if status not in {"completed", "failed"}:
            raise ValueError("query status must be completed or failed")

        self.mark_work_ended()
        payload = {
            "query_execution_id": self.query_execution_id,
            "status": status,
            "work_started_at_unix_ns": self.work_started_at_unix_ns,
            "work_started_at_monotonic_ns": self.work_started_at_monotonic_ns,
            "work_ended_at_unix_ns": self.work_ended_at_unix_ns,
            "work_ended_at_monotonic_ns": self.work_ended_at_monotonic_ns,
        }
        if error_type:
            payload["error_type"] = error_type

        self.client._post("/queries/finish", payload)
        self.finished = True
