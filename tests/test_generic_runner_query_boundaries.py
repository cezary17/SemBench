import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runner.generic_runner import GenericQueryMetric, GenericRunner  # noqa: E402


class RecordingBoundary:
    def __init__(self, record):
        self.record = record

    def mark_work_ended(self):
        self.record["work_ended"] = True

    def finish(self, *, status, error_type=None):
        self.record["finish_status"] = status
        self.record["error_type"] = error_type


class RecordingClient:
    def __init__(self):
        self.records = []

    def start_query(self, **kwargs):
        record = dict(kwargs)
        self.records.append(record)
        return RecordingBoundary(record)


class DummyRunner(GenericRunner):
    def __init__(self):
        super().__init__(
            use_case="movie",
            scale_factor=1,
            model_name="dummy",
            concurrent_llm_worker=1,
            skip_setup=True,
        )

    def get_system_name(self):
        return "lotus"

    def execute_query(self, query_id):
        if query_id == 2:
            raise RuntimeError("query failed")
        return GenericQueryMetric(
            query_id=query_id,
            status="success",
            execution_time=0.1,
            results=pd.DataFrame({"id": [1]}),
        )


class GenericRunnerQueryBoundaryTest(unittest.TestCase):
    def test_shared_query_loop_records_success_and_failure(self):
        with patch.dict(os.environ, {}, clear=True):
            runner = DummyRunner()
        client = RecordingClient()
        runner.query_measurement_client = client

        results = runner.execute_queries([1, 2])

        self.assertEqual(results[1].status, "success")
        self.assertEqual(results[2].status, "failed")
        self.assertEqual(len(client.records), 2)
        self.assertEqual(client.records[0]["query_ordinal"], 1)
        self.assertEqual(client.records[0]["finish_status"], "completed")
        self.assertEqual(client.records[1]["query_ordinal"], 2)
        self.assertEqual(client.records[1]["finish_status"], "failed")
        self.assertEqual(client.records[1]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
