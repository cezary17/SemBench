import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runner.measurement_client import (  # noqa: E402
    ENV_MEASUREMENT_RUN_ID,
    ENV_MEASUREMENT_URL,
    QueryMeasurementClient,
    QueryMeasurementError,
)


class RecordingHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.requests.append((self.path, payload))
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class QueryMeasurementClientTest(unittest.TestCase):
    def setUp(self):
        RecordingHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_records_start_and_finish_with_text_query_id(self):
        client = QueryMeasurementClient(self.base_url, "run-1")
        boundary = client.start_query(
            system="lotus",
            use_case="mmqa",
            query_id="2a",
            query_ordinal=3,
        )
        boundary.finish(status="failed", error_type="RuntimeError")

        self.assertEqual(len(RecordingHandler.requests), 2)
        start_path, start = RecordingHandler.requests[0]
        finish_path, finish = RecordingHandler.requests[1]
        self.assertEqual(start_path, "/queries/start")
        self.assertEqual(start["run_id"], "run-1")
        self.assertEqual(start["query_id"], "2a")
        self.assertEqual(start["query_ordinal"], 3)
        self.assertEqual(finish_path, "/queries/finish")
        self.assertEqual(finish["status"], "failed")
        self.assertEqual(finish["error_type"], "RuntimeError")
        self.assertEqual(finish["query_execution_id"], start["query_execution_id"])
        self.assertLessEqual(
            finish["work_started_at_monotonic_ns"],
            finish["work_ended_at_monotonic_ns"],
        )

    def test_environment_configuration_is_optional_but_atomic(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(QueryMeasurementClient.from_env())

        with patch.dict(
            os.environ,
            {ENV_MEASUREMENT_URL: self.base_url},
            clear=True,
        ):
            with self.assertRaises(QueryMeasurementError):
                QueryMeasurementClient.from_env()

        with patch.dict(
            os.environ,
            {
                ENV_MEASUREMENT_URL: self.base_url,
                ENV_MEASUREMENT_RUN_ID: "run-1",
            },
            clear=True,
        ):
            client = QueryMeasurementClient.from_env()
            self.assertIsNotNone(client)
            self.assertEqual(client.run_id, "run-1")


if __name__ == "__main__":
    unittest.main()
