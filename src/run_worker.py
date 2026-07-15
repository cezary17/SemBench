"""
Subprocess worker for running a single system in an isolated environment.

This script is invoked by run.py when per-system virtual environments are
detected. It runs a single system's queries and saves results to disk.
The main run.py process then reads those results for evaluation.

Usage (called automatically by run.py):
    .venvs/lotus/bin/python src/run_worker.py \
        --system lotus --use-case movie --queries 1 3 \
        --model gemini-2.5-flash --scale-factor 2000
"""

import argparse
import json
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run a single system's queries in an isolated environment"
    )
    parser.add_argument(
        "--system", required=True, help="System name (e.g., lotus, palimpzest)"
    )
    parser.add_argument(
        "--use-case", required=True, help="Use case name (e.g., movie, ecomm)"
    )
    parser.add_argument(
        "--queries", nargs="+", default=None, help="Query IDs to run"
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash", help="Model name"
    )
    parser.add_argument(
        "--scale-factor", type=int, default=None, help="Dataset scale factor"
    )
    parser.add_argument(
        "--skip-setup", action="store_true", help="Skip data setup phase"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["default", "local"],
        default="default",
        help="LLM provider mode",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for local provider",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for local provider",
    )
    parser.add_argument(
        "--llm-params",
        type=str,
        default="{}",
        help="JSON object of model parameters for local provider",
    )

    args = parser.parse_args()

    from runner.llm_provider_config import (
        llm_provider_config_from_args,
        set_llm_provider_env,
        validate_no_bigquery_local,
    )

    try:
        llm_config = llm_provider_config_from_args(args)
        validate_no_bigquery_local([args.system], llm_config)
        set_llm_provider_env(llm_config)
    except ValueError as e:
        print(f"\n__WORKER_ERROR__{str(e)}__END_WORKER_ERROR__")
        sys.exit(1)

    # Import runner infrastructure
    from run import get_runner_class, parse_query_ids

    # Parse query IDs
    query_ids = None
    if args.queries:
        query_ids = parse_query_ids(args.queries)

    # Get runner class
    runner_class = get_runner_class(args.system, args.use_case)
    if not runner_class:
        print(f"Failed to import runner for {args.system}")
        sys.exit(1)

    # Initialize and run
    try:
        runner = runner_class(
            use_case=args.use_case,
            scale_factor=args.scale_factor,
            skip_setup=args.skip_setup,
            model_name=args.model,
        )
        metrics = runner.run_all_queries(queries=query_ids)

        # Write a completion marker with summary for the parent process
        summary = {}
        for query_id, metric in metrics.items():
            summary[f"Q{query_id}"] = runner.metric_to_dict(metric)

        # Print summary as JSON to stdout for parent process.
        print(f"\n__WORKER_RESULT__{json.dumps(summary)}__END_WORKER_RESULT__")

        succeeded = bool(metrics) and all(
            getattr(metric, "status", None) == "success"
            for metric in metrics.values()
        )
        return 0 if succeeded else 1

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\n__WORKER_ERROR__{str(e)}__END_WORKER_ERROR__")
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
