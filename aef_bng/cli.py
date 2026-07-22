"""CLI entry point for AEF-BNG processing."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

import click

from aef_bng.config import AEFBNGConfig


@contextmanager
def stopwatch():
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()

    total_seconds = int(end_time - start_time)

    # Calculate HH:MM:SS
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"
    click.secho(f"\nDuration: {timestamp}", fg="cyan", bold=True)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """AEF embeddings on British National Grid."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


@main.command(name="spark-run")
@click.option("--bounds", required=True)
@click.option("--years", required=True)
@click.option("--table-name", required=True)
@click.option("--resampling", default="nearest")
def spark_run(bounds: str, years: str, table_name: str, resampling: str) -> None:
    """Execute pipeline on-cluster (called by python_wheel_task)."""
    config = AEFBNGConfig(
        years=[int(y) for y in years.strip().strip(",").split(",")],
        bounds=tuple(int(b) for b in bounds.strip().strip(",").split(",")),  # type: ignore[arg-type]
        table_name=table_name,
        resampling=resampling,
    )

    with stopwatch():
        from aef_bng.spark import process_with_spark

        process_with_spark(config)


def entrypoint() -> None:
    """Entry point for console script.

    Catches SystemExit(0) for Databricks python_wheel_task compatibility —
    Databricks treats any SystemExit as a task failure, but Click calls
    sys.exit(0) on success.
    """
    try:
        main()
    except SystemExit as e:
        if e.code != 0:
            raise


if __name__ == "__main__":
    main()
