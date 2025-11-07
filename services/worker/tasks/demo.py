from __future__ import annotations

from celery import shared_task


@shared_task(name="services.worker.tasks.demo.echo")
def echo(value: str) -> str:
    """Simple echo task used for smoke tests."""

    return value

