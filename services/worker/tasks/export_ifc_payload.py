from __future__ import annotations

import asyncio
from uuid import UUID

from celery import shared_task

from services.api.ifc_job_runner import process_ifc_job_async


@shared_task(name="services.worker.tasks.export_ifc_payload")
def export_ifc_payload(job_id: str) -> str:
    asyncio.run(process_ifc_job_async(UUID(job_id)))
    return "processed"




