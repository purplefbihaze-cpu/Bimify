from celery import Celery

from core.settings import get_settings


def create_app() -> Celery:
    settings = get_settings()
    celery_app = Celery(
        "bimify-worker",
        broker=settings.queue.broker_url,
        backend=settings.queue.result_backend,
    )
    celery_app.conf.task_default_queue = "bimify"
    celery_app.conf.task_routes = {
        "services.worker.tasks.*": {"queue": "bimify"},
    }
    return celery_app


app = create_app()


@app.task(name="services.worker.tasks.health")
def health() -> str:
    return "ok"


__all__ = ["app", "create_app", "health"]

