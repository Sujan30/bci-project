"""ARQ WorkerSettings — configuration for the Sleep-BCI worker process.

Start the worker with:
    arq sleep_bci.workers.settings.WorkerSettings

Or via Docker Compose:
    docker-compose up worker
"""
import os

from arq.connections import RedisSettings

from sleep_bci.workers.tasks import preprocess_task, train_task


class WorkerSettings:
    """ARQ worker configuration.

    All long-running jobs (preprocessing, training) execute here,
    isolated from the API process for fault tolerance and scalability.
    """

    functions = [preprocess_task, train_task]

    redis_settings = RedisSettings.from_dsn(
        os.environ.get("REDIS_URL", "redis://localhost:6379")
    )

    # Limit concurrent jobs to avoid memory pressure from large NPZ loads
    max_jobs = 4

    # Allow up to 1 hour per job (preprocessing large EDF files can take time)
    job_timeout = 3600

    # Retry failed jobs once before marking permanently failed
    max_tries = 2
