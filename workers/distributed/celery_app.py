"""
Celery app configuration for distributed processing in Code Review Tool.
"""

import os
from celery import Celery

# Get Redis configuration from environment or use defaults
REDIS_HOST = os.environ.get('CODEREVIEW_REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('CODEREVIEW_REDIS_PORT', '6379')
REDIS_DB = os.environ.get('CODEREVIEW_REDIS_DB', '0')
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Create Celery app
app = Celery('code_review_tool',
             broker=REDIS_URL,
             backend=REDIS_URL,
             include=['workers.distributed.tasks'])

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    enable_utc=True,
    task_time_limit=3600,  # 1 hour time limit for tasks
    task_soft_time_limit=3000,  # 50 minutes soft time limit
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_prefetch_multiplier=1,  # Don't prefetch tasks (one at a time)
    task_acks_late=True,  # Only acknowledge tasks after they're completed
    task_reject_on_worker_lost=True,  # Requeue tasks if worker dies
    broker_transport_options={'visibility_timeout': 3600},  # 1 hour visibility timeout
)

# Start Celery programmatically if this module is run directly
if __name__ == '__main__':
    app.start()