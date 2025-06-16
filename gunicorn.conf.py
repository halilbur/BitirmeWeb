# Gunicorn configuration for Render.com deployment
import multiprocessing
import os

# Basic settings
bind = "0.0.0.0:8000"
workers = 1  # Single worker for free tier to avoid memory issues
worker_class = "sync"
worker_connections = 1000

# Timeout settings - crucial for ML model loading
timeout = 120  # 2 minutes
keepalive = 5
graceful_timeout = 120

# Memory management
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 50  # Add randomness to prevent all workers restarting at once
preload_app = True  # Load app before forking workers

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "flask-ml-app"

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
worker_tmp_dir = "/dev/shm"  # Use RAM for temporary files if available

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)
    # Preload models after fork to avoid cold starts
    try:
        from src.main import preload_models
        preload_models()
    except Exception as e:
        server.log.error("Failed to preload models: %s", e)
