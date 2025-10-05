# app/jobs.py
from concurrent.futures import ThreadPoolExecutor

_executor: ThreadPoolExecutor | None = None

def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        # 2 hilos son más que suficientes para este MVP
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jobs")
    return _executor

def submit(fn, *args, **kwargs):
    return get_executor().submit(fn, *args, **kwargs)

def shutdown():
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
