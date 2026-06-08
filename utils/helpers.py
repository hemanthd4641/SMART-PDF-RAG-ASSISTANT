import logging
import time
from typing import Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with the specified name."""
    return logging.getLogger(name)

def time_it(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and log execution time of a function."""
    logger = get_logger("timer")
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Function {func.__name__} completed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def format_file_size(size_in_bytes: int) -> str:
    """Formats raw file size in bytes into human-readable string (KB, MB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"
