import time
from typing import Any, Callable, Tuple


def measure_time(func: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure execution time of a function.
    Returns:
    - function result
    - elapsed time in seconds
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()

    return result, end - start


def print_search_results(
    query: str,
    results: list[str],
    distances: list[float],
) -> None:
    """
    Pretty-print FAISS search results.
    """
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    for i, (text, dist) in enumerate(zip(results, distances), start=1):
        print(f"{i:02d}. Distance={dist:.4f} | {text}")
