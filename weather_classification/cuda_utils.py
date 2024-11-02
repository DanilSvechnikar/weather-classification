"""This file contains cuda related functions."""

from torch import cuda


def print_cuda_available() -> None:
    """Print cuda is available."""
    print("Torch cuda:", cuda.is_available())


def print_cuda_info() -> None:
    """Prints cuda info in console."""
    if cuda.is_available():
        print(cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(cuda.memory_reserved(0) / 1024**3, 1), "GB")


if __name__ == "__main__":
    print_cuda_available()
