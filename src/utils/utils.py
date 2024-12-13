"""Module to track the time or progress in our model using a progress bar and formatting
time, som utilities here"."""

# Standard libraries
from typing import Callable, Optional
import sys
import time

__all__ = [
    "progress_bar",
    "logger",
    "format_time",
]


def progress_bar(
        current: int,
        start_time: float,
        msg: Optional[str] = None) -> None:
    """Displays a progress bar with the time taken per step and the total elapsed time.

    Args:
        current (int): The current step in the progress.
        total (int): The total number of steps.
        start_time (float): The time at which the process started.
        msg (Optional[str], optional): An additional message to display.
        Defaults to None.
    """
    cur_time = time.time()
    elapsed_time = cur_time - start_time
    step_time = elapsed_time / (current + 1)  # Avoid division by 0

    progress_message = f"Step: {format_time(step_time)} \
        | Tot: {format_time(elapsed_time)}"

    if msg:
        progress_message += f" | {msg}"

    sys.stdout.write(progress_message + '\r')
    sys.stdout.flush()


def logger(verbose: bool = True) -> Callable[[str], None]:
    """Creates a logging function that prints messages to the console if verbose is
    True.

    Args:
        verbose (bool, optional): If True, the logger will print messages.
        Defaults to True.

    Returns:
        Callable[[str], None]: A function that takes a message
        (or messages) and prints it if verbose is True.
    """
    def log(*msg: str) -> None:
        if verbose:
            print(*msg)

    return log


def format_time(seconds: float) -> str:
    """
    Converts a given time in seconds into a readable string format:
    days, hours, minutes, seconds, and milliseconds.

    Args:
        seconds (float): The time duration in seconds to format.

    Returns:
        str: The formatted time string (e.g., '1D 2h 3m 4s 500ms').
    """
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    secondsf = int(seconds)
    millis = int((seconds - secondsf) * 1000)

    time_parts = []
    if days > 0:
        time_parts.append(f"{int(days)}D")
    if hours > 0:
        time_parts.append(f"{int(hours)}h")
    if minutes > 0:
        time_parts.append(f"{int(minutes)}m")
    if secondsf > 0:
        time_parts.append(f"{secondsf}s")
    if millis > 0:
        time_parts.append(f"{millis}ms")

    return ' '.join(time_parts) if time_parts else '0ms'
