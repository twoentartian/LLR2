from pathlib import Path
import logging
import random
from collections import deque
from typing import Optional

import numpy as np
import torch

def basename_without_extension(name: str) -> str:
    return Path(name).stem

def re_initialize_model(model: torch.nn.Module):
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters() # type: ignore

def setup_logging(
    target_logger: logging.Logger,
    tag: str,
    log_file_path: Optional[str | Path] = None,
    exit_on_critical: bool = False,
) -> None:
    formatter = logging.Formatter(
        f"[%(asctime)s {tag}] %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handlers = [
        handler
        for handler in target_logger.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]
    has_stream_handler = len(stream_handlers) > 0
    for handler in stream_handlers:
        handler.setFormatter(formatter)
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        target_logger.addHandler(stream_handler)

    if log_file_path is not None:
        file_path = Path(log_file_path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        matching_file_handlers = [
            handler
            for handler in target_logger.handlers
            if isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).expanduser().resolve() == file_path
        ]
        has_file_handler = len(matching_file_handlers) > 0
        for handler in matching_file_handlers:
            handler.setFormatter(formatter)
        if not has_file_handler:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            target_logger.addHandler(file_handler)

    target_logger.setLevel(logging.INFO)
    target_logger.propagate = False

    if exit_on_critical and not getattr(target_logger, "_critical_exits_process", False):
        original_critical = target_logger.critical

        def critical_and_exit(msg, *args, **kwargs):
            original_critical(msg, *args, **kwargs)
            raise SystemExit(1)

        target_logger.critical = critical_and_exit  # type: ignore[method-assign]
        target_logger._critical_exits_process = True  # type: ignore[attr-defined]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def expand_path(p):
    # convert to Path and expand ~, then resolve to absolute
    return Path(str(p)).expanduser().resolve()

def prompt_selection(options, prompt_message="Please make a selection:", allow_quit=True):
    """
    Display a list of options and prompt user to make one selection.

    Args:
        options (list): List of strings to choose from
        prompt_message (str): Custom prompt message
        allow_quit (bool): Whether to allow 'q' to quit

    Returns:
        str: Selected option, or None if user quits
    """
    if not options:
        print("No options provided.")
        return None

    while True:
        print(f"\n{prompt_message}")
        print("-" * len(prompt_message))

        # Display numbered options
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        if allow_quit:
            print("q. Quit")

        # Get user input
        choice = input(f"\nEnter your choice (1-{len(options)}" + ("or 'q'" if allow_quit else "") + "): ").strip().lower()

        # Handle quit
        if allow_quit and choice in ['q', 'quit']:
            return None

        # Handle numeric selection
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                selected = options[choice_num - 1]
                print(f"\nYou selected: {selected}")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def geodesic_distance(a: torch.Tensor, b: torch.Tensor) -> Optional[torch.Tensor]:
    if not a.dtype.is_floating_point or not b.dtype.is_floating_point:
        return None
    a_flat = a.flatten()
    b_flat = b.flatten()
    na = torch.norm(a_flat)
    nb = torch.norm(b_flat)
    if na == 0 or nb == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=a.device)
    radius = (na + nb) / 2
    cos_theta = torch.clamp(torch.dot(a_flat, b_flat) / (na * nb), -1.0, 1.0)
    return radius * torch.acos(cos_theta)

class MovingAverage:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.sum = 0.0

    def add(self, value):
        if len(self.window) == self.window_size:
            self.sum -= self.window[0]
        self.window.append(value)
        self.sum += value
        return self.get_average()

    def get_average(self):
        if not self.window:
            return 0
        return self.sum / len(self.window)

class MovingMax:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.window = deque()
        self.max_deque = deque()

    def add(self, value):
        self.window.append(value)
        while self.max_deque and self.max_deque[-1] < value:
            self.max_deque.pop()
        self.max_deque.append(value)
        if len(self.window) > self.window_size:
            oldest = self.window.popleft()
            if oldest == self.max_deque[0]:
                self.max_deque.popleft()
        return self.get_max()

    def get_max(self):
        if not self.max_deque:
            return None
        return self.max_deque[0]
    
