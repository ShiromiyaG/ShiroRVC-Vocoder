"""Filesystem layout, derived from this file's location rather than the
working directory."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: One folder per run.
LOGS_DIR = os.path.join(ROOT, "logs")
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "44100.json")
