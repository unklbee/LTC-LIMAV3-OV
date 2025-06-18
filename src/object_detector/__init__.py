"""Package initialization for object_detector."""

import logging
from .config import Config

# Configure root logging once using configurable level.
logging.basicConfig(level=Config.get_log_level())
