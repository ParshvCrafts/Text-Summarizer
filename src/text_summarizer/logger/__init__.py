import logging
import os
import sys

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"

os.makedirs(log_dir, exist_ok=True)

# Option 1: Using only handlers (RECOMMENDED)
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "running_logs.log")),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("text_summarizer_logger")
