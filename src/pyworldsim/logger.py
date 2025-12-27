import logging
import sys
import os
from datetime import datetime

def setup_logger(log_dir: str = "logs", log_name: str = "wfrp_run") -> logging.Logger:
    """
    Sets up a logger that writes to:
    1. A timestamped file in the 'logs' directory.
    2. The Console (Standard Output).
    """
    # 1. Create Log Directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. Generate Timestamped Filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(log_dir, f"{log_name}_{timestamp}.log")

    # 3. Configure the Root Logger
    # We use the root logger so all imported modules inherit these settings
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate logs if run multiple times in Jupyter
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- File Handler (Detailed) ---
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- Console Handler (Clean) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # Console logs don't need timestamps, just the message
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Writing to: {filename}")
    return logger