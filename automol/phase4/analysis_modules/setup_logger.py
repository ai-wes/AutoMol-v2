import logging
import sys

def setup_logger(log_file='biomolecular_analysis.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to handlers
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers to logger if they haven't been added yet
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger