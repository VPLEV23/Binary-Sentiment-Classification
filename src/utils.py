import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_root_dir():
    """Returns the root directory of the project."""
    # Assuming this file is in the 'src' directory and the root is one level up
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(relative_path):
    """Constructs a path to the data directory from a relative path."""
    return os.path.join(get_root_dir(), 'data', relative_path)

def get_output_path(relative_path):
    """Constructs a path to the outputs directory from a relative path."""
    return os.path.join(get_root_dir(), 'outputs', relative_path)

def get_logger(name):
    """Returns a logger with the given name."""
    return logging.getLogger(name)
