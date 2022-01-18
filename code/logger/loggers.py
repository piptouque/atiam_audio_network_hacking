import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config=None, default_level=logging.INFO):
    """
    Setup logging configuration
    """
    if log_config is not None:
        # modify logging paths based on run config
        for _, handler in log_config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(log_config)
    else:
        print("Warning: logging configuration file was not given.")
        logging.basicConfig(level=default_level)
