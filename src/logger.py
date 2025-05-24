import logging
import os
from datetime import datetime

logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok =True)

# the fie_name looks like - log_2025-02-21
log_file = os.path.join(logs_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# info, warning and error are the message types
logging.basicConfig(
    filename = log_file,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    level = logging.INFO # only info, warning and error messages are shown here

)

def get_logger(name):
    logger =  logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
