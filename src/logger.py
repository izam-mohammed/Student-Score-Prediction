import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_file = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(logs_file, exist_ok = True)

LOG_FILE_PATH = os.path.join(logs_file, LOG_FILE) 

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[%(asctime)s ] %(lneno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
    
)