import os
import logging
import datetime

class Logger:

    def __init__(self,log_floder):
        # Create a logger and set its level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.max_info = 200
        # Create a console handler and set its level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create a file handler and set its level
        if not os.path.exists(log_floder):
            os.makedirs(log_floder)
        log_filename = datetime.datetime.now().strftime('%Y-%m-%d.log')
        log_path = os.path.join(log_floder, log_filename)
        file_handler = logging.FileHandler(log_path,encoding="utf-8",mode="a")  # Create or append to the log file
        file_handler.setLevel(logging.INFO)

        # Create a formatter and attach it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Attach the handlers to the logger
        self.logger.addHandler(ch)
        self.logger.addHandler(file_handler)

    def info(self,info):
        if len(info) > self.max_info:
            self.logger.info(f'{info[:self.max_info]} ...')
        else:
            self.logger.info(info)
