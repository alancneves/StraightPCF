#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from datetime import datetime


class LogHandler(logging.StreamHandler):
    """
    Defines the logging format class
    """

    colors = {
        logging.DEBUG: '\033[42m',
        logging.INFO: '\033[32m',
        logging.WARNING: '\033[33m',
        logging.ERROR: '\033[31m',
        logging.CRITICAL: '\033[101m',
    }
    reset = '\033[0m'
    fmtr = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record):
        color = self.colors[record.levelno]
        log = self.fmtr.format(record)
        return color + log + self.reset


class Logger:

    def __init__(self, logName, logFileName=None, level=logging.INFO, generate=False) -> None:
        self.logName = logName
        if logFileName is None and generate:
            date_fmt = datetime.now().strftime('%Y%m%d_%H%M')
            logFileName = f"{logName}_{date_fmt}.log"
        self.logFile = logFileName
        self.configure(level)

    def configure(self, level=logging.INFO):
        handlers = []

        # Create logger with loggerStrName
        logger = logging.getLogger(self.logName)
        logger.setLevel(level)
        logger.propagate = False

        # Create console handler with higher log level
        ch = logging.StreamHandler()
        ch.setFormatter(LogHandler())        
        handlers.append(ch)

        # Create time rotating file handler
        if self.logFile is not None:
            # create log directory if dos not exist
            Path(self.logFile).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(self.logFile, mode='a', encoding='utf8')
            fh.setFormatter(logging.Formatter(
                "[%(levelname)s][%(asctime)s] %(message)s"))
            handlers.append(fh)

        # Add each handler
        for h in handlers:
            logger.addHandler(h)

        self.logger = logger

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

