import logging
from logging import Logger
from typing import List, Tuple
import os
from colorama import Fore, Back, Style


def get_all_loggers_name() -> List[str]:
    return list(logging.Logger.manager.loggerDict.keys())

def set_all_loggers_level_to(level: int = logging.WARNING):
    for name in get_all_loggers_name():
        logging.getLogger(name).setLevel(level)


def markdown_to_text(md: str) -> Tuple[str, bool]:
    bold = False
    while len(md) > 0 and (md[0] == '#' or md[0] == '*'): # or md[0] == ' '):
        md = md[1:]
        if md[0] == '*': bold = True
    while len(md) > 0 and (md[-1] == '*'):
        md = md[:-1]
        if md[-1] == '*': bold = True
    return md, bold



def get_logger(name: str = "my_logger",
               log_file: str = None,
               console_log_level: int = logging.DEBUG,
               log_file_level: int = logging.DEBUG,
               console_format: str = None) -> Logger:
    """
    Gets or creates a new logger.
    """
    if console_format is None:
        console_format = "%(name)s: %(message)s"
        if name == "__main__": console_format = "%(message)s"

    if name in get_all_loggers_name():
        # logging.warning(f"Logger with name '{name}' already exists. Returning the logger.")
        return logging.getLogger(name)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(FromMarkdownToConsoleFormatter(console_format))

    # Create log file or empty the log file
    if log_file is not None:
        if os.path.exists(log_file): os.remove(log_file)
        f = open(log_file, "x"); f.close()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)

        if log_file[-3:] == ".md": file_handler.setFormatter(KeepMarkdownFormatter())
        else: file_handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger(name)

    # The overall level of the logger is not defined, as it assumed to be inherited form the logging.basicConfig
    # logger.setLevel(logging.DEBUG) # overall level of the logger

    # Add the two handlers (order is important since the message is getting changed for the console output)
    if log_file is not None: logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False # don't propagate to the root logger

    return logger



class KeepMarkdownFormatter(logging.Formatter):

    format = "%(message)s"

    formats = {
        logging.DEBUG:    "<span style='color:gray'>" + format + "</span><br/>",
        logging.INFO:     format + "<br/>",
        logging.WARNING:  "<span style='color:yellow'>" + format + "</span><br/>",
        logging.ERROR:    "<span style='color:red'>" + format + "</span><br/>",
        logging.CRITICAL: "**<span style='color:red'>" + format + "</span>**<br/>",
    }

    def format(self, record):

        # format message
        self._style._fmt = KeepMarkdownFormatter.formats[record.levelno]

        return super().format(record)


class FromMarkdownToConsoleFormatter(logging.Formatter):

    def __init__(self, format: str = "%(name)s: %(message)s"):

        super().__init__()
        self.formats = {
            logging.DEBUG: Fore.LIGHTBLACK_EX + format + Style.RESET_ALL,
            #logging.INFO: format + Style.RESET_ALL,
            logging.INFO: Fore.LIGHTBLUE_EX + format + Style.RESET_ALL,
            logging.WARNING: Fore.YELLOW + format + Style.RESET_ALL,
            logging.ERROR: Fore.RED + format + Style.RESET_ALL,
            logging.CRITICAL: Fore.RED + Style.BRIGHT + format + Style.RESET_ALL,
            "success": Fore.GREEN + format + Style.RESET_ALL
        }

    def format(self, record):

        # Get message and remove markdown characters
        message = str(record.msg)
        record.msg, bold = markdown_to_text(message)

        # Get format
        msg_format = self.formats[record.levelno]
        # Success
        if record.levelno == logging.INFO:
            if "success" in message.lower():
                msg_format = self.formats["success"]
        # Bold
        if bold: msg_format = Style.BRIGHT + msg_format

        # Format message
        self._style._fmt = msg_format

        return super().format(record)